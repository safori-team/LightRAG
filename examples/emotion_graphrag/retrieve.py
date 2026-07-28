"""Deterministic candidate retrieval — the GraphRAG read half.

Two sources, exactly as DESIGN.md §5 specifies, but written against the
artifact layout produced by build_layer1_v2.py and with the evidence strings
the judge prompt needs:

  source 1 (graph)        get_node_edges -> get_edges_batch, keep co_occurs,
                          score with the directional P(B|A) from cooccur_stats
  source 2 (situational)  chunks_vdb.query on the utterance -> full_doc_id ->
                          labels_map -> similarity-weighted label frequency

Nothing here calls an LLM. infer.py is deliberately not imported: it binds
Ollama at module import time, which would make this path fail on an OpenAI or
Gemini configuration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lightrag import LightRAG

import providers
import rules
import schema


@dataclass
class Artifacts:
    rag: LightRAG
    labels_map: dict
    directional: dict
    build_info: dict


async def open_artifacts(artifact_dir: str | Path) -> Artifacts:
    """Open one build's storage + side tables together.

    Storage and side tables must come from the same build: the directional
    statistics reference edges that only exist in that graph.
    """
    d = Path(artifact_dir)

    def _load(name: str, default):
        p = d / name
        if not p.exists():
            return default
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    rag = LightRAG(
        working_dir=str(d / "rag_storage"),
        workspace="",
        llm_model_func=providers.noop_llm,
        embedding_func=providers.make_embedding_func(),
    )
    await rag.initialize_storages()
    return Artifacts(
        rag=rag,
        labels_map=_load("labels_map.json", {}),
        directional=_load("cooccur_stats.json", {}),
        build_info=_load("build_info.json", {}),
    )


# --- source 1: graph co_occurs neighbours -------------------------------------
async def graph_candidates(art: Artifacts, detected: set[str]) -> dict[str, dict]:
    """detected emotions -> {candidate: {w, p, lift, count, via}}.

    get_node_edges returns neighbour pairs only, so edge attributes are fetched
    in a second batched call; belongs_to (hierarchy) edges are filtered out
    because taxonomy membership is not evidence of co-occurrence.
    """
    graph = art.rag.chunk_entity_relation_graph
    out: dict[str, dict] = {}

    for emo in sorted(detected):
        pairs = await graph.get_node_edges(emo) or []
        neighbours = {t for _, t in pairs} | {s for s, _ in pairs}
        neighbours.discard(emo)
        if not neighbours:
            continue
        props = await graph.get_edges_batch(
            [{"src": emo, "tgt": n} for n in neighbours]
        )
        for (s, t), p in props.items():
            if p.get("keywords") != schema.EDGE_CO_OCCURS:
                continue
            other = t if s == emo else s
            if other in detected or not schema.is_known_sub(other):
                continue
            # The graph edge is undirected with a symmetric weight; the
            # asymmetric P(other|emo) lives in the side table (DESIGN.md §4).
            stat = art.directional.get(f"{emo}|{other}")
            weight = stat["weight"] if stat else float(p.get("weight", 0.0))
            prev = out.get(other)
            if prev is None or weight > prev["w"]:
                out[other] = {
                    "w": round(weight, 4),
                    "p": stat["p_b_given_a"] if stat else None,
                    "lift": stat["lift"] if stat else None,
                    "count": stat["count"] if stat else None,
                    "via": emo,
                }
    return out


# --- source 2: situational similarity -----------------------------------------
def _similarity(rank: int, rec: dict) -> float:
    """Backend-agnostic similarity: use an explicit score if the vector store
    reports one, otherwise decay by rank."""
    for key in ("distance", "score", "similarity"):
        v = rec.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 1.0 / (rank + 1)


async def situational_candidates(
    art: Artifacts, utterance: str, detected: set[str]
) -> tuple[dict[str, float], dict[str, dict]]:
    """Similar training utterances -> label scores + one example per candidate.

    Returns ({emotion: score}, {emotion: {"text", "sim"}}). The example is what
    the judge prompt cites, so a candidate's situational claim is auditable.
    """
    if not utterance.strip():
        return {}, {}
    recs = await art.rag.chunks_vdb.query(utterance, top_k=rules.SIT_TOPK)

    neighbours: list[tuple[float, set[str], str]] = []
    for rank, rec in enumerate(recs):
        # custom_kg chunks store source_id as full_doc_id; the vdb payload
        # keeps full_doc_id but drops source_id.
        src = rec.get("source_id") or rec.get("full_doc_id") or rec.get("id")
        entry = art.labels_map.get(src)
        if not entry:
            continue
        text = rec.get("content") or rec.get("text") or ""
        neighbours.append((_similarity(rank, rec), set(entry.get("sub", [])), text))

    pairs = [(sim, labels) for sim, labels, _ in neighbours]
    universe = {e for _, labels in pairs for e in labels} - detected
    scores = {e: rules.sit_score(pairs, e) for e in universe}

    examples: dict[str, dict] = {}
    for emo in universe:
        best = max((n for n in neighbours if emo in n[1]),
                   key=lambda n: n[0], default=None)
        if best:
            examples[emo] = {"text": best[2][:80], "sim": round(best[0], 3)}
    return scores, examples


# --- fusion -------------------------------------------------------------------
async def candidates_for(art: Artifacts, utterance: str, detected: set[str],
                         *, use_situational: bool = True) -> dict:
    """Full candidate stage for one utterance.

    Returns {"fused": [rules.Candidate], "graph": {...}, "sit": {...},
             "examples": {...}} so the caller can build evidence lines and dump
    the whole retrieval state into the per-clip result record.
    """
    graph = await graph_candidates(art, detected)
    if use_situational:
        sit, examples = await situational_candidates(art, utterance, detected)
    else:
        sit, examples = {}, {}
    fused = rules.fuse({k: v["w"] for k, v in graph.items()}, sit, detected)
    return {"fused": fused, "graph": graph, "sit": sit, "examples": examples}
