"""Realtime correction loop (reviewer): Gemini 1st-pass -> missing emotions.

Two candidate sources are fused (DESIGN.md sections 3-4):
  source 1  graph co_occurs neighbours (deterministic: get_node_edges +
            directional weight from cooccur_stats.json)
  source 2  situational similarity (embed utterance -> similar train
            utterances via chunks_vdb -> aggregate their labels)

The graph/statistics only PROPOSE candidates; the final add is gated by
``judge_fn`` (situational similarity + an LLM confirmation on the utterance).
Layer-2 personalization and Gemini itself are external and injected here.
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import partial

import ollama
from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

import rules
import schema

load_dotenv(dotenv_path=".env", override=False)

HERE = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.getenv("WORKING_DIR", os.path.join(HERE, "rag_storage"))
WORKSPACE = os.getenv("WORKSPACE", "emotion")


def _load_side(name: str) -> dict:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        workspace=WORKSPACE,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "exaone3.5:7.8b"),
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": int(os.getenv("OLLAMA_LLM_NUM_CTX", "32768"))},
            "timeout": int(os.getenv("TIMEOUT", "300")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=partial(
                ollama_embed.func,
                embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )
    await rag.initialize_storages()
    return rag


# --- source 1: graph co_occurs neighbours ------------------------------------
async def graph_candidates(rag: LightRAG, detected: set[str],
                           directional: dict) -> dict[str, float]:
    """For each detected emotion, collect co_occurs neighbours and score them
    by the directional weight P(neighbour|detected) from the side table."""
    graph = rag.chunk_entity_relation_graph
    weights: dict[str, float] = {}
    for emo in detected:
        pairs = await graph.get_node_edges(emo) or []
        neighbours = {t for (s, t) in pairs} | {s for (s, t) in pairs}
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
            # prefer directional P(other|emo); fall back to undirected weight
            d = directional.get(f"{emo}|{other}")
            w = d["weight"] if d else float(p.get("weight", 0.0))
            weights[other] = max(weights.get(other, 0.0), w)
    return weights


# --- source 2: situational similarity ----------------------------------------
def _similarity(rank: int, rec: dict) -> float:
    """Similarity weight for a retrieved chunk. Prefer an explicit score field;
    otherwise fall back to rank decay (backend-agnostic)."""
    for key in ("distance", "score", "similarity"):
        v = rec.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 1.0 / (rank + 1)


async def situational_candidates(rag: LightRAG, utterance: str,
                                 labels_map: dict) -> dict[str, float]:
    """Retrieve similar train utterances and aggregate their sub labels into a
    per-emotion situational score."""
    recs = await rag.chunks_vdb.query(utterance, top_k=rules.SIT_TOPK)
    neighbours: list[tuple[float, set[str]]] = []
    for rank, rec in enumerate(recs):
        # custom_kg chunks store source_id as full_doc_id (=utt_{clip_id});
        # the vdb query payload omits source_id but keeps full_doc_id.
        src = rec.get("source_id") or rec.get("full_doc_id") or rec.get("id")
        entry = labels_map.get(src)
        if not entry:
            continue
        neighbours.append((_similarity(rank, rec), set(entry.get("sub", []))))
    universe = {e for _, labels in neighbours for e in labels}
    return {e: rules.sit_score(neighbours, e) for e in universe}


# --- default judge (pluggable) ------------------------------------------------
_LLM_MODEL = os.getenv("LLM_MODEL", "exaone3.5:7.8b-instruct-q4_K_M")
_LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")


async def llm_judge(utterance: str, emotion: str) -> bool:
    """Ask the LLM (Ollama directly) whether the utterance carries a cue for
    `emotion`. Called outside LightRAG's wrapper, so we talk to Ollama directly
    rather than rag.llm_model_func (which requires injected hashing_kv)."""
    kr = schema.SUB_KR.get(emotion, emotion)
    prompt = (
        f"발화: \"{utterance}\"\n"
        f"이 발화에 '{kr}({emotion})' 감정의 단서가 있으면 YES, 없으면 NO로만 답하세요."
    )
    client = ollama.AsyncClient(host=_LLM_HOST)
    resp = await client.generate(
        model=_LLM_MODEL, prompt=prompt,
        options={"temperature": 0.2,
                 "num_ctx": int(os.getenv("OLLAMA_LLM_NUM_CTX", "32768"))},
    )
    return "YES" in resp.get("response", "").upper()


# --- main correction --------------------------------------------------------
async def correct(rag: LightRAG, utterance: str, gemini_emotions: list[str],
                  labels_map: dict, directional: dict, judge_fn=None) -> dict:
    """Return {'detected', 'added', 'candidates'} for one utterance."""
    detected = {e for e in gemini_emotions if schema.is_known_sub(e)}

    graph_w = await graph_candidates(rag, detected, directional)
    sit = await situational_candidates(rag, utterance, labels_map)
    candidates = rules.fuse(graph_w, sit, detected)

    added: list[str] = []
    judge_fn = judge_fn or (lambda emo: llm_judge(utterance, emo))
    for c in candidates:
        if c.fused < rules.MIN_FUSED_TO_JUDGE:
            continue
        present = await judge_fn(c.emotion)
        if rules.accept_missing(present, c.fused):
            added.append(c.emotion)

    return {
        "detected": sorted(detected),
        "added": added,
        "candidates": [vars(c) for c in candidates],
    }


async def _demo() -> None:
    """Smoke test against a few holdout rows (requires build_layer1.py first)."""
    labels_map = _load_side("labels_map.json")
    directional = _load_side("cooccur_stats.json")
    holdout = _load_side("holdout.json") or []

    rag = await initialize_rag()
    try:
        for row in (holdout[:5] if isinstance(holdout, list) else []):
            # simulate Gemini by passing the first gold sub as the 1st-pass signal
            gold = row.get("gold_sub") or []
            gemini = gold[:1]
            out = await correct(rag, row.get("transcript", ""), gemini,
                                labels_map, directional)
            print(json.dumps({"clip": row.get("clip_id"), "gemini": gemini,
                              "gold": gold, **out}, ensure_ascii=False))
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(_demo())
