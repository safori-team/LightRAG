"""Inference-loop skeleton (실시간 보정) — candidate query + clue check stub.

The graph only SUPPLIES candidates and evidence; the final add/explain decision
belongs to the LLM (wired later). This module implements the deterministic
graph-read half:

  1. given the emotions Gemini already found, look up co-occurring / missing
     emotion candidates from the 선1 edges (ranked by weight),
  2. a stub ``clue_check`` that the LLM/keyword step will replace,
  3. ``suggest_missing`` that returns candidates Gemini did NOT already emit.

Reads go through the public BaseGraphStorage API (same as emotion_engine's
graph_builder), so no core file is touched.
"""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import Taxonomy, MINOR_PREFIX


@dataclass
class Candidate:
    emotion: str          # taxonomy local id (e.g. minor Korean name)
    node_id: str
    weight: float
    reason: str           # edge description (carries P(B|A), count)


def _local(node_id: str) -> str:
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


async def cooccur_candidates(
    graph, tax: Taxonomy, emotion_minor: str, *, top_k: int = 5
) -> list[Candidate]:
    """선1 neighbors of one emotion, ranked by co-occurrence weight."""
    src = tax.minor_node(emotion_minor)
    edges = await graph.get_node_edges(src) or []
    cands: list[Candidate] = []
    for a, b in edges:
        other = b if a == src else a
        if not other.startswith(f"{MINOR_PREFIX}:"):
            continue
        data = await graph.get_edge(a, b) or {}
        kw = data.get("keywords", "")
        if "CO_OCCURS" not in kw:
            continue
        cands.append(
            Candidate(
                emotion=_local(other),
                node_id=other,
                weight=float(data.get("weight", 0.0)),
                reason=data.get("description", ""),
            )
        )
    cands.sort(key=lambda c: -c.weight)
    return cands[:top_k]


def clue_check(transcript: str, emotion_minor: str) -> bool:
    """STUB: does the utterance actually contain a clue for this emotion?

    Replaced later by an LLM call / keyword evidence check. For now returns
    True so the skeleton flows end to end.
    """
    return True


async def suggest_missing(
    graph,
    tax: Taxonomy,
    gemini_emotions: list[str],
    transcript: str,
    *,
    top_k: int = 5,
) -> list[Candidate]:
    """Candidates co-occurring with Gemini's emotions that Gemini missed
    AND that pass the clue check. This is the layer-1 보정 proposal."""
    have = set(gemini_emotions)
    seen: dict[str, Candidate] = {}
    for emo in gemini_emotions:
        for c in await cooccur_candidates(graph, tax, emo, top_k=top_k):
            if c.emotion in have:
                continue
            if not clue_check(transcript, c.emotion):
                continue
            prev = seen.get(c.emotion)
            if prev is None or c.weight > prev.weight:
                seen[c.emotion] = c
    return sorted(seen.values(), key=lambda c: -c.weight)[:top_k]
