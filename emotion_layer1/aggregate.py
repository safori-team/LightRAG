"""Edge aggregation — COUNTED from gold labels, thresholded.

선1 (emotion co-occurrence): for each record, every unordered pair of emotions
labeled together is counted. Weight = co-occurrence count; the conditional
probabilities P(B|A) and P(A|B) are carried in the description for the
inference loop. Pairs below ``min_count`` are dropped (weak edges discarded).

Aggregation runs at minor level by default ('섭섭' = 슬픔+사랑 lives here) and
optionally at major level. Only TRAIN records are passed in (holdout never
enters the graph).

cue -> emotion (선cue) is aggregated the same way: (cue, minor) co-occurrence.
선2 (context -> emotion) is intentionally deferred (context not controlled yet).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from .schema import Record
from .taxonomy import Taxonomy
from .nodes import SOURCE_ID


@dataclass
class CoocConfig:
    level: str = "minor"          # "minor" | "major"
    min_count: int = 2            # weak-edge threshold
    confidence_floor: float = 0.0 # ignore emotions below this confidence


def _emotion_keys(rec: Record, level: str, floor: float) -> list[str]:
    keys = []
    for e in rec.emotions:
        if e.confidence < floor:
            continue
        keys.append(e.minor if level == "minor" else e.major)
    # dedupe within a record so a repeated label is not a self-pair
    return sorted(set(keys))


def cooccurrence(records: list[Record], cfg: CoocConfig) -> dict[tuple[str, str], int]:
    """Unordered emotion pair -> co-occurrence count."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    for rec in records:
        keys = _emotion_keys(rec, cfg.level, cfg.confidence_floor)
        for a, b in combinations(keys, 2):
            pair_counts[tuple(sorted((a, b)))] += 1
    return dict(pair_counts)


def marginal(records: list[Record], cfg: CoocConfig) -> Counter:
    """Per-emotion record frequency (denominator for P(B|A))."""
    counts: Counter[str] = Counter()
    for rec in records:
        for k in _emotion_keys(rec, cfg.level, cfg.confidence_floor):
            counts[k] += 1
    return counts


def build_cooccurrence_edges(
    records: list[Record], tax: Taxonomy, cfg: CoocConfig | None = None
) -> list[dict]:
    cfg = cfg or CoocConfig()
    pair_counts = cooccurrence(records, cfg)
    marg = marginal(records, cfg)
    node = tax.minor_node if cfg.level == "minor" else tax.major_node

    edges: list[dict] = []
    for (a, b), count in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        if count < cfg.min_count:
            continue
        p_b_given_a = count / marg[a] if marg[a] else 0.0
        p_a_given_b = count / marg[b] if marg[b] else 0.0
        edges.append({
            "src_id": node(a),
            "tgt_id": node(b),
            "description": (
                f"감정 '{a}'와 '{b}' 공존 {count}회. "
                f"P({b}|{a})={p_b_given_a:.2f}, P({a}|{b})={p_a_given_b:.2f}."
            ),
            "keywords": f"CO_OCCURS,선1,{cfg.level}",
            "weight": float(count),
            "source_id": SOURCE_ID,
        })
    return edges


def build_cue_edges(
    records: list[Record], tax: Taxonomy, cfg: CoocConfig | None = None
) -> list[dict]:
    """(cue, minor-emotion) co-occurrence -> cue->emotion edges."""
    cfg = cfg or CoocConfig()
    pair_counts: Counter[tuple[str, str]] = Counter()
    cue_marg: Counter[str] = Counter()
    for rec in records:
        emos = _emotion_keys(rec, "minor", cfg.confidence_floor)
        cues = sorted(set(rec.nonverbal_tags))
        for c in cues:
            cue_marg[c] += 1
            for emo in emos:
                pair_counts[(c, emo)] += 1

    edges: list[dict] = []
    for (cue, emo), count in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        if count < cfg.min_count:
            continue
        p_emo_given_cue = count / cue_marg[cue] if cue_marg[cue] else 0.0
        edges.append({
            "src_id": tax.cue_node(cue),
            "tgt_id": tax.minor_node(emo),
            "description": (
                f"cue '{cue}' 동반 시 감정 '{emo}' {count}회. "
                f"P({emo}|{cue})={p_emo_given_cue:.2f}."
            ),
            "keywords": "CUE_INDICATES,cue선",
            "weight": float(count),
            "source_id": SOURCE_ID,
        })
    return edges
