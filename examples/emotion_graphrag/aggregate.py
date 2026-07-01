"""Offline label aggregation: CSV -> co_occurs edges, utterance chunks, labels.

Co-occurrence is aggregated from per-annotator sub_code readings (~497), NOT
from the majority label (which is a single code for 71/100 records). See
DESIGN.md sections 1 and 4. Pure stdlib; depends on rules.py and schema.py.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter

import rules
import schema


# --- CSV loading & field access ----------------------------------------------
def load_records(csv_path: str) -> list[dict]:
    # utf-8-sig strips a leading BOM so the first column key is 'clip_id'.
    with open(csv_path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def speaker_of(row: dict) -> str:
    """Speaker id = filename prefix, e.g. F0001_102994.wav -> F0001."""
    return row["filename"].split("_")[0]


def _json_list(row: dict, key: str) -> list:
    try:
        return json.loads(row[key]) or []
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def annotator_readings(row: dict):
    """Yield each annotator's set of KNOWN sub codes for one record."""
    try:
        labels = json.loads(row["annotator_labels"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return
    for _annotator, d in labels.items():
        codes = {c for c in d.get("sub_codes", []) if schema.is_known_sub(c)}
        if codes:
            yield codes


# --- Speaker-disjoint train / holdout split ----------------------------------
def train_holdout_split(rows: list[dict], holdout_frac: float = 0.2,
                        seed: int = 13) -> tuple[list[dict], list[dict]]:
    """Split by SPEAKER so no speaker leaks across the boundary.

    Holdout must never enter the graph or the retrieval index (DESIGN.md §8).
    """
    speakers = sorted({speaker_of(r) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_holdout = max(1, round(len(speakers) * holdout_frac))
    holdout_speakers = set(speakers[:n_holdout])
    train = [r for r in rows if speaker_of(r) not in holdout_speakers]
    holdout = [r for r in rows if speaker_of(r) in holdout_speakers]
    return train, holdout


# --- co_occurs aggregation ----------------------------------------------------
def aggregate_cooccurs(rows: list[dict]) -> tuple[list[dict], dict[str, dict], dict]:
    """Aggregate co_occurs from per-annotator readings.

    Returns (undirected_edges, directional_stats, debug_stats).

    The graph edge is UNDIRECTED with a symmetric weight, because LightRAG's
    default graph store (NetworkX) is undirected and would otherwise collapse
    a->b and b->a. The asymmetric P(B|A) info is kept in ``directional_stats``
    (a code-side table read at inference), consistent with R1: graph carries a
    signal, precise directional weights live outside it. ``lift`` is symmetric
    so the pass/keep decision is direction-independent.
    """
    sub_count: Counter = Counter()          # count(A) over readings
    pair_count: Counter = Counter()         # count(A, B) ordered
    n_readings = 0

    for row in rows:
        for codes in annotator_readings(row):
            n_readings += 1
            for a in codes:
                sub_count[a] += 1
                for b in codes:
                    if a != b:
                        pair_count[(a, b)] += 1

    edges: list[dict] = []
    directional: dict[str, dict] = {}       # "A|B" -> P(B|A)-based stat
    seen_pairs: set[frozenset] = set()
    kept = 0

    for (a, b), c_ab in pair_count.items():
        ab = rules.cooccur_stat(a, b, c_ab, sub_count[a], sub_count[b], n_readings)
        if not rules.passes_cooccur(ab):
            continue
        # directional entry (both a->b and b->a get their own row)
        directional[f"{a}|{b}"] = {
            "weight": ab.weight, "p_b_given_a": ab.p_b_given_a,
            "lift": ab.lift, "count": ab.count_ab,
        }
        # undirected graph edge: emit once per unordered pair
        key = frozenset((a, b))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        ba = rules.cooccur_stat(b, a, c_ab, sub_count[b], sub_count[a], n_readings)
        weight = max(ab.weight, ba.weight)   # symmetric strength
        # Embedded description stays clean/semantic (numeric evidence lives in
        # cooccur_stats.json). Numeric soup here both hurts vector search and
        # triggers bge-m3 NaN embeddings under load.
        desc = f"{schema.SUB_KR.get(a, a)}과(와) {schema.SUB_KR.get(b, b)} 감정이 함께 나타나는 경향"
        edges.append(schema.cooccur_edge(a, b, weight, desc))
        kept += 1

    stats = {
        "n_readings": n_readings,
        "n_pairs": len(pair_count),
        "n_undirected_edges": kept,
        "n_directional_entries": len(directional),
        "unique_subs_seen": len(sub_count),
    }
    return edges, directional, stats


# --- utterance chunks + label side-map ---------------------------------------
def utterance_chunks(rows: list[dict]) -> list[dict]:
    """One chunk per record (situational-similarity retrieval index)."""
    chunks: list[dict] = []
    for row in rows:
        transcript = (row.get("transcript") or "").strip()
        if not transcript:
            continue
        chunks.append(schema.utterance_chunk(row["clip_id"], transcript))
    return chunks


def labels_map(rows: list[dict]) -> dict[str, dict]:
    """utt_{clip_id} -> {major: [...], sub: [...]} for post-retrieval lookup."""
    out: dict[str, dict] = {}
    for row in rows:
        clip_id = row["clip_id"]
        out[schema._utt_source_id(clip_id)] = {
            "clip_id": clip_id,
            "speaker": speaker_of(row),
            "major": _json_list(row, "human_majority_major_codes"),
            "sub": [c for c in _json_list(row, "human_majority_sub_codes")
                    if schema.is_known_sub(c)],
        }
    return out
