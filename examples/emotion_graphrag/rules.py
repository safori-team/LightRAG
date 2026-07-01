"""Deterministic decision layer for the emotion GraphRAG reviewer.

All precise weights, thresholds and fusion coefficients live here as code
constants (design requirement R1). The graph only carries the numbers this
module produces; nothing here imports LightRAG. See DESIGN.md sections 4 and 5.

The starting constants are NOT final — they are tuned on a held-out split via
the recall / false-add tradeoff (DESIGN.md section 8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Tuning constants (holdout-selected, not frozen) --------------------------
SUPPORT_FLOOR = 2       # co_occurrence count below this = one-off noise -> drop
TAU_LIFT = 1.3          # graph co_occurs candidate cut (chance-corrected)
WILSON_Z = 1.28         # 80% lower bound (lenient for small data)

# Situational similarity (source 2) + fusion
SIT_TOPK = 8            # neighbours retrieved for situational scoring
SIGMA_SIT = 0.25        # situational candidate cut
LAMBDA_SIT = 0.5        # weight of situational signal in fused score
CANDIDATE_TOP_K = 5     # max candidates handed to the judge

# Personalization (layer 2, deferred until repeated speakers exist)
BOOST_ALPHA = 0.35
BOOST_CAP = 2.0

# Judge gate
MIN_FUSED_TO_JUDGE = 0.05   # don't bother the LLM with near-zero candidates


# --- Co-occurrence statistics -------------------------------------------------
def wilson_lb(k: int, n: int, z: float = WILSON_Z) -> float:
    """Wilson score lower bound of the proportion k/n.

    Shrinks unreliable (small-n) estimates toward 0 so weak-evidence edges
    self-penalise in ranking. Used as the co_occurs edge weight.
    """
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


@dataclass
class CooccurStat:
    src: str
    tgt: str
    count_ab: int          # count(A, B)
    p_b_given_a: float      # P(B|A)
    lift: float             # P(B|A) / P(B)
    pmi: float              # log( P(A,B) / (P(A)P(B)) )
    weight: float           # wilson_lb(P(B|A))


def cooccur_stat(src: str, tgt: str, count_ab: int, count_a: int,
                 count_b: int, n_readings: int) -> CooccurStat:
    """Compute all evidence numbers for a directed pair src -> tgt."""
    p_b_given_a = count_ab / count_a if count_a else 0.0
    p_b = count_b / n_readings if n_readings else 0.0
    lift = (p_b_given_a / p_b) if p_b > 0 else 0.0
    p_ab = count_ab / n_readings if n_readings else 0.0
    p_a = count_a / n_readings if n_readings else 0.0
    pmi = math.log(p_ab / (p_a * p_b)) if (p_ab > 0 and p_a > 0 and p_b > 0) else float("-inf")
    return CooccurStat(
        src=src, tgt=tgt, count_ab=count_ab,
        p_b_given_a=round(p_b_given_a, 3),
        lift=round(lift, 2),
        pmi=round(pmi, 2),
        weight=round(wilson_lb(count_ab, count_a), 3),
    )


def passes_cooccur(stat: CooccurStat) -> bool:
    """4-step funnel: support floor + lift (chance-corrected).

    Wilson (weight) and direction are handled by keeping the number and by
    aggregating each direction separately in aggregate.py.
    """
    return stat.count_ab >= SUPPORT_FLOOR and stat.lift >= TAU_LIFT


def cooccur_evidence(stat: CooccurStat) -> str:
    """Human-readable, auditable justification stored on the edge."""
    return (
        f"{stat.src}→{stat.tgt} 동반. "
        f"count={stat.count_ab}, P(B|A)={stat.p_b_given_a}, "
        f"lift={stat.lift}, pmi={stat.pmi}"
    )


# --- Inference-time fusion (source 1 graph prior + source 2 situational) -------
def sit_score(neighbours: list[tuple[float, set[str]]], emotion: str) -> float:
    """Similarity-weighted label frequency for one candidate emotion.

    neighbours: list of (similarity, labels) for retrieved similar utterances.
    Returns Σ sim·1[emotion∈labels] / Σ sim  in [0, 1].
    """
    total = sum(sim for sim, _ in neighbours)
    if total <= 0:
        return 0.0
    hit = sum(sim for sim, labels in neighbours if emotion in labels)
    return hit / total


def personalize_boost(base_weight: float, n_speaker: int) -> float:
    """Layer-2 personalization multiplier (deferred; cold start -> 1.0)."""
    if n_speaker <= 0:
        return base_weight
    boost = min(BOOST_CAP, 1 + BOOST_ALPHA * math.log(1 + n_speaker))
    return base_weight * boost


@dataclass
class Candidate:
    emotion: str
    graph_weight: float
    sit: float
    fused: float
    source: str          # "graph", "situational", or "both"


def fuse(graph_weights: dict[str, float], sit_scores: dict[str, float],
         detected: set[str]) -> list[Candidate]:
    """Combine the two candidate sources into a ranked candidate list.

    A candidate qualifies if it clears EITHER the graph or situational cut;
    agreement across both is rewarded via addition. Already-detected emotions
    are excluded (the reviewer only proposes *missing* ones).
    """
    emotions = (set(graph_weights) | set(sit_scores)) - detected
    out: list[Candidate] = []
    for e in emotions:
        gw = graph_weights.get(e, 0.0)
        st = sit_scores.get(e, 0.0)
        in_graph = gw > 0.0
        in_sit = st >= SIGMA_SIT
        if not (in_graph or in_sit):
            continue
        fused = gw + LAMBDA_SIT * st
        source = "both" if (in_graph and in_sit) else ("graph" if in_graph else "situational")
        out.append(Candidate(emotion=e, graph_weight=round(gw, 3),
                              sit=round(st, 3), fused=round(fused, 3), source=source))
    # stable secondary key (emotion) so ties are reproducible across processes
    out.sort(key=lambda c: (-c.fused, c.emotion))
    return out[:CANDIDATE_TOP_K]


def accept_missing(llm_says_present: bool, fused: float) -> bool:
    """Final judge gate. The LLM confirms the cue on the actual utterance
    (design: graph proposes, LLM decides); the fused floor only avoids judging
    near-zero candidates. Graph-only candidates (sit=0) can still be added."""
    return llm_says_present and fused >= MIN_FUSED_TO_JUDGE
