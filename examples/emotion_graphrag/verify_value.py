"""Objective check: does the co_occurs GRAPH add value over naive baselines?

No LLM, no embeddings — pure train-label statistics vs the holdout. Compares
candidate recall@K for increasingly "smart" proposers. If the co-occurrence
lookup does NOT beat the context-free / same-major baselines, then the graph
structure is not earning its complexity for this task.

Baselines (given a simulated 1st-pass detected = gold_sub[:1], recover gold_sub[1:]):
  B1 global-freq   : propose globally most frequent sub-codes (ignores detected)
  B2 same-major    : propose sibling sub-codes of detected's major, by freq
                     (uses ONLY the taxonomy hierarchy, no co_occur stats)
  B3 cooccur-1hop  : propose neighbours of detected ranked by P(B|A) wilson
                     (this is exactly what our "graph" does)

Run: python verify_value.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

import aggregate
import rules
import schema

HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = 5
CSV = os.getenv(
    "EMOTION_CSV",
    r"C:\Users\windowadmin6\Desktop\safori\data\emotion_experiment_dataset.csv",
)


def train_stats(train_rows):
    sub_count: Counter = Counter()
    pair_count: Counter = Counter()
    n = 0
    for row in train_rows:
        for codes in aggregate.annotator_readings(row):
            n += 1
            for a in codes:
                sub_count[a] += 1
                for b in codes:
                    if a != b:
                        pair_count[(a, b)] += 1
    return sub_count, pair_count, n


def propose_global(sub_count, detected, k=TOPK):
    ranked = [s for s, _ in sub_count.most_common() if s not in detected]
    return ranked[:k]


def propose_same_major(sub_count, detected, k=TOPK):
    majors = {schema.SUB2MAJOR[d] for d in detected if d in schema.SUB2MAJOR}
    sibs = [s for s in schema.ALL_SUBS
            if schema.SUB2MAJOR[s] in majors and s not in detected]
    sibs.sort(key=lambda s: -sub_count.get(s, 0))
    return sibs[:k]


def propose_cooccur(sub_count, pair_count, detected, k=TOPK):
    scores: dict[str, float] = {}
    for a in detected:
        ca = sub_count.get(a, 0)
        if not ca:
            continue
        for (x, b), c_ab in pair_count.items():
            if x != a or b in detected:
                continue
            w = rules.wilson_lb(c_ab, ca)
            # gate identical to the build: support + lift
            lift = (c_ab / ca) / (sub_count[b] / sum(sub_count.values()))
            if c_ab >= rules.SUPPORT_FLOOR and lift >= rules.TAU_LIFT:
                scores[b] = max(scores.get(b, 0.0), w)
    return [e for e, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def main():
    rows = aggregate.load_records(CSV)
    train, _ = aggregate.train_holdout_split(rows, 0.2)
    holdout = json.load(open(os.path.join(HERE, "holdout.json"), encoding="utf-8"))
    cases = [r for r in holdout if len(r.get("gold_sub") or []) >= 2]

    sub_count, pair_count, n = train_stats(train)

    methods = {
        "B1 global-freq ": propose_global,
        "B2 same-major  ": propose_same_major,
        "B3 cooccur-1hop": propose_cooccur,
    }
    rec = {m: 0 for m in methods}
    total_missing = 0

    for r in cases:
        gold = [c for c in r["gold_sub"] if c]
        detected = {gold[0]}
        missing = set(gold[1:])
        total_missing += len(missing)
        for name, fn in methods.items():
            if fn is propose_cooccur:
                cand = fn(sub_count, pair_count, detected)
            else:
                cand = fn(sub_count, detected)
            rec[name] += len(set(cand) & missing)

    print(f"\ncases={len(cases)}  total_missing={total_missing}  "
          f"(recall@{TOPK}, no LLM, no embeddings)\n")
    for name in methods:
        r_ = rec[name]
        print(f"  {name}  recall={r_/total_missing:.3f} ({r_}/{total_missing})")
    print("\n비교: B3(그래프 co_occur)가 B1/B2보다 확실히 높으면 그래프에 값이 있음.")
    print("      B2(계층만)와 비슷하면, 값의 대부분은 '같은 대분류'에서 나온 것.")


if __name__ == "__main__":
    main()
