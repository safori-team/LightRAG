"""Holdout evaluation for the emotion reviewer.

Separates two questions (DESIGN.md §8, "신호 vs 해석 정확도 분리"):

  A. Candidate quality (no LLM, fast): given a simulated 1st-pass that keeps
     only the top gold sub-code, do the FUSED candidates contain the *missing*
     gold sub-codes? Compares graph-only vs hybrid (adds situational source).

  B. Judge quality (optional --judge, slow): after the LLM gate, what are the
     final missing-recall and false-add rates?

Gemini is simulated deterministically as ``detected = gold_sub[:1]`` so the
rest of gold_sub becomes the "missing" set the reviewer must recover. Holdout
speakers are never in the graph / retrieval index (leakage-free).

Run (after build_layer1.py):
    python evaluate.py            # candidate-level A only (fast)
    python evaluate.py --judge    # also run B (LLM judge, slow)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter

import aggregate
import infer
import rules
import schema

# "acceptable" gold = sub-codes given by at least this many annotators.
# Adding an acceptable emotion is NOT counted as a false-add (matches the
# "emotions are subjective/compound" premise; majority-only is too strict).
ACCEPT_MIN = 2
EMOTION_CSV = os.getenv(
    "EMOTION_CSV",
    r"C:\Users\windowadmin6\Desktop\safori\data\emotion_experiment_dataset.csv",
)


def build_accept_map(rows, min_annot=ACCEPT_MIN):
    """clip_id -> {sub-codes given by >= min_annot annotators} (known subs only)."""
    out = {}
    for row in rows:
        cnt = Counter()
        for codes in aggregate.annotator_readings(row):
            for s in codes:
                cnt[s] += 1
        out[row["clip_id"]] = {s for s, c in cnt.items() if c >= min_annot}
    return out

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def _cand_list(graph_w: dict, sit: dict, detected: set, hybrid: bool):
    """Fused candidate list for graph-only or hybrid."""
    return rules.fuse(graph_w, sit if hybrid else {}, detected)


async def main() -> None:
    run_judge = "--judge" in sys.argv
    run_judge2 = "--judge2" in sys.argv
    holdout = _load("holdout.json")
    labels_map = _load("labels_map.json")
    directional = _load("cooccur_stats.json")

    # keep only records that actually have a missing sub-code to recover
    cases = [r for r in holdout if len(r.get("gold_sub") or []) >= 2]

    # acceptable gold (>=ACCEPT_MIN annotators) from the raw CSV, for a fair
    # false-add measurement.
    accept_map = build_accept_map(aggregate.load_records(EMOTION_CSV))

    rag = await infer.initialize_rag()
    agg = {
        "graph": {"recovered": 0, "rank_sum": 0, "rank_n": 0, "cross": 0, "same": 0},
        "hybrid": {"recovered": 0, "rank_sum": 0, "rank_n": 0, "cross": 0, "same": 0},
    }
    total_missing = 0
    total_cross = 0   # missing sub-codes whose major differs from detected's
    total_same = 0
    # judge-level accumulators
    # added_in_maj  = added ∩ majority gold (strict precision)
    # added_in_acc  = added ∩ acceptable(>=2 annot) gold (fair precision)
    j = {"missing": 0, "recovered": 0, "added": 0,
         "added_in_maj": 0, "added_in_acc": 0}
    # judge v2: (confidence, is_target_missing, is_majority, is_acceptable)
    judged2: list[tuple[float, bool, bool, bool]] = []

    try:
        for r in cases:
            gold = [c for c in r["gold_sub"] if c]
            detected = {gold[0]}
            missing = set(gold[1:])
            total_missing += len(missing)

            det_majors = {schema.SUB2MAJOR.get(d) for d in detected}
            is_cross = {m: schema.SUB2MAJOR.get(m) not in det_majors for m in missing}
            total_cross += sum(is_cross.values())
            total_same += len(missing) - sum(is_cross.values())

            graph_w = await infer.graph_candidates(rag, detected, directional)
            sit = await infer.situational_candidates(rag, r.get("transcript", ""),
                                                     labels_map)

            for mode, use_hybrid in (("graph", False), ("hybrid", True)):
                cands = _cand_list(graph_w, sit, detected, use_hybrid)
                cand_emos = [c.emotion for c in cands]
                for m in missing:
                    if m in cand_emos:
                        agg[mode]["recovered"] += 1
                        agg[mode]["rank_sum"] += cand_emos.index(m) + 1
                        agg[mode]["rank_n"] += 1
                        if is_cross[m]:
                            agg[mode]["cross"] += 1
                        else:
                            agg[mode]["same"] += 1

            if run_judge:
                # final (hybrid) recall + false-add after the LLM gate
                cands = _cand_list(graph_w, sit, detected, True)
                added = []
                for c in cands:
                    if c.fused < rules.MIN_FUSED_TO_JUDGE:
                        continue
                    if await infer.llm_judge(r.get("transcript", ""), c.emotion):
                        added.append(c.emotion)
                gold_set = set(gold)                     # majority (target)
                accept = accept_map.get(r["clip_id"], set()) | gold_set  # >=2 annot
                added_set = set(added)
                j["missing"] += len(missing)
                j["recovered"] += len(added_set & missing)
                j["added"] += len(added)
                j["added_in_maj"] += len(added_set & gold_set)
                j["added_in_acc"] += len(added_set & accept)

            if run_judge2:
                cands = _cand_list(graph_w, sit, detected, True)
                confs = await infer.llm_judge_batch(
                    r.get("transcript", ""), detected, cands)
                accept = accept_map.get(r["clip_id"], set()) | set(gold)
                for c in cands:
                    judged2.append((confs.get(c.emotion, 0.0),
                                    c.emotion in missing,
                                    c.emotion in set(gold),
                                    c.emotion in accept))
    finally:
        await rag.finalize_storages()

    # --- report ---------------------------------------------------------------
    def _r(x, n):
        return f"{(x / n if n else 0.0):.3f} ({x}/{n})"

    print(f"\n[조건 A: data-only]  holdout cases={len(cases)}  "
          f"missing total={total_missing}  (same-major={total_same}, "
          f"cross-major={total_cross})\n")
    print("후보 회수율 (LLM 무관)   overall / same-major / CROSS-MAJOR(그래프 니치)")
    for mode in ("graph", "hybrid"):
        a = agg[mode]
        mrank = (a["rank_sum"] / a["rank_n"]) if a["rank_n"] else 0.0
        print(f"   {mode:6}  overall={_r(a['recovered'], total_missing)}  "
              f"same={_r(a['same'], total_same)}  "
              f"CROSS={_r(a['cross'], total_cross)}  mean_rank={mrank:.2f}")
    print("\n비교기준: same-major는 계층 휴리스틱(verify_value.py의 B2)도 잡음.")
    print("          CROSS-MAJOR는 계층이 구조상 0 → 여기서 그래프/하이브리드 회수=그래프 고유 가치.")

    if run_judge:
        added = j["added"]
        recall = j["recovered"] / j["missing"] if j["missing"] else 0.0
        strict_fa = 1 - j["added_in_maj"] / added if added else 0.0
        fair_fa = 1 - j["added_in_acc"] / added if added else 0.0
        print("\nB. After LLM judge (hybrid)")
        print(f"   target-recall (다수결 회수)     = {recall:.3f} "
              f"({j['recovered']}/{j['missing']})")
        print(f"   오탐율 [엄격: 다수결에 없음]    = {strict_fa:.3f} "
              f"(정답매칭 {j['added_in_maj']}/{added})")
        print(f"   오탐율 [공정: ≥{ACCEPT_MIN}명도 없음]   = {fair_fa:.3f} "
              f"(사람지지 {j['added_in_acc']}/{added})")
        print("   → 두 오탐율의 차이 = '틀렸다'고 셌지만 실제로 소수 어노테이터가 지지한 추가.")

    if run_judge2:
        print(f"\nB2. judge v2 (근거주입+선택형+확신도, TOP_K={rules.CANDIDATE_TOP_K}, "
              f"{len(judged2)} candidates judged)")
        print("   tau   target-recall      오탐[엄격]  오탐[공정≥2]  added")
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
            added_rows = [x for x in judged2 if x[0] >= tau]
            n = len(added_rows)
            recovered = sum(1 for x in added_rows if x[1])
            maj_hits = sum(1 for x in added_rows if x[2])
            acc_hits = sum(1 for x in added_rows if x[3])
            recall = recovered / total_missing if total_missing else 0.0
            strict_fa = 1 - maj_hits / n if n else 0.0
            fair_fa = 1 - acc_hits / n if n else 0.0
            print(f"   {tau:.1f}   {recall:.3f} ({recovered}/{total_missing})"
                  f"        {strict_fa:.3f}      {fair_fa:.3f}        {n}")
        print("   (tau↑ → 회수율↓·오탐↓. 공정 잣대에서 최적 지점을 고름)")


if __name__ == "__main__":
    asyncio.run(main())
