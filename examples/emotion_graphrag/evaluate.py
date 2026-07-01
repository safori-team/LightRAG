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

import infer
import rules

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def _cand_list(graph_w: dict, sit: dict, detected: set, hybrid: bool):
    """Fused candidate list for graph-only or hybrid."""
    return rules.fuse(graph_w, sit if hybrid else {}, detected)


async def main() -> None:
    run_judge = "--judge" in sys.argv
    holdout = _load("holdout.json")
    labels_map = _load("labels_map.json")
    directional = _load("cooccur_stats.json")

    # keep only records that actually have a missing sub-code to recover
    cases = [r for r in holdout if len(r.get("gold_sub") or []) >= 2]

    rag = await infer.initialize_rag()
    agg = {
        "graph": {"recovered": 0, "rank_sum": 0, "rank_n": 0},
        "hybrid": {"recovered": 0, "rank_sum": 0, "rank_n": 0},
    }
    total_missing = 0
    # judge-level accumulators
    j = {"missing": 0, "recovered": 0, "added": 0, "added_in_gold": 0}

    try:
        for r in cases:
            gold = [c for c in r["gold_sub"] if c]
            detected = {gold[0]}
            missing = set(gold[1:])
            total_missing += len(missing)

            graph_w = await infer.graph_candidates(rag, detected, directional)
            sit = await infer.situational_candidates(rag, r.get("transcript", ""),
                                                     labels_map)

            for mode, use_hybrid in (("graph", False), ("hybrid", True)):
                cands = _cand_list(graph_w, sit, detected, use_hybrid)
                cand_emos = [c.emotion for c in cands]
                cand_set = set(cand_emos)
                agg[mode]["recovered"] += len(cand_set & missing)
                for m in missing:
                    if m in cand_emos:
                        agg[mode]["rank_sum"] += cand_emos.index(m) + 1
                        agg[mode]["rank_n"] += 1

            if run_judge:
                # final (hybrid) recall + false-add after the LLM gate
                cands = _cand_list(graph_w, sit, detected, True)
                added = []
                for c in cands:
                    if c.fused < rules.MIN_FUSED_TO_JUDGE:
                        continue
                    if await infer.llm_judge(r.get("transcript", ""), c.emotion):
                        added.append(c.emotion)
                gold_set = set(gold)
                j["missing"] += len(missing)
                j["recovered"] += len(set(added) & missing)
                j["added"] += len(added)
                j["added_in_gold"] += len(set(added) & gold_set)
    finally:
        await rag.finalize_storages()

    # --- report ---------------------------------------------------------------
    print(f"\nholdout cases with missing sub-code: {len(cases)}  "
          f"total missing sub-codes: {total_missing}\n")
    print("A. Candidate recall (no LLM)   recall = recovered / total_missing")
    for mode in ("graph", "hybrid"):
        rec = agg[mode]["recovered"]
        recall = rec / total_missing if total_missing else 0.0
        mrank = (agg[mode]["rank_sum"] / agg[mode]["rank_n"]) if agg[mode]["rank_n"] else 0.0
        print(f"   {mode:6}  recall={recall:.3f} ({rec}/{total_missing})  "
              f"mean_rank_of_recovered={mrank:.2f}")

    if run_judge:
        recall = j["recovered"] / j["missing"] if j["missing"] else 0.0
        precision = j["added_in_gold"] / j["added"] if j["added"] else 0.0
        false_add = 1 - precision if j["added"] else 0.0
        print("\nB. After LLM judge (hybrid)")
        print(f"   missing-recall = {recall:.3f} ({j['recovered']}/{j['missing']})")
        print(f"   add-precision  = {precision:.3f} ({j['added_in_gold']}/{j['added']})")
        print(f"   false-add rate = {false_add:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
