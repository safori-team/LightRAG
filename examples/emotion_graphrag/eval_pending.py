"""Evaluate the emotion reviewer on the pending_100 test set.

Gold = the human labels in the manifest; the 1st pass = Gemini's analysis JSON.
Six arms share one candidate/judge pipeline so each comparison isolates one
variable:

    A0  Gemini only                          baseline
    A1  + same-major siblings (top-K)        non-graph hierarchy heuristic
    A2  + globally frequent labels (top-K)   non-graph frequency heuristic
    A3  + graph/hybrid candidates (top-K)    graph signal, no LLM
    A4  + LLM judge over all 48, no evidence LLM without the graph
    A5  + LLM judge over graph candidates    the full system

A5 − A4 is the graph's contribution beyond the LLM; A5 − A1 is the graph's
margin over the cheap heuristic. Reporting A5 alone answers neither.

The threshold tau is applied by this script, not by the judge, so a stored run
can be re-scored at other thresholds without re-calling the model. Choose tau
on a tuning split (holdout / legacy100) and pass it here fixed — sweeping it on
pending_100 and reporting the best point is fitting the test set.

    python eval_pending.py --artifacts artifacts/full100 --arms A0,A1,A2,A3
    python eval_pending.py --artifacts artifacts/full100 --arms A5 --tau 0.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import dataset_pending
import judge as judge_mod
import retrieve
import rules
import schema

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ALL_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5"]
LLM_ARMS = {"A4", "A5"}


# --- non-graph baselines ------------------------------------------------------
def global_frequency(labels_map: dict) -> Counter:
    """Training-set label frequency, the denominator-free 'just guess common
    emotions' baseline (and the ranking used for sibling selection)."""
    freq: Counter = Counter()
    for entry in labels_map.values():
        for sub in entry.get("sub", []):
            freq[sub] += 1
    return freq


def sibling_candidates(detected: set[str], freq: Counter, top_k: int) -> list[str]:
    """Same-major siblings of the detected emotions, most frequent first."""
    majors = {schema.SUB2MAJOR[c] for c in detected if schema.is_known_sub(c)}
    pool = [s for s in schema.ALL_SUBS
            if schema.SUB2MAJOR[s] in majors and s not in detected]
    pool.sort(key=lambda s: (-freq.get(s, 0), s))
    return pool[:top_k]


def frequent_candidates(detected: set[str], freq: Counter, top_k: int) -> list[str]:
    pool = [s for s in schema.ALL_SUBS if s not in detected]
    pool.sort(key=lambda s: (-freq.get(s, 0), s))
    return pool[:top_k]


# --- evidence lines for the judge prompt --------------------------------------
def evidence_views(cands, graph: dict, examples: dict) -> list[judge_mod.CandidateView]:
    views = []
    for c in cands:
        g = graph.get(c.emotion)
        if g and g.get("count") is not None:
            graph_line = (f"{g['via']}와(과) {g['count']}회 동반, "
                          f"P({c.emotion}|{g['via']})={g['p']}, lift {g['lift']}")
        elif g:
            graph_line = f"{g['via']}와(과) 동반 (weight {g['w']})"
        else:
            graph_line = ""
        ex = examples.get(c.emotion)
        sit_line = (f'유사 발화 "{ex["text"]}"(유사도 {ex["sim"]})에 이 감정이 라벨됨'
                    if ex else "")
        views.append(judge_mod.CandidateView(
            code=c.emotion, graph_line=graph_line, sit_line=sit_line))
    return views


# --- scoring ------------------------------------------------------------------
def score_one(final: set[str], gold: set[str], detected: set[str]) -> dict:
    inter = final & gold
    union = final | gold
    missing = gold - detected
    det_majors = {schema.SUB2MAJOR[c] for c in detected if schema.is_known_sub(c)}
    cross = {c for c in missing if schema.SUB2MAJOR.get(c) not in det_majors}
    return {
        "jaccard": len(inter) / len(union) if union else 0.0,
        "recall": len(inter) / len(gold) if gold else 0.0,
        "precision": len(inter) / len(final) if final else 0.0,
        "n_final": len(final),
        "cross_major_missing": len(cross),
        "cross_major_recovered": len(cross & final),
    }


def paired_bootstrap(a: list[float], b: list[float], n: int = 5000,
                     seed: int = 13) -> dict:
    """95% CI of mean(a) - mean(b), resampling clips (arms share the clips)."""
    rng = random.Random(seed)
    idx = range(len(a))
    diffs = []
    for _ in range(n):
        sample = [rng.choice(idx) for _ in idx]
        diffs.append(statistics.fmean(a[i] for i in sample)
                     - statistics.fmean(b[i] for i in sample))
    diffs.sort()
    return {
        "delta": round(statistics.fmean(a) - statistics.fmean(b), 4),
        "ci_low": round(diffs[int(0.025 * n)], 4),
        "ci_high": round(diffs[int(0.975 * n)], 4),
    }


# --- one arm ------------------------------------------------------------------
async def _run_one(arm: str, rec, art, freq: Counter, *, tau: float,
                   top_k: int, use_situational: bool) -> dict:
    detected = set(rec.detected)
    added: list[str] = []
    cand_dump: list[dict] = []
    judgements: list[dict] = []
    parse_fail = False
    hallucinated: list[str] = []

    if arm == "A0":
        pass
    elif arm == "A1":
        added = sibling_candidates(detected, freq, top_k)
    elif arm == "A2":
        added = frequent_candidates(detected, freq, top_k)
    else:
        state = await retrieve.candidates_for(
            art, rec.transcript, detected, use_situational=use_situational)
        fused = state["fused"]
        cand_dump = [
            {**asdict(c), "stats": state["graph"].get(c.emotion),
             "example": state["examples"].get(c.emotion)}
            for c in fused
        ]
        if arm == "A3":
            added = [c.emotion for c in fused[:top_k]]
        else:
            if arm == "A4":
                # Control: no graph. Every taxonomy code is a candidate and
                # no grounding is supplied, so only the LLM's own knowledge
                # separates the arms.
                views = [judge_mod.CandidateView(code=s)
                         for s in schema.ALL_SUBS if s not in detected]
                with_evidence = False
            else:  # A5
                views = evidence_views(fused, state["graph"], state["examples"])
                with_evidence = True

            outcome = await judge_mod.judge(
                transcript=rec.transcript, summary=rec.summary,
                prosody=rec.prosody, detected=detected,
                candidates=views, with_evidence=with_evidence)
            parse_fail = outcome.parse_fail
            hallucinated = outcome.hallucinated
            judgements = [asdict(j) for j in outcome.judgements]
            added = [j.code for j in outcome.judgements if j.confidence >= tau]

    final = detected | set(added)
    return {
        "clip": rec.clip, "clip_id": rec.clip_id, "speaker": rec.speaker,
        "arm": arm, "tau": tau if arm in LLM_ARMS else None,
        "transcript": rec.transcript,
        "gold": sorted(rec.gold), "detected": sorted(detected),
        "candidates": cand_dump, "judgements": judgements,
        "added": sorted(added), "final": sorted(final),
        "parse_fail": parse_fail, "hallucinated": hallucinated,
        **score_one(final, rec.gold, detected),
    }


async def run_arm(arm: str, records, art, freq: Counter, *, tau: float,
                  top_k: int, use_situational: bool,
                  concurrency: int = 8) -> list[dict]:
    """Score every clip for one arm, clips processed concurrently.

    Each clip is independent — its own graph reads, its own single judge call —
    so the wall clock is dominated by waiting on the API rather than by any
    ordering constraint. A0/A1/A2 do no I/O and run inline. Results are
    gathered positionally, so output order matches the input regardless of
    completion order.
    """
    if arm not in LLM_ARMS and arm != "A3":
        return [await _run_one(arm, rec, art, freq, tau=tau, top_k=top_k,
                               use_situational=use_situational)
                for rec in records]

    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0
    total = len(records)

    async def worker(rec):
        nonlocal done
        async with sem:
            row = await _run_one(arm, rec, art, freq, tau=tau, top_k=top_k,
                                 use_situational=use_situational)
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  {arm}: {done}/{total}", flush=True)
        return row

    return list(await asyncio.gather(*(worker(r) for r in records)))


def summarize(arm: str, rows: list[dict]) -> dict:
    n = len(rows) or 1
    inter = sum(len(set(r["final"]) & set(r["gold"])) for r in rows)
    tot_final = sum(len(r["final"]) for r in rows)
    tot_gold = sum(len(r["gold"]) for r in rows)
    cross_missing = sum(r["cross_major_missing"] for r in rows)
    cross_recovered = sum(r["cross_major_recovered"] for r in rows)
    return {
        "arm": arm,
        "macro_jaccard": round(statistics.fmean(r["jaccard"] for r in rows), 4),
        "macro_recall": round(statistics.fmean(r["recall"] for r in rows), 4),
        "macro_precision": round(statistics.fmean(r["precision"] for r in rows), 4),
        "micro_recall": round(inter / tot_gold, 4) if tot_gold else 0.0,
        "micro_precision": round(inter / tot_final, 4) if tot_final else 0.0,
        "avg_pred_size": round(tot_final / n, 2),
        "cross_major_recall": (round(cross_recovered / cross_missing, 4)
                               if cross_missing else None),
        "parse_fail": sum(1 for r in rows if r["parse_fail"]),
        "hallucinated": sum(len(r["hallucinated"]) for r in rows),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts/full100")
    ap.add_argument("--group", default="weekly_last",
                    choices=sorted(dataset_pending.GROUPS)
                    + ["weekly_last", "weekly_before"])
    ap.add_argument("--gold-tier", default="accept",
                    choices=["target", "accept", "union"],
                    help="target: majority of 5 (>=3); accept: >=2 annotators; "
                         "union: any annotator")
    ap.add_argument("--arms", default="A0,A1,A2,A3")
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=rules.CANDIDATE_TOP_K)
    ap.add_argument("--limit", type=int, default=0, help="first N clips (smoke)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="clips judged in parallel (A3/A4/A5)")
    ap.add_argument("--holdout-of", default="",
                    help="artifact dir whose holdout.json restricts the clips; "
                         "use with --group legacy100 to tune tau on speakers "
                         "that are absent from that build's graph and index")
    ap.add_argument("--no-situational", action="store_true",
                    help="graph source only; use when the build has no chunk index")
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ALL_ARMS]
    if bad:
        raise SystemExit(f"unknown arms: {bad} (choose from {ALL_ARMS})")

    if args.group.startswith("weekly_"):
        import labels_kr

        clips = labels_kr.load_batch(args.group.split("_", 1)[1])
        print("batch stats:",
              json.dumps(labels_kr.report(clips), ensure_ascii=False))
        records = labels_kr.to_records(clips, args.gold_tier)
    else:
        records = dataset_pending.load_records(args.group)
    if args.holdout_of:
        hp = Path(args.holdout_of)
        hp = (HERE / hp if not hp.is_absolute() else hp) / "holdout.json"
        with open(hp, encoding="utf-8") as f:
            keep = {row["clip_id"] for row in json.load(f)}
        records = [r for r in records if r.clip_id in keep]
        print(f"restricted to {len(records)} holdout clips from {hp}")
    if args.limit:
        records = records[: args.limit]
    print(json.dumps(dataset_pending.coverage_report(records),
                     ensure_ascii=False, indent=2))
    tag = f"{args.group}_{args.gold_tier}"

    art = await retrieve.open_artifacts(HERE / args.artifacts
                                        if not Path(args.artifacts).is_absolute()
                                        else args.artifacts)
    freq = global_frequency(art.labels_map)
    RESULTS.mkdir(exist_ok=True)

    summaries, per_arm = [], {}
    try:
        for arm in arms:
            print(f"running {arm} ...")
            rows = await run_arm(arm, records, art, freq, tau=args.tau,
                                 top_k=args.top_k,
                                 use_situational=not args.no_situational,
                                 concurrency=args.concurrency)
            out = RESULTS / f"{tag}_{arm}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            per_arm[arm] = rows
            summaries.append(summarize(arm, rows))
            print("  ", json.dumps(summaries[-1], ensure_ascii=False))
    finally:
        await art.rag.finalize_storages()

    # Paired CI against the baseline: the arms score the same clips, so the
    # comparison must be paired rather than treating them as independent means.
    deltas = {}
    if "A0" in per_arm:
        base = [r["jaccard"] for r in per_arm["A0"]]
        for arm, rows in per_arm.items():
            if arm == "A0":
                continue
            deltas[f"{arm}-A0"] = paired_bootstrap(
                [r["jaccard"] for r in rows], base)
    if "A4" in per_arm and "A5" in per_arm:
        deltas["A5-A4"] = paired_bootstrap(
            [r["jaccard"] for r in per_arm["A5"]],
            [r["jaccard"] for r in per_arm["A4"]])

    summary = {
        "group": args.group,
        "gold_tier": args.gold_tier,
        "tau": args.tau,
        "top_k": args.top_k,
        "concurrency": args.concurrency,
        "situational": not args.no_situational,
        "build_info": art.build_info,
        "arms": summaries,
        "paired_delta_jaccard": deltas,
    }
    with open(RESULTS / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
