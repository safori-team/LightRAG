"""Run the same Gemini judge over W31 KG candidate arms."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from dataclasses import asdict
from pathlib import Path

import judge
from eval_w31_kg import DEFAULT_CSV, DEFAULT_JSON, load_records, set_metrics

HERE = Path(__file__).resolve().parent
ARMS = ("cooccurs", "utterance_W1", "context_W1")


def views_for(row, arm):
    views = []
    for code in row[arm]["candidates"]:
        if arm == "cooccurs":
            views.append(judge.CandidateView(
                code=code,
                graph_line="학습 라벨에서 1차 감정과 함께 나타난 감정",
            ))
            continue
        evidence = row[arm].get("evidence", {}).get(code, [])
        best = max(evidence, key=lambda x: x.get("similarity", 0), default=None)
        sit = ""
        if best:
            sit = (
                f'유사 사례 "{best["content"][:100]}" '
                f'(유사도 {best["similarity"]}, 라벨 W1 {best["edge_weight"]})'
            )
        views.append(judge.CandidateView(code=code, sit_line=sit))
    return views


def summarize(rows, arm, keep=lambda row: True):
    selected = [r for r in rows if keep(r)]
    vals = [r["judge"][arm] for r in selected]
    return {
        "n": len(vals),
        "macro_jaccard": round(statistics.fmean(v["metrics"]["jaccard"] for v in vals), 4),
        "macro_recall": round(statistics.fmean(v["metrics"]["recall"] for v in vals), 4),
        "macro_precision": round(statistics.fmean(v["metrics"]["precision"] for v in vals), 4),
        "avg_pred_size": round(statistics.fmean(v["metrics"]["n_pred"] for v in vals), 2),
        "avg_added": round(statistics.fmean(len(v["added"]) for v in vals), 2),
        "parse_fail": sum(v["parse_fail"] for v in vals),
        "hallucinated": sum(len(v["hallucinated"]) for v in vals),
    }


def bootstrap(rows, a, b, keep=lambda row: True, n=5000, seed=13):
    selected = [r for r in rows if keep(r)]
    x = [r["judge"][a]["metrics"]["jaccard"] for r in selected]
    y = ([r["judge"][b]["metrics"]["jaccard"] for r in selected]
         if b != "A0" else [r["A0_metrics"]["jaccard"] for r in selected])
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(len(x)) for _ in x]
        diffs.append(statistics.fmean(x[i] - y[i] for i in idx))
    diffs.sort()
    return {"delta": round(statistics.fmean(x)-statistics.fmean(y), 4),
            "ci_low": round(diffs[int(.025*n)], 4),
            "ci_high": round(diffs[int(.975*n)], 4)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path,
                    default=HERE / "results/w31/kg_candidate_rows.jsonl")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    records = {r.clip_id: r for r in load_records(DEFAULT_CSV, DEFAULT_JSON)}
    sem = asyncio.Semaphore(args.concurrency)

    async def judge_one(row, arm):
        rec = records[row["clip_id"]]
        last_error = None
        for attempt in range(5):
            try:
                async with sem:
                    outcome = await judge.judge(
                        transcript=rec.transcript, summary=rec.summary,
                        prosody=rec.prosody, detected=rec.detected,
                        candidates=views_for(row, arm), with_evidence=True,
                    )
                break
            except Exception as exc:
                last_error = exc
                if attempt == 4:
                    raise
                await asyncio.sleep(3 * (attempt + 1))
        accepted = sorted(j.code for j in outcome.judgements
                          if j.confidence >= args.tau)
        final = rec.detected | set(accepted)
        return {
            "judgements": [asdict(j) for j in outcome.judgements],
            "added": accepted, "final": sorted(final),
            "metrics": set_metrics(final, rec.gold(2)),
            "parse_fail": outcome.parse_fail,
            "hallucinated": outcome.hallucinated,
        }

    async def one(row):
        rec = records[row["clip_id"]]
        row["A0_metrics"] = set_metrics(rec.detected, rec.gold(2))
        results = await asyncio.gather(*(judge_one(row, arm) for arm in ARMS))
        row["judge"] = dict(zip(ARMS, results))
        return row

    out_dir = HERE / "results/w31"
    partial_path = out_dir / "kg_judge_partial.jsonl"
    judged = ([json.loads(line) for line in partial_path.read_text().splitlines()]
              if partial_path.exists() else [])
    completed = {r["clip_id"] for r in judged}
    pending = [r for r in rows if r["clip_id"] not in completed]
    for start in range(0, len(pending), args.concurrency):
        judged.extend(await asyncio.gather(
            *(one(r) for r in pending[start:start+args.concurrency])))
        with partial_path.open("w", encoding="utf-8") as f:
            for saved in judged:
                f.write(json.dumps(saved, ensure_ascii=False) + "\n")
        print(f"judged {len(judged)}/{len(rows)}", flush=True)

    clean = lambda r: not r["exact_train_transcript"]
    strict = lambda r: clean(r) and not r["train_speaker_overlap"]
    def group(keep):
        a0 = [r for r in judged if keep(r)]
        return {
            "A0": {
                "n": len(a0),
                "macro_jaccard": round(statistics.fmean(r["A0_metrics"]["jaccard"] for r in a0), 4),
                "macro_recall": round(statistics.fmean(r["A0_metrics"]["recall"] for r in a0), 4),
                "macro_precision": round(statistics.fmean(r["A0_metrics"]["precision"] for r in a0), 4),
                "avg_pred_size": round(statistics.fmean(r["A0_metrics"]["n_pred"] for r in a0), 2),
            },
            **{arm: summarize(judged, arm, keep) for arm in ARMS},
        }
    output = {
        "tau": args.tau,
        "warning": "tau=0.5 was not tuned for Gemini on an independent holdout",
        "all_100": group(lambda r: True),
        "clean_no_exact_98": group(clean),
        "strict_no_exact_or_speaker_overlap": group(strict),
        "paired_jaccard": {
            "utterance_W1-A0_clean": bootstrap(judged, "utterance_W1", "A0", clean),
            "context_W1-A0_clean": bootstrap(judged, "context_W1", "A0", clean),
            "utterance_W1-cooccurs_clean": bootstrap(judged, "utterance_W1", "cooccurs", clean),
            "context_W1-cooccurs_clean": bootstrap(judged, "context_W1", "cooccurs", clean),
        },
    }
    (out_dir / "kg_judge_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "kg_judge_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in judged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
