"""Re-score a stored judge run at other tau values — no LLM calls.

eval_pending.py stores every candidate's confidence, not just the accepted
ones, so the acceptance threshold can be varied afterwards. That separation is
what makes an honest protocol affordable: sweep tau on a tuning group, fix it,
and apply that one value to the test group's stored run.

    python rescore.py results/legacy100_A5.jsonl --sweep
    python rescore.py results/pending100_A5.jsonl --tau 0.25

Sweeping on the group you intend to report is fitting the test set; the sweep
output labels itself accordingly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import eval_pending
import schema


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def regold(rows: list[dict], batch: str, tier: str) -> list[dict]:
    """Swap in a different gold tier by clip_id — no LLM calls involved.

    The stored judgements do not depend on the gold standard, so target /
    accept / union can all be reported from a single run. Only the scoring
    changes, which is exactly what the tier is.
    """
    import labels_kr

    min_votes = {"target": labels_kr.TARGET_MIN,
                 "accept": labels_kr.ACCEPT_MIN, "union": 1}[tier]
    gold_by_clip = {c.clip_id: c.gold(min_votes)
                    for c in labels_kr.load_batch(batch)}
    return [{**r, "gold": sorted(gold_by_clip.get(r["clip_id"], set()))}
            for r in rows]


def rescore(rows: list[dict], tau: float) -> list[dict]:
    out = []
    for r in rows:
        detected = set(r["detected"])
        gold = set(r["gold"])
        added = sorted(j["code"] for j in r["judgements"]
                       if j["confidence"] >= tau)
        final = detected | set(added)
        out.append({**r, "tau": tau, "added": added, "final": sorted(final),
                    **eval_pending.score_one(final, gold, detected)})
    return out


def sweep(rows: list[dict], taus: list[float]) -> list[dict]:
    return [eval_pending.summarize(f"tau={t:.2f}", rescore(rows, t)) for t in taus]


def confidence_profile(rows: list[dict]) -> dict:
    """Where the model actually puts its confidence — a sweep grid that misses
    the mass would silently report 'the judge adds nothing'."""
    vals = [j["confidence"] for r in rows for j in r["judgements"]]
    if not vals:
        return {"n": 0}
    vals.sort()
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]  # noqa: E731
    return {
        "n": len(vals),
        "min": vals[0], "p25": q(0.25), "median": q(0.5),
        "p75": q(0.75), "p90": q(0.9), "max": vals[-1],
        "mean": round(statistics.fmean(vals), 3),
        "share_ge_0.5": round(sum(v >= 0.5 for v in vals) / len(vals), 3),
    }


def hit_rate_by_confidence(rows: list[dict]) -> list[dict]:
    """Is confidence informative at all? Precision of the judged candidates in
    each confidence band, against gold."""
    bands = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    out = []
    for lo, hi in bands:
        hit = tot = 0
        for r in rows:
            gold = set(r["gold"])
            for j in r["judgements"]:
                if lo <= j["confidence"] < hi:
                    tot += 1
                    hit += j["code"] in gold
        out.append({"band": f"[{lo},{hi})", "n": tot,
                    "in_gold": round(hit / tot, 3) if tot else None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--regold", nargs=2, metavar=("BATCH", "TIER"),
                    help="rescore against another gold tier, e.g. --regold last union")
    ap.add_argument("--write", action="store_true",
                    help="overwrite the jsonl with the --tau re-scoring")
    args = ap.parse_args()

    rows = load_rows(args.jsonl)
    if args.regold:
        batch, tier = args.regold
        rows = regold(rows, batch, tier)
        print(f"gold tier -> {tier} (batch {batch})")
    print("confidence profile:",
          json.dumps(confidence_profile(rows), ensure_ascii=False))
    print("candidate quality by band:",
          json.dumps(hit_rate_by_confidence(rows), ensure_ascii=False, indent=2))

    if args.sweep:
        grid = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
        print(f"\ntau sweep on {args.jsonl.name} "
              f"(tuning use only — selecting tau here and reporting the same "
              f"group is fitting the test set):")
        for s in sweep(rows, grid):
            print("  ", json.dumps(s, ensure_ascii=False))

    if args.tau is not None:
        scored = rescore(rows, args.tau)
        print(f"\ntau={args.tau}:",
              json.dumps(eval_pending.summarize("rescored", scored),
                         ensure_ascii=False))
        if args.write:
            with open(args.jsonl, "w", encoding="utf-8") as f:
                for r in scored:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"rewrote {args.jsonl}")


if __name__ == "__main__":
    main()
