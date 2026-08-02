"""Evaluate utterance-case label weights with the existing weekly data.

This is a candidate-stage experiment.  It deliberately makes no judge calls:
the question is whether a retrieved case graph can place the missing human
labels inside a small top-k candidate set.  Three edge weights are compared:

W0: accept-tier binary edge (at least two of five annotators)
W1: raw vote ratio
W2: reliability-aware latent-label posterior (Dawid-Skene EM)

The existing ``wk_before`` transcript vector index supplies nearest cases.
The test set is the disjoint ``last`` batch.  A context-enhanced query appends
Gemini's semantic summary; it is reported as derived context, not as a manually
validated CONTEXT_* node.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import labels_kr
import retrieve
import schema

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _clip_observations(clips):
    return {
        c.clip_id: {
            e: {r.annotator: int(e in r.subs) for r in c.readings}
            for e in schema.ALL_SUBS
        }
        for c in clips
    }


def dawid_skene(clips, *, iterations: int = 80, prior_strength: float = 4.0):
    """Binary multi-label Dawid-Skene with major-level reliability pooling.

    Each emotion keeps its own prevalence, while annotator sensitivity and
    specificity are pooled within the emotion's major category.  Beta(2, 2)
    pseudo-counts shrink estimates away from unstable 0/1 values.
    """
    obs = _clip_observations(clips)
    annotators = labels_kr.ANNOTATORS
    post = {}
    for c in clips:
        votes = c.sub_votes()
        for e in schema.ALL_SUBS:
            post[(c.clip_id, e)] = (votes.get(e, 0) + 1.0) / 7.0

    sens = {(a, m): 0.7 for a in annotators for m in schema.ALL_MAJORS}
    spec = {(a, m): 0.9 for a in annotators for m in schema.ALL_MAJORS}
    prevalence = {e: 0.1 for e in schema.ALL_SUBS}
    alpha = beta = prior_strength / 2

    for _ in range(iterations):
        for e in schema.ALL_SUBS:
            vals = [post[(c.clip_id, e)] for c in clips]
            prevalence[e] = (sum(vals) + 1.0) / (len(vals) + 2.0)

        for a in annotators:
            for major in schema.ALL_MAJORS:
                emos = [e for e in schema.ALL_SUBS if schema.SUB2MAJOR[e] == major]
                tp = fn = tn = fp = 0.0
                for c in clips:
                    for e in emos:
                        p = post[(c.clip_id, e)]
                        x = obs[c.clip_id][e][a]
                        tp += p * x
                        fn += p * (1 - x)
                        tn += (1 - p) * (1 - x)
                        fp += (1 - p) * x
                sens[(a, major)] = (tp + alpha) / (tp + fn + alpha + beta)
                spec[(a, major)] = (tn + alpha) / (tn + fp + alpha + beta)

        max_delta = 0.0
        for c in clips:
            for e in schema.ALL_SUBS:
                major = schema.SUB2MAJOR[e]
                pi = min(0.999, max(0.001, prevalence[e]))
                log1, log0 = math.log(pi), math.log(1 - pi)
                for a in annotators:
                    x = obs[c.clip_id][e][a]
                    se = min(0.999, max(0.001, sens[(a, major)]))
                    sp = min(0.999, max(0.001, spec[(a, major)]))
                    log1 += math.log(se if x else 1 - se)
                    log0 += math.log(1 - sp if x else sp)
                p = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, log0 - log1))))
                key = (c.clip_id, e)
                max_delta = max(max_delta, abs(p - post[key]))
                post[key] = p
        if max_delta < 1e-7:
            break
    return post, sens, spec, prevalence


def case_weights(clips, posterior):
    out = {}
    for c in clips:
        votes = c.sub_votes()
        out[f"utt_{c.clip_id}"] = {
            "clip_id": c.clip_id,
            "W0": {e: float(v >= 2) for e, v in votes.items()},
            "W1": {e: v / 5.0 for e, v in votes.items()},
            # Keep only latent-positive edges.  Without a floor every emotion
            # has a tiny non-zero Bayesian posterior, which turns the graph
            # into a dense taxonomy lookup and inflates recall unfairly.
            "W2": {e: posterior[(c.clip_id, e)] for e in schema.ALL_SUBS
                   if posterior[(c.clip_id, e)] >= 0.5},
        }
    return out


def _sim(rank, rec):
    return retrieve._similarity(rank, rec)


async def neighbours(art, text: str, top_k: int):
    recs = await art.rag.chunks_vdb.query(text, top_k=top_k)
    out = []
    for rank, rec in enumerate(recs):
        src = rec.get("source_id") or rec.get("full_doc_id") or rec.get("id")
        if src:
            out.append((src, _sim(rank, rec), rec.get("content") or ""))
    return out


def _ngrams(text: str, n: int = 3):
    clean = re.sub(r"\s+", " ", text.strip().lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}


def lexical_neighbours(train, text: str, top_k: int, *, context: bool):
    """Fully offline fallback used when the embedding quota is unavailable."""
    q = _ngrams(text)
    rows = []
    for c in train:
        case_text = c.transcript
        if context:
            case_text = f"{case_text}\n상황 요약: {c.summary}"
        d = _ngrams(case_text)
        sim = len(q & d) / math.sqrt(max(1, len(q)) * max(1, len(d)))
        rows.append((f"utt_{c.clip_id}", sim, case_text))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:top_k]


def rank_cases(rows, weights, method, detected, top_k):
    allowed_majors = {schema.SUB2MAJOR[e] for e in detected if e in schema.SUB2MAJOR}
    scores = defaultdict(float)
    evidence = defaultdict(list)
    denom = sum(max(0.0, sim) for _, sim, _ in rows) or 1.0
    for src, sim, text in rows:
        entry = weights.get(src, {}).get(method, {})
        for emo, weight in entry.items():
            if emo in detected or weight <= 0:
                continue
            if allowed_majors and schema.SUB2MAJOR.get(emo) not in allowed_majors:
                continue
            value = max(0.0, sim) * weight / denom
            scores[emo] += value
            evidence[emo].append({"case": src, "sim": round(sim, 4),
                                   "edge_weight": round(weight, 4),
                                   "text": text[:100]})
    ranked = sorted(scores, key=lambda e: (-scores[e], e))[:top_k]
    return ranked, scores, evidence


def metrics(candidates, gold, detected):
    missing = set(gold) - set(detected)
    hits = set(candidates) & missing
    return {
        "candidate_recall": len(hits) / len(missing) if missing else 1.0,
        "candidate_precision": len(hits) / len(candidates) if candidates else 0.0,
        "hit": bool(hits), "n_missing": len(missing), "n_hit": len(hits),
    }


def bootstrap_delta(a, b, n=5000, seed=13):
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(len(a)) for _ in a]
        diffs.append(statistics.fmean(a[i] - b[i] for i in idx))
    diffs.sort()
    return {"delta": round(statistics.fmean(a) - statistics.fmean(b), 4),
            "ci_low": round(diffs[int(.025*n)], 4),
            "ci_high": round(diffs[int(.975*n)], 4)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/wk_before")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--neighbours", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--backend", choices=["lexical", "lightrag"],
                    default="lexical")
    args = ap.parse_args()

    train = labels_kr.load_batch("before")
    test = labels_kr.load_batch("last")
    posterior, sens, spec, prevalence = dawid_skene(train)
    weights = case_weights(train, posterior)
    art = (await retrieve.open_artifacts(HERE / args.artifacts)
           if args.backend == "lightrag" else None)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(c):
        if args.backend == "lightrag":
            async with sem:
                plain, contextual = await asyncio.gather(
                    neighbours(art, c.transcript, args.neighbours),
                    neighbours(art, f"{c.transcript}\n상황 요약: {c.summary}", args.neighbours),
                )
        else:
            plain = lexical_neighbours(train, c.transcript, args.neighbours,
                                       context=False)
            contextual = lexical_neighbours(
                train, f"{c.transcript}\n상황 요약: {c.summary}",
                args.neighbours, context=True)
        result = {"clip_id": c.clip_id, "stem": c.stem,
                  "detected": sorted(c.detected), "gold": sorted(c.gold(2))}
        for query_name, rows in (("utterance", plain), ("context", contextual)):
            for method in ("W0", "W1", "W2"):
                ranked, scores, evidence = rank_cases(
                    rows, weights, method, c.detected, args.top_k)
                result[f"{query_name}_{method}"] = {
                    "candidates": ranked,
                    "scores": {e: round(scores[e], 6) for e in ranked},
                    "evidence": {e: evidence[e] for e in ranked},
                    **metrics(ranked, c.gold(2), c.detected),
                }
        return result

    try:
        rows = list(await asyncio.gather(*(one(c) for c in test)))
    finally:
        if art is not None:
            await art.rag.finalize_storages()

    summaries = {}
    keys = [f"{q}_{w}" for q in ("utterance", "context") for w in ("W0", "W1", "W2")]
    for key in keys:
        vals = [r[key] for r in rows]
        summaries[key] = {
            "macro_candidate_recall": round(statistics.fmean(v["candidate_recall"] for v in vals), 4),
            "macro_candidate_precision": round(statistics.fmean(v["candidate_precision"] for v in vals), 4),
            "hit_rate": round(statistics.fmean(float(v["hit"]) for v in vals), 4),
            "micro_missing_recall": round(sum(v["n_hit"] for v in vals) / max(1, sum(v["n_missing"] for v in vals)), 4),
            "avg_candidates": round(statistics.fmean(len(v["candidates"]) for v in vals), 2),
        }
    deltas = {
        "utterance_W2-W1": bootstrap_delta(
            [r["utterance_W2"]["candidate_recall"] for r in rows],
            [r["utterance_W1"]["candidate_recall"] for r in rows]),
        "context_W2-utterance_W2": bootstrap_delta(
            [r["context_W2"]["candidate_recall"] for r in rows],
            [r["utterance_W2"]["candidate_recall"] for r in rows]),
    }
    reliability = {
        a: {m: {"sensitivity": round(sens[(a,m)], 4),
                "specificity": round(spec[(a,m)], 4)} for m in schema.ALL_MAJORS}
        for a in labels_kr.ANNOTATORS
    }
    output = {"design": {"train": "before 100", "test": "last 100",
                         "gold": "accept >=2", "top_k": args.top_k,
                         "neighbours": args.neighbours,
                         "retrieval_backend": args.backend,
                         "major_filter": "majors represented in Gemini detected subs",
                         "context": "Gemini summary appended to query; derived, not manually labeled"},
              "summaries": summaries, "paired_deltas": deltas,
              "reliability": reliability, "prevalence": prevalence}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "case_weight_experiment.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "case_weight_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
