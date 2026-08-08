"""Evaluate a fixed-size co-occurrence + utterance-evidence hybrid on W31."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from dataclasses import asdict
from pathlib import Path

import networkx as nx

import judge
import schema
from eval_w31_kg import DEFAULT_CSV, DEFAULT_JSON, load_records, set_metrics

HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH = (
    HERE / "artifacts/wk_before_case_w1_gemini_1536/rag_storage/"
    "graph_chunk_entity_relation.graphml"
)


def reciprocal_rank_fusion(row: dict, top_k: int, rrf_k: int = 60):
    """Fuse ranks without assuming co-occurrence and case scores share a scale."""
    rankings = (
        row["cooccurs"]["candidates"],
        row["utterance_W1"]["candidates"],
    )
    scores: dict[str, float] = {}
    support: dict[str, list[str]] = {}
    for source, ranked in zip(("cooccurs", "utterance"), rankings):
        for rank, code in enumerate(ranked, 1):
            scores[code] = scores.get(code, 0.0) + 1.0 / (rrf_k + rank)
            support.setdefault(code, []).append(source)
    candidates = sorted(
        scores,
        key=lambda code: (-scores[code], -len(support[code]), code),
    )[:top_k]
    return candidates, scores, support


def cooccur_line(graph: nx.Graph, detected: set[str], candidate: str) -> str:
    evidence = []
    for source in sorted(detected):
        if not graph.has_edge(source, candidate):
            continue
        attrs = graph[source][candidate]
        if attrs.get("keywords") != "co_occurs":
            continue
        evidence.append((float(attrs.get("weight", 0.0)), source))
    if not evidence:
        return "감정 상관관계 후보에는 포함되지 않음"
    weight, source = max(evidence)
    return f"{source}와 co_occurs 엣지로 연결됨 (가중치 {weight:.4f})"


def utterance_line(row: dict, candidate: str) -> str:
    evidence = row["utterance_W1"].get("evidence", {}).get(candidate, [])
    best = max(evidence, key=lambda item: item.get("similarity", 0), default=None)
    if not best:
        return "유사 발화 후보에는 포함되지 않음"
    return (
        f'유사 사례 "{best["content"][:100]}" '
        f'(유사도 {best["similarity"]}, expresses W1 {best["edge_weight"]})'
    )


def candidate_views(row: dict, graph: nx.Graph, top_k: int):
    candidates, scores, support = reciprocal_rank_fusion(row, top_k)
    detected = set(row["detected"])
    views = [
        judge.CandidateView(
            code=code,
            graph_line=cooccur_line(graph, detected, code),
            sit_line=utterance_line(row, code),
        )
        for code in candidates
    ]
    metadata = {
        "candidates": candidates,
        "rrf_scores": {code: round(scores[code], 8) for code in candidates},
        "support": {code: support[code] for code in candidates},
    }
    return views, metadata


def summarize(rows: list[dict], keep=lambda row: True):
    selected = [row for row in rows if keep(row)]
    vals = [row["hybrid"] for row in selected]
    return {
        "n": len(vals),
        "macro_jaccard": round(statistics.fmean(v["metrics"]["jaccard"] for v in vals), 4),
        "macro_recall": round(statistics.fmean(v["metrics"]["recall"] for v in vals), 4),
        "macro_precision": round(statistics.fmean(v["metrics"]["precision"] for v in vals), 4),
        "avg_pred_size": round(statistics.fmean(v["metrics"]["n_pred"] for v in vals), 2),
        "avg_added": round(statistics.fmean(len(v["added"]) for v in vals), 2),
        "avg_candidates": round(statistics.fmean(len(v["candidates"]) for v in vals), 2),
        "parse_fail": sum(v["parse_fail"] for v in vals),
        "hallucinated": sum(len(v["hallucinated"]) for v in vals),
    }


def bootstrap(rows, baseline_rows, baseline, keep, n=5000, seed=13):
    baseline_by_id = {row["clip_id"]: row for row in baseline_rows}
    selected = [row for row in rows if keep(row)]
    x = [row["hybrid"]["metrics"]["jaccard"] for row in selected]
    if baseline == "A0":
        y = [baseline_by_id[row["clip_id"]]["A0_metrics"]["jaccard"] for row in selected]
    else:
        y = [baseline_by_id[row["clip_id"]]["judge"][baseline]["metrics"]["jaccard"]
             for row in selected]
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        indexes = [rng.randrange(len(x)) for _ in x]
        diffs.append(statistics.fmean(x[i] - y[i] for i in indexes))
    diffs.sort()
    return {
        "delta": round(statistics.fmean(x) - statistics.fmean(y), 4),
        "ci_low": round(diffs[int(0.025 * n)], 4),
        "ci_high": round(diffs[int(0.975 * n)], 4),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path,
                        default=HERE / "results/w31/kg_candidate_rows.jsonl")
    parser.add_argument("--baseline-rows", type=Path,
                        default=HERE / "results/w31/kg_judge_rows.jsonl")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.rows.read_text().splitlines()]
    baseline_rows = [json.loads(line) for line in args.baseline_rows.read_text().splitlines()]
    records = {record.clip_id: record for record in load_records(DEFAULT_CSV, DEFAULT_JSON)}
    graph = nx.read_graphml(args.graph)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(row):
        record = records[row["clip_id"]]
        views, metadata = candidate_views(row, graph, args.top_k)
        last_error = None
        for attempt in range(5):
            try:
                async with semaphore:
                    outcome = await judge.judge(
                        transcript=record.transcript,
                        summary=record.summary,
                        prosody=record.prosody,
                        detected=record.detected,
                        candidates=views,
                        with_evidence=True,
                    )
                break
            except Exception as exc:
                last_error = exc
                if attempt == 4:
                    raise RuntimeError(f"judge failed for {record.clip_id}") from last_error
                await asyncio.sleep(3 * (attempt + 1))
        accepted = sorted(
            item.code for item in outcome.judgements if item.confidence >= args.tau
        )
        final = record.detected | set(accepted)
        return {
            "clip_id": row["clip_id"],
            "exact_train_transcript": row["exact_train_transcript"],
            "train_speaker_overlap": row["train_speaker_overlap"],
            "hybrid": {
                **metadata,
                "judgements": [asdict(item) for item in outcome.judgements],
                "added": accepted,
                "final": sorted(final),
                "metrics": set_metrics(final, record.gold(2)),
                "parse_fail": outcome.parse_fail,
                "hallucinated": outcome.hallucinated,
            },
        }

    out_dir = HERE / "results/w31"
    partial_path = out_dir / "kg_hybrid_partial.jsonl"
    judged = ([json.loads(line) for line in partial_path.read_text().splitlines()]
              if partial_path.exists() else [])
    completed = {row["clip_id"] for row in judged}
    pending = [row for row in rows if row["clip_id"] not in completed]
    for start in range(0, len(pending), args.concurrency):
        judged.extend(await asyncio.gather(
            *(one(row) for row in pending[start:start + args.concurrency])
        ))
        with partial_path.open("w", encoding="utf-8") as file:
            for saved in judged:
                file.write(json.dumps(saved, ensure_ascii=False) + "\n")
        print(f"judged {len(judged)}/{len(rows)}", flush=True)

    clean = lambda row: not row["exact_train_transcript"]
    strict = lambda row: clean(row) and not row["train_speaker_overlap"]
    output = {
        "design": {
            "arm": "cooccurs + utterance_W1",
            "fusion": "reciprocal rank fusion",
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "tau": args.tau,
            "warning": "parameters were fixed before this run, not tuned on W31",
        },
        "all_100": summarize(judged),
        "clean_no_exact_98": summarize(judged, clean),
        "strict_no_exact_or_speaker_overlap": summarize(judged, strict),
        "paired_jaccard_clean": {
            "hybrid-A0": bootstrap(judged, baseline_rows, "A0", clean),
            "hybrid-cooccurs": bootstrap(judged, baseline_rows, "cooccurs", clean),
            "hybrid-utterance_W1": bootstrap(
                judged, baseline_rows, "utterance_W1", clean
            ),
        },
    }
    (out_dir / "kg_hybrid_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "kg_hybrid_rows.jsonl").open("w", encoding="utf-8") as file:
        for row in judged:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    partial_path.unlink(missing_ok=True)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
