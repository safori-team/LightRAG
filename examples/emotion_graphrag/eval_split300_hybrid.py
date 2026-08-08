"""Tune on validation-60 and evaluate once on test-60 for the train-180 KG."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import statistics
from dataclasses import asdict
from pathlib import Path

import networkx as nx

import judge
import retrieve
import schema
from eval_w31_kg import set_metrics
from split300_data import DEFAULT_SPLIT_DIR, SplitRecord, load_split

HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = HERE / "artifacts/split300_train180_case_w1_gemini_1536"


def cooccurs(graph: nx.Graph, record: SplitRecord, top_k: int):
    scores, evidence = {}, {}
    for detected in record.detected:
        if detected not in graph:
            continue
        for candidate, attrs in graph[detected].items():
            if attrs.get("keywords") != "co_occurs":
                continue
            if candidate in record.detected or not schema.is_known_sub(candidate):
                continue
            if record.major and schema.SUB2MAJOR[candidate] not in record.major:
                continue
            weight = float(attrs.get("weight", 0.0))
            if weight > scores.get(candidate, -1):
                scores[candidate] = weight
                evidence[candidate] = detected
    ranked = sorted(scores, key=lambda code: (-scores[code], code))[:top_k]
    return ranked, scores, evidence


def utterance_candidates(graph, neighbours, case_by_source, record, top_k):
    scores, evidence = {}, {}
    denom = sum(max(0.0, float(row.get("distance", 0))) for row in neighbours) or 1.0
    for row in neighbours:
        case = case_by_source.get(row.get("full_doc_id"))
        if not case:
            continue
        similarity = max(0.0, float(row.get("distance", 0)))
        for emotion, attrs in graph[case].items():
            if attrs.get("keywords") != "expresses":
                continue
            if emotion in record.detected or not schema.is_known_sub(emotion):
                continue
            if record.major and schema.SUB2MAJOR[emotion] not in record.major:
                continue
            weight = float(attrs.get("weight", 0.0))
            scores[emotion] = scores.get(emotion, 0.0) + similarity * weight / denom
            item = {"similarity": similarity, "edge_weight": weight,
                    "content": (row.get("content") or "")[:160]}
            if similarity > evidence.get(emotion, {}).get("similarity", -1):
                evidence[emotion] = item
    ranked = sorted(scores, key=lambda code: (-scores[code], code))[:top_k]
    return ranked, scores, evidence


def fuse(co_ranked, utt_ranked, top_k, rrf_k=60):
    scores, support = {}, {}
    for source, ranked in (("cooccurs", co_ranked), ("utterance", utt_ranked)):
        for rank, code in enumerate(ranked, 1):
            scores[code] = scores.get(code, 0.0) + 1.0 / (rrf_k + rank)
            support.setdefault(code, []).append(source)
    ranked = sorted(scores, key=lambda code: (-scores[code], -len(support[code]), code))
    return ranked[:top_k], support


def views(arm, candidates, co_scores, co_evidence, utt_evidence):
    output = []
    for code in candidates:
        graph_line = "감정 상관관계 후보에는 포함되지 않음"
        if code in co_scores:
            graph_line = (
                f"{co_evidence[code]}와 co_occurs 엣지로 연결됨 "
                f"(가중치 {co_scores[code]:.4f})"
            )
        sit_line = "유사 발화 후보에는 포함되지 않음"
        if code in utt_evidence:
            item = utt_evidence[code]
            sit_line = (
                f'유사 사례 "{item["content"][:100]}" '
                f'(유사도 {item["similarity"]:.4f}, expresses W1 {item["edge_weight"]:.4f})'
            )
        output.append(judge.CandidateView(
            code=code,
            graph_line=graph_line,
            sit_line=sit_line if arm == "hybrid" else "",
        ))
    return output


async def generate_candidates(artifact: Path, split_dir: Path, top_k: int, neighbours: int):
    graph = nx.read_graphml(artifact / "rag_storage/graph_chunk_entity_relation.graphml")
    train = load_split("train", split_dir)
    case_by_source = {
        f"utt_{record.clip_id}": f"UTTERANCE_CASE_{record.stem}" for record in train
    }
    art = await retrieve.open_artifacts(artifact)
    output = []
    try:
        for split in ("validation", "test"):
            records = load_split(split, split_dir)
            for record in records:
                neighbours_found = await art.rag.chunks_vdb.query(
                    record.transcript, top_k=neighbours
                )
                co_ranked, co_scores, co_evidence = cooccurs(graph, record, top_k)
                utt_ranked, _, utt_evidence = utterance_candidates(
                    graph, neighbours_found, case_by_source, record, top_k
                )
                hybrid, support = fuse(co_ranked, utt_ranked, top_k)
                output.append({
                    "clip_id": record.clip_id,
                    "split": split,
                    "A0_metrics": set_metrics(record.detected, record.gold()),
                    "cooccurs": {"candidates": co_ranked},
                    "hybrid": {
                        "candidates": hybrid,
                        "support": {code: support[code] for code in hybrid},
                        "utterance_similarity": {
                            code: round(utt_evidence.get(code, {}).get("similarity", 0.0), 6)
                            for code in hybrid
                        },
                    },
                    "views": {
                        "cooccurs": [asdict(item) for item in views(
                            "cooccurs", co_ranked, co_scores, co_evidence, utt_evidence
                        )],
                        "hybrid": [asdict(item) for item in views(
                            "hybrid", hybrid, co_scores, co_evidence, utt_evidence
                        )],
                    },
                })
    finally:
        await art.rag.finalize_storages()
    return output


async def run_judges(rows, split_dir, concurrency, partial_path):
    records = {
        record.clip_id: record
        for split in ("validation", "test") for record in load_split(split, split_dir)
    }
    semaphore = asyncio.Semaphore(concurrency)

    async def one_arm(row, arm):
        record = records[row["clip_id"]]
        candidate_views = [judge.CandidateView(**item) for item in row["views"][arm]]
        for attempt in range(5):
            try:
                async with semaphore:
                    outcome = await judge.judge(
                        transcript=record.transcript,
                        summary=record.summary,
                        prosody=record.prosody,
                        detected=record.detected,
                        candidates=candidate_views,
                        with_evidence=True,
                    )
                return {
                    "confidences": outcome.confidences,
                    "judgements": [asdict(item) for item in outcome.judgements],
                    "parse_fail": outcome.parse_fail,
                    "hallucinated": outcome.hallucinated,
                }
            except Exception:
                if attempt == 4:
                    raise
                await asyncio.sleep(3 * (attempt + 1))

    async def one(row):
        results = await asyncio.gather(
            one_arm(row, "cooccurs"), one_arm(row, "hybrid")
        )
        row["judge"] = dict(zip(("cooccurs", "hybrid"), results))
        return row

    completed_rows = ([json.loads(line) for line in partial_path.read_text().splitlines()]
                      if partial_path.exists() else [])
    completed = {row["clip_id"] for row in completed_rows}
    pending = [row for row in rows if row["clip_id"] not in completed]
    for start in range(0, len(pending), concurrency):
        completed_rows.extend(await asyncio.gather(
            *(one(row) for row in pending[start:start + concurrency])
        ))
        with partial_path.open("w", encoding="utf-8") as file:
            for row in completed_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"judged {len(completed_rows)}/{len(rows)}", flush=True)
    return completed_rows


def prediction(row, records, arm, params):
    record = records[row["clip_id"]]
    accepted = set()
    for code in row[arm]["candidates"]:
        confidence = row["judge"][arm]["confidences"].get(code, 0.0)
        if arm == "cooccurs":
            threshold = params["tau"]
        else:
            sources = row["hybrid"]["support"][code]
            kind = "both" if len(sources) == 2 else sources[0]
            threshold = params[f"tau_{kind}"]
            if kind == "utterance":
                similarity = row["hybrid"]["utterance_similarity"][code]
                if similarity < params["utterance_min_similarity"]:
                    continue
        if confidence >= threshold:
            accepted.add(code)
    final = record.detected | accepted
    return set_metrics(final, record.gold()), accepted


def aggregate_metrics(rows, records, arm, params):
    values = [prediction(row, records, arm, params) for row in rows]
    metrics = [value[0] for value in values]
    return {
        "n": len(rows),
        "macro_jaccard": statistics.fmean(item["jaccard"] for item in metrics),
        "macro_recall": statistics.fmean(item["recall"] for item in metrics),
        "macro_precision": statistics.fmean(item["precision"] for item in metrics),
        "avg_pred_size": statistics.fmean(item["n_pred"] for item in metrics),
        "avg_added": statistics.fmean(len(value[1]) for value in values),
    }


def choose_parameters(rows, records):
    taus = [round(value / 100, 2) for value in range(35, 81, 5)]
    co_options = [{"tau": tau} for tau in taus]
    co_best = max(
        co_options,
        key=lambda option: tuple(aggregate_metrics(
            rows, records, "cooccurs", option
        )[key] for key in ("macro_jaccard", "macro_recall", "macro_precision")),
    )
    hybrid_options = [
        {"tau_both": both, "tau_cooccurs": co, "tau_utterance": utterance,
         "utterance_min_similarity": similarity}
        for both, co, utterance, similarity in itertools.product(
            taus, taus, taus, (0.0, 0.6, 0.65, 0.7, 0.75)
        )
    ]
    hybrid_best = max(
        hybrid_options,
        key=lambda option: tuple(aggregate_metrics(
            rows, records, "hybrid", option
        )[key] for key in ("macro_jaccard", "macro_recall", "macro_precision")),
    )
    return co_best, hybrid_best


def bootstrap(test_rows, records, hybrid_params, co_params, n=5000, seed=13):
    differences = []
    for row in test_rows:
        hybrid, _ = prediction(row, records, "hybrid", hybrid_params)
        co, _ = prediction(row, records, "cooccurs", co_params)
        differences.append(hybrid["jaccard"] - co["jaccard"])
    rng = random.Random(seed)
    sampled = []
    for _ in range(n):
        sampled.append(statistics.fmean(
            differences[rng.randrange(len(differences))] for _ in differences
        ))
    sampled.sort()
    return {
        "delta": round(statistics.fmean(differences), 4),
        "ci_low": round(sampled[int(0.025 * n)], 4),
        "ci_high": round(sampled[int(0.975 * n)], 4),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "results/split300_performance")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--neighbours", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    candidate_path = out / "candidate_rows.jsonl"
    if candidate_path.exists():
        rows = [json.loads(line) for line in candidate_path.read_text().splitlines()]
    else:
        rows = await generate_candidates(
            args.artifact, args.split_dir, args.top_k, args.neighbours
        )
        with candidate_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    judged = await run_judges(
        rows, args.split_dir, args.concurrency, out / "judge_partial.jsonl"
    )
    records = {
        record.clip_id: record
        for split in ("validation", "test") for record in load_split(split, args.split_dir)
    }
    validation = [row for row in judged if row["split"] == "validation"]
    test = [row for row in judged if row["split"] == "test"]
    co_params, hybrid_params = choose_parameters(validation, records)
    a0_params = {}
    def a0_metrics(selected):
        values = [row["A0_metrics"] for row in selected]
        return {
            "n": len(values),
            "macro_jaccard": statistics.fmean(item["jaccard"] for item in values),
            "macro_recall": statistics.fmean(item["recall"] for item in values),
            "macro_precision": statistics.fmean(item["precision"] for item in values),
            "avg_pred_size": statistics.fmean(item["n_pred"] for item in values),
        }
    output = {
        "design": {
            "train": 180, "validation": 60, "test": 60,
            "grouping": (
                "source prefix + transcript similarity connected components"
                if "transcript" in args.split_dir.name
                else "source prefix (not speaker ID)"
            ),
            "split_dir": str(args.split_dir),
            "top_k": args.top_k, "neighbours": args.neighbours,
        },
        "selected_on_validation": {
            "cooccurs": co_params, "hybrid": hybrid_params,
        },
        "validation": {
            "A0": a0_metrics(validation),
            "cooccurs": aggregate_metrics(validation, records, "cooccurs", co_params),
            "hybrid": aggregate_metrics(validation, records, "hybrid", hybrid_params),
        },
        "test_once": {
            "A0": a0_metrics(test),
            "cooccurs": aggregate_metrics(test, records, "cooccurs", co_params),
            "hybrid": aggregate_metrics(test, records, "hybrid", hybrid_params),
        },
        "paired_test_jaccard_hybrid_minus_cooccurs": bootstrap(
            test, records, hybrid_params, co_params
        ),
        "judge_quality": {
            "parse_fail": sum(row["judge"][arm]["parse_fail"]
                              for row in judged for arm in ("cooccurs", "hybrid")),
            "hallucinated": sum(len(row["judge"][arm]["hallucinated"])
                                for row in judged for arm in ("cooccurs", "hybrid")),
        },
    }
    # Round presentation metrics while retaining selected parameters exactly.
    for split in ("validation", "test_once"):
        for arm in output[split].values():
            for key, value in list(arm.items()):
                if isinstance(value, float):
                    arm[key] = round(value, 4)
    (out / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "judge_rows.jsonl").open("w", encoding="utf-8") as file:
        for row in judged:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "judge_partial.jsonl").unlink(missing_ok=True)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
