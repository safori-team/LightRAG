"""Evaluate the frozen W1 utterance-case KG on the disjoint W31 batch."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

import labels_kr
import retrieve
import schema

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = Path(
    "/Users/hann/Project/SAFORI/labeled-data/exports/emotion_relabeling/"
    "weekly/2026-W31/snapshot_2026-08-02/manifests/"
    "team_reference_pending_100_by_annotator.csv"
)
DEFAULT_JSON = Path(
    "/Users/hann/Project/SAFORI/data/"
    "gemini-2026-08-02-team-reference-pending-100"
)


@dataclass
class Record:
    clip_id: str
    stem: str
    speaker: str
    transcript: str
    summary: str
    prosody: str
    detected: set[str]
    major: set[str]
    votes: Counter

    def gold(self, threshold: int = 2) -> set[str]:
        return {e for e, n in self.votes.items() if n >= threshold}


def _gemini_code(name: str) -> str:
    value = name.strip().lower()
    return "ANXIETY_GENERAL" if value == "anxiety" else value.upper()


def load_records(csv_path: Path, json_dir: Path) -> list[Record]:
    analyses = {}
    for path in json_dir.glob("*.json"):
        if path.name == "batch-summary.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        analyses[str(data["clipId"])] = data.get("result") or {}

    records = []
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            clip_id = row["clip_id"]
            result = analyses[clip_id]
            detected, majors, notes = set(), set(), []
            for seg in result.get("segments", []):
                category = (seg.get("category") or "").strip().upper()
                if category in schema.ALL_MAJORS:
                    majors.add(category)
                for emo in seg.get("emotions", []):
                    code = _gemini_code(emo.get("name") or "")
                    if schema.is_known_sub(code):
                        detected.add(code)
                note = (seg.get("prosody_notes") or "").strip()
                if note:
                    notes.append(note)
            if not majors:
                majors = {schema.SUB2MAJOR[e] for e in detected}
            votes = Counter()
            for annotator in labels_kr.ANNOTATORS:
                tags = (row.get(f"{annotator}_sub_tags") or "").split(";")
                for tag in {t.strip() for t in tags if t.strip()}:
                    code = labels_kr.SUB_KR2CODE.get(tag)
                    if code:
                        votes[code] += 1
            stem = Path(row["storage_key"]).stem
            records.append(Record(
                clip_id=clip_id, stem=stem, speaker=stem.split("_")[0],
                transcript=(result.get("transcript") or "").strip(),
                summary=(result.get("summary") or "").strip(),
                prosody=" ".join(notes), detected=detected, major=majors,
                votes=votes,
            ))
    return records


def graph_maps(graph_path: Path):
    graph = nx.read_graphml(graph_path)
    content_to_case = {}
    train_transcripts = set()
    train_speakers = set()
    for node, data in graph.nodes(data=True):
        if data.get("entity_type") != "UtteranceCase":
            continue
        desc = str(data.get("description", ""))
        if not desc.startswith("발화: "):
            continue
        body = desc.removeprefix("발화: ")
        content, _, speaker = body.partition("\n화자: ")
        # The case node says "맥락 요약" while its indexed chunk says
        # "상황 요약".  Normalize the presentation label before joining.
        indexed_content = content.replace("\n맥락 요약:", "\n상황 요약:", 1)
        content_to_case[indexed_content] = node
        transcript = content.split("\n맥락 요약:", 1)[0].strip()
        train_transcripts.add(transcript)
        if speaker:
            train_speakers.add(speaker.strip())
    return graph, content_to_case, train_transcripts, train_speakers


def cooccur_candidates(graph, rec: Record, top_k: int) -> list[str]:
    scores = {}
    for detected in rec.detected:
        if detected not in graph:
            continue
        for other, attrs in graph[detected].items():
            if attrs.get("keywords") != "co_occurs":
                continue
            if other in rec.detected or not schema.is_known_sub(other):
                continue
            if rec.major and schema.SUB2MAJOR[other] not in rec.major:
                continue
            scores[other] = max(scores.get(other, 0.0), float(attrs.get("weight", 0)))
    return sorted(scores, key=lambda e: (-scores[e], e))[:top_k]


def case_candidates(graph, content_to_case, neighbours, rec: Record,
                    top_k: int):
    scores = defaultdict(float)
    evidence = defaultdict(list)
    denom = sum(max(0.0, float(row.get("distance", 0))) for row in neighbours) or 1.0
    for row in neighbours:
        content = (row.get("content") or "").strip()
        case = content_to_case.get(content)
        if not case:
            continue
        sim = max(0.0, float(row.get("distance", 0)))
        for emotion, attrs in graph[case].items():
            if attrs.get("keywords") != "expresses":
                continue
            if emotion in rec.detected or not schema.is_known_sub(emotion):
                continue
            if rec.major and schema.SUB2MAJOR[emotion] not in rec.major:
                continue
            weight = float(attrs.get("weight", 0))
            scores[emotion] += sim * weight / denom
            evidence[emotion].append({
                "case": case, "similarity": round(sim, 4),
                "edge_weight": round(weight, 4), "content": content[:160],
            })
    ranked = sorted(scores, key=lambda e: (-scores[e], e))[:top_k]
    return ranked, {e: round(scores[e], 6) for e in ranked}, {
        e: evidence[e] for e in ranked
    }


def set_metrics(predicted: set[str], gold: set[str]) -> dict:
    inter, union = predicted & gold, predicted | gold
    return {
        "jaccard": len(inter) / len(union) if union else 0.0,
        "recall": len(inter) / len(gold) if gold else 0.0,
        "precision": len(inter) / len(predicted) if predicted else 0.0,
        "n_pred": len(predicted),
    }


def candidate_metrics(candidates, rec: Record) -> dict:
    missing = rec.gold(2) - rec.detected
    hit = set(candidates) & missing
    return {
        "missing_recall": len(hit) / len(missing) if missing else 1.0,
        "candidate_precision": len(hit) / len(candidates) if candidates else 0.0,
        "n_missing": len(missing), "n_hit": len(hit),
    }


def summarize(rows, key, keep=lambda row: True):
    selected = [r for r in rows if keep(r)]
    vals = [r[key] for r in selected]
    return {
        "n": len(selected),
        "macro_candidate_recall": round(statistics.fmean(v["missing_recall"] for v in vals), 4),
        "macro_candidate_precision": round(statistics.fmean(v["candidate_precision"] for v in vals), 4),
        "micro_missing_recall": round(sum(v["n_hit"] for v in vals) / max(1, sum(v["n_missing"] for v in vals)), 4),
        "macro_final_jaccard": round(statistics.fmean(v["final_metrics"]["jaccard"] for v in vals), 4),
        "macro_final_recall": round(statistics.fmean(v["final_metrics"]["recall"] for v in vals), 4),
        "macro_final_precision": round(statistics.fmean(v["final_metrics"]["precision"] for v in vals), 4),
        "avg_candidates": round(statistics.fmean(len(v["candidates"]) for v in vals), 2),
    }


def bootstrap(rows, a, b, keep=lambda row: True, n=5000, seed=13):
    selected = [r for r in rows if keep(r)]
    x = [r[a]["missing_recall"] for r in selected]
    y = [r[b]["missing_recall"] for r in selected]
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(len(x)) for _ in x]
        diffs.append(statistics.fmean(x[i] - y[i] for i in idx))
    diffs.sort()
    return {"delta": round(statistics.fmean(x) - statistics.fmean(y), 4),
            "ci_low": round(diffs[int(.025*n)], 4),
            "ci_high": round(diffs[int(.975*n)], 4)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--json-dir", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--artifacts", type=Path,
                    default=Path("artifacts/wk_before_case_w1_gemini_1536"))
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--neighbours", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    records = load_records(args.csv, args.json_dir)
    graph_path = args.artifacts / "rag_storage/graph_chunk_entity_relation.graphml"
    graph, content_to_case, train_transcripts, train_speakers = graph_maps(graph_path)
    art = await retrieve.open_artifacts(args.artifacts)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(rec):
        async with sem:
            utterance, context = await asyncio.gather(
                art.rag.chunks_vdb.query(rec.transcript, top_k=args.neighbours),
                art.rag.chunks_vdb.query(
                    f"{rec.transcript}\n상황 요약: {rec.summary}",
                    top_k=args.neighbours),
            )
        result = {
            "clip_id": rec.clip_id, "stem": rec.stem,
            "speaker": rec.speaker, "transcript": rec.transcript,
            "summary": rec.summary, "detected": sorted(rec.detected),
            "major": sorted(rec.major), "gold": sorted(rec.gold(2)),
            "exact_train_transcript": rec.transcript in train_transcripts,
            "train_speaker_overlap": rec.speaker in train_speakers,
            "A0": {"candidates": [], "missing_recall": 0.0,
                   "candidate_precision": 0.0,
                   "n_missing": len(rec.gold(2)-rec.detected), "n_hit": 0,
                   "final_metrics": set_metrics(rec.detected, rec.gold(2))},
        }
        co = cooccur_candidates(graph, rec, args.top_k)
        uw, us, ue = case_candidates(graph, content_to_case, utterance, rec, args.top_k)
        cw, cs, ce = case_candidates(graph, content_to_case, context, rec, args.top_k)
        for key, candidates, scores, evidence in (
            ("cooccurs", co, {}, {}), ("utterance_W1", uw, us, ue),
            ("context_W1", cw, cs, ce)):
            result[key] = {"candidates": candidates, "scores": scores,
                           "evidence": evidence,
                           **candidate_metrics(candidates, rec),
                           "final_metrics": set_metrics(
                               rec.detected | set(candidates), rec.gold(2))}
        return result

    try:
        rows = list(await asyncio.gather(*(one(r) for r in records)))
    finally:
        await art.rag.finalize_storages()

    keys = ["A0", "cooccurs", "utterance_W1", "context_W1"]
    clean = lambda r: not r["exact_train_transcript"]
    strict = lambda r: clean(r) and not r["train_speaker_overlap"]
    output = {
        "design": {"test": "W31 100", "train": "W30 before 100 frozen KG",
                   "gold": "accept >=2 of 4 available annotators",
                   "top_k": args.top_k, "neighbours": args.neighbours,
                   "embedding": "gemini-embedding-001/1536"},
        "coverage": {
            "records": len(rows),
            "exact_train_transcript": sum(r["exact_train_transcript"] for r in rows),
            "train_speaker_overlap": sum(r["train_speaker_overlap"] for r in rows),
            "avg_gold": round(statistics.fmean(len(r["gold"]) for r in rows), 2),
            "avg_detected": round(statistics.fmean(len(r["detected"]) for r in rows), 2),
        },
        "all_100": {k: summarize(rows, k) for k in keys},
        "clean_no_exact_98": {k: summarize(rows, k, clean) for k in keys},
        "strict_no_exact_or_speaker_overlap": {k: summarize(rows, k, strict) for k in keys},
        "paired_delta_context_minus_cooccurs": {
            "all": bootstrap(rows, "context_W1", "cooccurs"),
            "clean": bootstrap(rows, "context_W1", "cooccurs", clean),
            "strict": bootstrap(rows, "context_W1", "cooccurs", strict),
        },
    }
    out_dir = HERE / "results/w31"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kg_candidate_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "kg_candidate_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
