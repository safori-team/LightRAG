"""Build a case KG from only the source-prefix-disjoint train-180 split."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import aggregate
import build_layer1_v2
import providers
import schema
from split300_data import DEFAULT_SPLIT_DIR, SplitRecord, coverage, load_split

HERE = Path(__file__).resolve().parent


def aggregate_rows(records: list[SplitRecord]) -> list[dict]:
    rows = []
    for record in records:
        rows.append({
            "clip_id": record.clip_id,
            "filename": f"{record.stem}.wav",
            "transcript": record.transcript,
            "annotator_labels": json.dumps({
                name: {"major_codes": [], "sub_codes": sorted(labels)}
                for name, labels in record.readings.items()
            }, ensure_ascii=False),
            "human_majority_major_codes": "[]",
            "human_majority_sub_codes": json.dumps(sorted(record.gold()), ensure_ascii=False),
        })
    return rows


def case_entity(record: SplitRecord) -> dict:
    return {
        "entity_name": f"UTTERANCE_CASE_{record.stem}",
        "entity_type": "UtteranceCase",
        "description": (
            f"발화: {record.transcript}\n맥락 요약: {record.summary}\n"
            f"출처 prefix: {record.source_prefix}"
        ),
        "source_id": f"utt_{record.clip_id}",
    }


def expresses_edges(record: SplitRecord) -> list[dict]:
    denominator = len(record.readings)
    return [{
        "src_id": f"UTTERANCE_CASE_{record.stem}",
        "tgt_id": emotion,
        "keywords": "expresses",
        "weight": round(votes / denominator, 4),
        "description": (
            f"이 발화에서 {schema.SUB_KR[emotion]}({emotion})을(를) "
            f"라벨러 {votes}/{denominator}명이 선택함"
        ),
        "source_id": f"utt_{record.clip_id}",
    } for emotion, votes in sorted(record.votes().items())]


def case_chunk(record: SplitRecord) -> dict:
    return {
        "content": f"{record.transcript}\n상황 요약: {record.summary}",
        "source_id": f"utt_{record.clip_id}",
        "chunk_order_index": 0,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="split300_train180_case_w1_gemini_1536")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    args = parser.parse_args()
    providers.load_env()
    records = load_split("train", args.split_dir)
    rows = aggregate_rows(records)
    cooccurs, directional, aggregate_stats = aggregate.aggregate_cooccurs(rows)
    entities = schema.taxonomy_entities() + [case_entity(record) for record in records]
    belongs = schema.belongs_to_edges()
    expresses = [edge for record in records for edge in expresses_edges(record)]
    chunks = [case_chunk(record) for record in records]
    custom_kg = {
        "entities": entities,
        "relationships": belongs + cooccurs + expresses,
        "chunks": chunks,
    }
    out = HERE / "artifacts" / args.run
    working_dir = out / "rag_storage"
    working_dir.mkdir(parents=True, exist_ok=True)
    rag = await build_layer1_v2.initialize_rag(working_dir)
    try:
        await rag.ainsert_custom_kg(custom_kg)
    finally:
        await rag.finalize_storages()
    info = {
        "run": args.run,
        "backend": providers.describe(),
        "split": "source-prefix-disjoint train",
        "split_dir": str(args.split_dir),
        "coverage": coverage(records),
        "entities": len(entities),
        "co_occurs": len(cooccurs),
        "expresses": len(expresses),
        "chunks": len(chunks),
        "weighting": "W1=votes/available_annotators_per_clip",
        "aggregate": aggregate_stats,
    }
    (out / "build_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "cooccur_stats.json").write_text(
        json.dumps(directional, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
