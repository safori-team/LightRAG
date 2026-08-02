"""Build a visualizable utterance-case KG from the weekly before batch.

The graph uses the currently selected W1 weighting:
``weight = annotator votes / 5``.  Every observed label is retained so the UI
can show disagreement; ``label_votes`` is also written into the description.
This artifact is separate from the production/evaluation graph.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import aggregate
import build_layer1_v2
import labels_kr
import providers
import schema

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"


def case_entity(clip: labels_kr.Clip) -> dict:
    source_id = f"utt_{clip.clip_id}"
    description = (
        f"발화: {clip.transcript}\n"
        f"맥락 요약: {clip.summary}\n"
        f"화자: {clip.speaker}"
    )
    return {
        "entity_name": f"UTTERANCE_CASE_{clip.stem}",
        "entity_type": "UtteranceCase",
        "description": description,
        "source_id": source_id,
    }


def expresses_edges(clip: labels_kr.Clip) -> list[dict]:
    source_id = f"utt_{clip.clip_id}"
    case_id = f"UTTERANCE_CASE_{clip.stem}"
    return [
        {
            "src_id": case_id,
            "tgt_id": emotion,
            "keywords": "expresses",
            "weight": round(votes / 5.0, 4),
            "description": (
                f"이 발화에서 {schema.SUB_KR[emotion]}({emotion})을(를) "
                f"라벨러 {votes}/5명이 선택함. W1={votes / 5.0:.1f}"
            ),
            "source_id": source_id,
        }
        for emotion, votes in sorted(clip.sub_votes().items())
    ]


def case_chunk(clip: labels_kr.Clip) -> dict:
    return {
        "content": f"{clip.transcript}\n상황 요약: {clip.summary}",
        "source_id": f"utt_{clip.clip_id}",
        "chunk_order_index": 0,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="wk_before_case_w1_gemini_1536")
    args = ap.parse_args()

    providers.load_env()
    clips = labels_kr.load_batch("before")
    rows = labels_kr.to_aggregate_rows(clips)
    cooccurs, directional, agg_stats = aggregate.aggregate_cooccurs(rows)

    entities = schema.taxonomy_entities() + [case_entity(c) for c in clips]
    belongs = schema.belongs_to_edges()
    expresses = [edge for c in clips for edge in expresses_edges(c)]
    chunks = [case_chunk(c) for c in clips]
    custom_kg = {
        "entities": entities,
        "relationships": belongs + cooccurs + expresses,
        "chunks": chunks,
    }

    out = ARTIFACTS / args.run
    working_dir = out / "rag_storage"
    working_dir.mkdir(parents=True, exist_ok=True)
    rag = await build_layer1_v2.initialize_rag(working_dir)
    try:
        await rag.ainsert_custom_kg(custom_kg)
    finally:
        await rag.finalize_storages()

    stats = {
        "run": args.run,
        "backend": providers.describe(),
        "clips": len(clips),
        "entities": len(entities),
        "taxonomy_entities": 54,
        "utterance_case_entities": len(clips),
        "belongs_to": len(belongs),
        "co_occurs": len(cooccurs),
        "expresses": len(expresses),
        "chunks_input": len(chunks),
        "weighting": "W1=label_votes/5",
        "aggregate": agg_stats,
    }
    (out / "build_info.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "cooccur_stats.json").write_text(
        json.dumps(directional, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
