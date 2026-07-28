"""Layer-1 rebuild with a pluggable backend (openai / gemini / ollama).

Same graph contract as build_layer1.py — fixed taxonomy nodes, belongs_to
hierarchy, data-aggregated co_occurs edges, one chunk per training utterance,
all injected through ``ainsert_custom_kg`` so gold labels are never diluted by
LLM extraction. Three things differ:

  1. the backend comes from providers.py (repo-root .env), not hard-coded Ollama;
  2. transcripts are normalized (prosody markers stripped) before indexing, so
     the index matches the clean Gemini transcripts used as queries;
  3. every run writes into its own ``artifacts/<run>/`` directory — storage and
     side tables together — so builds never clobber each other and a result
     file can always be traced back to the graph that produced it.

Two runs are expected (see the evaluation plan):

    python build_layer1_v2.py --run train69 --holdout-frac 0.2   # tau tuning
    python build_layer1_v2.py --run full100 --holdout-frac 0     # final eval

Neither leaks the pending_100 test set: those clips are absent from the
training CSV entirely (verified: zero clip_id overlap).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from lightrag import LightRAG

import aggregate
import providers
import schema
from textnorm import normalize_transcript

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
DEFAULT_CSV = "/Users/hann/Project/SAFORI/data/emotion_experiment_dataset.csv"


def utterance_chunks_normalized(rows: list[dict]) -> list[dict]:
    """One chunk per record, with prosody markers stripped.

    aggregate.utterance_chunks indexes the raw transcript; here the text is
    normalized first so index and query live in the same surface form.
    """
    chunks: list[dict] = []
    for row in rows:
        text = normalize_transcript(row.get("transcript") or "")
        if not text:
            continue
        chunks.append(schema.utterance_chunk(row["clip_id"], text))
    return chunks


def build_custom_kg(train_rows: list[dict]) -> tuple[dict, dict, dict]:
    entities = schema.taxonomy_entities()          # 54 fixed nodes
    belongs = schema.belongs_to_edges()            # 48 hierarchy edges
    cooccurs, directional, agg_stats = aggregate.aggregate_cooccurs(train_rows)
    chunks = utterance_chunks_normalized(train_rows)

    custom_kg = {
        "entities": entities,
        "relationships": belongs + cooccurs,
        "chunks": chunks,
    }
    stats = {
        "entities": len(entities),
        "belongs_to": len(belongs),
        "co_occurs": len(cooccurs),
        "chunks": len(chunks),
        **agg_stats,
    }
    return custom_kg, directional, stats


async def initialize_rag(working_dir: Path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(working_dir),
        workspace="",  # flat layout; keep build and inference in agreement
        llm_model_func=providers.noop_llm,
        embedding_func=providers.make_embedding_func(),
    )
    await rag.initialize_storages()
    return rag


def _holdout_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "clip_id": r["clip_id"],
            "speaker": aggregate.speaker_of(r),
            "transcript": normalize_transcript(r.get("transcript", "")),
            "gold_major": aggregate._json_list(r, "human_majority_major_codes"),
            "gold_sub": [
                c for c in aggregate._json_list(r, "human_majority_sub_codes")
                if schema.is_known_sub(c)
            ],
        }
        for r in rows
    ]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="train69", help="artifact directory name")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--csv", default=os.getenv("EMOTION_CSV", DEFAULT_CSV))
    ap.add_argument("--source", default="csv", choices=["csv", "weekly-before"],
                    help="csv: the older 5-annotator experiment CSV; "
                         "weekly-before: labeling_before_last_week_100.csv, the "
                         "batch whose clips are disjoint from the test batch")
    args = ap.parse_args()

    providers.load_env()
    out_dir = ARTIFACTS / args.run
    working_dir = out_dir / "rag_storage"
    working_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "weekly-before":
        import labels_kr

        clips = labels_kr.load_batch("before")
        rows = labels_kr.to_aggregate_rows(clips)
        source_desc = "labeling_before_last_week_100.csv (5 annotators)"
        print("label source stats:",
              json.dumps(labels_kr.report(clips), ensure_ascii=False))
    else:
        rows = aggregate.load_records(args.csv)
        source_desc = f"{args.csv} (5 annotators)"
    print("label source:", source_desc)
    if args.holdout_frac > 0:
        train, holdout = aggregate.train_holdout_split(rows, args.holdout_frac)
    else:
        train, holdout = rows, []
    print(f"records={len(rows)}  train={len(train)}  holdout={len(holdout)}")
    print("backend:", json.dumps(providers.describe(), ensure_ascii=False))

    custom_kg, directional, stats = build_custom_kg(train)
    print("custom_kg:", json.dumps(stats, ensure_ascii=False))

    rag = await initialize_rag(working_dir)
    try:
        await rag.ainsert_custom_kg(custom_kg)
        print("ainsert_custom_kg done.")
    finally:
        await rag.finalize_storages()

    def _write(name: str, obj) -> None:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    _write("labels_map.json", aggregate.labels_map(train))
    _write("cooccur_stats.json", directional)
    _write("holdout.json", _holdout_rows(holdout))
    _write("build_info.json", {
        "run": args.run,
        "label_source": source_desc,
        "holdout_frac": args.holdout_frac,
        "n_train": len(train),
        "n_holdout": len(holdout),
        "backend": providers.describe(),
        "custom_kg": stats,
    })

    # The vector index is what situational retrieval depends on; a build that
    # silently produced no chunk vectors would score like a graph-only run.
    produced = sorted(p.name for p in working_dir.iterdir())
    print("storage files:", produced)
    if not any(n.startswith("vdb_chunks") for n in produced):
        raise SystemExit(
            "ERROR: no vdb_chunks file — chunk embedding did not complete. "
            "Situational retrieval (source 2) would be silently dead."
        )
    print(f"wrote artifacts to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
