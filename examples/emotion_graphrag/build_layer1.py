"""Layer-1 offline build: labels -> custom_kg -> LightRAG (Ollama embeddings).

Injects the fixed taxonomy nodes, belongs_to hierarchy, data-aggregated
co_occurs edges, and per-utterance chunks via ``ainsert_custom_kg`` — bypassing
LightRAG's LLM extraction so gold labels are not diluted. Only the TRAIN split
is injected; the speaker-disjoint holdout is written out for evaluation.

Run:
    python build_layer1.py
Env (see ../.env / DESIGN.md §9):
    EMOTION_CSV, WORKING_DIR, EMBEDDING_MODEL, EMBEDDING_DIM,
    EMBEDDING_BINDING_HOST, LLM_MODEL, LLM_BINDING_HOST
"""

from __future__ import annotations

import asyncio
import json
import os
from functools import partial

from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

import aggregate
import schema

load_dotenv(dotenv_path=".env", override=False)

HERE = os.path.dirname(os.path.abspath(__file__))
EMOTION_CSV = os.getenv(
    "EMOTION_CSV",
    r"C:\Users\windowadmin6\Desktop\safori\data\emotion_experiment_dataset.csv",
)
WORKING_DIR = os.getenv("WORKING_DIR", os.path.join(HERE, "rag_storage"))
WORKSPACE = os.getenv("WORKSPACE", "emotion")
HOLDOUT_FRAC = float(os.getenv("HOLDOUT_FRAC", "0.2"))


async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        workspace=WORKSPACE,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "exaone3.5:7.8b"),
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": int(os.getenv("OLLAMA_LLM_NUM_CTX", "32768"))},
            "timeout": int(os.getenv("TIMEOUT", "300")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=partial(
                ollama_embed.func,  # unwrapped to avoid double EmbeddingFunc wrap
                embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )
    await rag.initialize_storages()
    return rag


def build_custom_kg(train_rows: list[dict]) -> tuple[dict, dict, dict]:
    """Assemble the custom_kg payload from the TRAIN split.

    Returns (custom_kg, directional_stats, debug_stats).
    """
    entities = schema.taxonomy_entities()                 # 54 fixed nodes
    belongs = schema.belongs_to_edges()                   # 48 hierarchy edges
    cooccurs, directional, agg_stats = aggregate.aggregate_cooccurs(train_rows)
    chunks = aggregate.utterance_chunks(train_rows)

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


async def main() -> None:
    rows = aggregate.load_records(EMOTION_CSV)
    train, holdout = aggregate.train_holdout_split(rows, HOLDOUT_FRAC)
    print(f"records={len(rows)}  train={len(train)}  holdout={len(holdout)}")

    custom_kg, directional, stats = build_custom_kg(train)
    print("custom_kg:", json.dumps(stats, ensure_ascii=False))

    rag = await initialize_rag()
    try:
        await rag.ainsert_custom_kg(custom_kg)
        print("ainsert_custom_kg done.")
    finally:
        await rag.finalize_storages()

    # Side artifacts: label lookup + directional co_occur stats (TRAIN only)
    # + holdout for evaluation.
    with open(os.path.join(HERE, "labels_map.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate.labels_map(train), f, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, "cooccur_stats.json"), "w", encoding="utf-8") as f:
        json.dump(directional, f, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, "holdout.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"clip_id": r["clip_id"], "speaker": aggregate.speaker_of(r),
              "transcript": r.get("transcript", ""),
              "gold_major": aggregate._json_list(r, "human_majority_major_codes"),
              "gold_sub": aggregate._json_list(r, "human_majority_sub_codes")}
             for r in holdout],
            f, ensure_ascii=False, indent=2,
        )
    print("wrote labels_map.json + cooccur_stats.json + holdout.json")


if __name__ == "__main__":
    asyncio.run(main())
