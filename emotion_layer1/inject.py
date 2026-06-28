"""Inject a custom_kg into LightRAG, offline (file backend, mock embedding).

Smoke / offline build needs no API key: the embedding is a deterministic
hash-based vector and the LLM func is never invoked (custom_kg insert and
graph-neighbor reads do not call the LLM). Swap ``make_rag`` for a real
embedding + Neo4j backend when moving past smoke.
"""

from __future__ import annotations

import hashlib

import numpy as np

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer

MOCK_DIM = 32


class _CharTokenizer:
    """Dependency-free tokenizer (no tiktoken / no network) for offline smoke."""

    def encode(self, content: str, **kwargs) -> list[int]:
        return list(content.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


async def _mock_embed(texts: list[str]) -> np.ndarray:
    """Deterministic hash-based embedding (reproducible, key-free)."""
    out = np.zeros((len(texts), MOCK_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vec = np.frombuffer((h * (MOCK_DIM // len(h) + 1))[:MOCK_DIM], dtype=np.uint8)
        out[i] = vec.astype(np.float32) / 255.0
    return out


async def _dummy_llm(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    # Never called during custom_kg insert or graph reads; present so init is valid.
    return ""


async def make_rag(working_dir: str) -> LightRAG:
    rag = LightRAG(
        working_dir=working_dir,
        tokenizer=Tokenizer(model_name="char", tokenizer=_CharTokenizer()),
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=MOCK_DIM,
            max_token_size=512,
            func=_mock_embed,
        ),
    )
    await rag.initialize_storages()
    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass
    return rag


async def inject(rag: LightRAG, custom_kg: dict) -> None:
    await rag.ainsert_custom_kg(custom_kg)
