"""Environment-driven settings for the Lambda emotion selection service.

Every value has a default that reproduces the 91-record native query
experiment. Secrets are never defaulted: ``GEMINI_API_KEY`` must come from the
Lambda environment or Secrets Manager.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Experiment-locked query settings. Overriding these via environment changes
# retrieval behaviour and invalidates the recorded native query result.
DEFAULT_MODE = "hybrid"
DEFAULT_TOP_K = 8
DEFAULT_CHUNK_TOP_K = 8


class ConfigError(RuntimeError):
    """Raised when the container is missing required configuration."""


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path
    work_dir: Path
    llm_model: str
    embedding_model: str
    embedding_dim: int
    mode: str
    top_k: int
    chunk_top_k: int
    enable_rerank: bool
    log_level: str
    api_token: str

    @property
    def api_key(self) -> str:
        """Read the Gemini key lazily so an unset key never blocks a health check."""
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise ConfigError("GEMINI_API_KEY is not set")
        return key


@lru_cache(maxsize=1)
def settings() -> Settings:
    artifact_dir = os.getenv("LIGHTRAG_ARTIFACT_DIR", "").strip()
    if not artifact_dir:
        raise ConfigError("LIGHTRAG_ARTIFACT_DIR is not set")
    return Settings(
        artifact_dir=Path(artifact_dir),
        # Lambda's package directory is read-only; LightRAG writes cache and
        # status files into working_dir, so it must live under /tmp.
        work_dir=Path(os.getenv("LIGHTRAG_WORK_DIR", "/tmp/emotion-artifact")),
        llm_model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash").strip(),
        embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        ).strip(),
        embedding_dim=_int_env("EMBEDDING_DIM", 1536),
        mode=os.getenv("LIGHTRAG_MODE", DEFAULT_MODE).strip() or DEFAULT_MODE,
        top_k=_int_env("LIGHTRAG_TOP_K", DEFAULT_TOP_K),
        chunk_top_k=_int_env("LIGHTRAG_CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K),
        enable_rerank=False,
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        api_token=os.getenv("EMOTION_API_TOKEN", "").strip(),
    )
