"""LightRAG native query service: cold-start initialization and selection.

Design constraints this module exists to enforce:

* The graph artifact baked into the image is read-only (Lambda's package
  directory is not writable), so ``rag_storage`` is copied to ``/tmp`` once per
  container and LightRAG's ``working_dir`` points there.
* Storages are initialized exactly once per container and are **never**
  finalized — closing them would break warm-invocation reuse.
* A single module-level event loop is reused across invocations; Lambda's
  runtime does not provide one, and creating a new loop per request would
  discard LightRAG's loop-bound locks and connection state.
* The only outbound API calls are the query embedding and the native query's
  final generation. No emotion re-detection, no separate judge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from functools import partial
from pathlib import Path
from typing import Any

from emotion_api import prompts, schemas
from emotion_api.config import Settings, settings

logger = logging.getLogger(__name__)

_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_RAG: Any = None
_LOOP: asyncio.AbstractEventLoop | None = None


class UpstreamError(RuntimeError):
    """The Gemini API or the LightRAG query failed (5xx)."""


class ArtifactError(RuntimeError):
    """The bundled graph artifact is missing or unusable (5xx)."""


REQUIRED_STORAGE_FILES = (
    "graph_chunk_entity_relation.graphml",
    "kv_store_text_chunks.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "vdb_relationships.json",
)

# The experiment pipeline emits these files under "rag_storage/". They are
# shipped as "graph_storage/" because the repository's .gitignore excludes
# every "rag_storage/" directory, which would otherwise keep the production
# artifact from being committed without a permanent `git add -f`.
STORAGE_DIRNAME = "graph_storage"


def artifact_version(config: Settings | None = None) -> str:
    config = config or settings()
    build_info = config.artifact_dir / "build_info.json"
    if not build_info.is_file():
        return "unknown"
    try:
        with build_info.open(encoding="utf-8") as file:
            return str(json.load(file).get("run", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "unknown"


def verify_artifact(config: Settings | None = None) -> dict[str, Any]:
    """Check the read-only artifact without importing LightRAG or calling out.

    Used by the image's build-time import check and by the ``health`` action.
    """
    config = config or settings()
    source = config.artifact_dir / STORAGE_DIRNAME
    missing = [name for name in REQUIRED_STORAGE_FILES if not (source / name).is_file()]
    if missing:
        raise ArtifactError(f"graph artifact is incomplete: {', '.join(missing)}")
    return {
        "artifact_version": artifact_version(config),
        "embedding_dim": config.embedding_dim,
        "mode": config.mode,
        "top_k": config.top_k,
        "chunk_top_k": config.chunk_top_k,
    }


def _stage_working_dir(config: Settings) -> Path:
    """Copy the read-only artifact into the writable /tmp working directory.

    Copies to a sibling path first and renames, so a container that dies
    mid-copy cannot leave a half-populated directory that a later cold start
    would treat as complete.
    """
    working = config.work_dir / STORAGE_DIRNAME
    if working.is_dir():
        return working
    source = config.artifact_dir / STORAGE_DIRNAME
    verify_artifact(config)
    staging = config.work_dir / f"{STORAGE_DIRNAME}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    staging.rename(working)
    logger.info("staged graph artifact to %s", working)
    return working


def _build_embedding_func(config: Settings):
    from lightrag.llm.gemini import gemini_embed
    from lightrag.utils import EmbeddingFunc

    return EmbeddingFunc(
        embedding_dim=config.embedding_dim,
        max_token_size=8192,
        # ``.func`` unwraps @wrap_embedding_func_with_attrs so the outer
        # EmbeddingFunc settings win — same as the experiment's providers.py.
        func=partial(
            gemini_embed.func,
            model=config.embedding_model,
            api_key=config.api_key,
            embedding_dim=config.embedding_dim,
        ),
    )


async def _create_rag(config: Settings):
    from lightrag import LightRAG
    from lightrag.llm.gemini import gemini_model_complete

    working = _stage_working_dir(config)
    rag = LightRAG(
        working_dir=str(working),
        workspace="",
        llm_model_func=gemini_model_complete,
        llm_model_name=config.llm_model,
        llm_model_kwargs={"api_key": config.api_key},
        embedding_func=_build_embedding_func(config),
        enable_llm_cache=False,
    )
    await rag.initialize_storages()
    logger.info("LightRAG storages initialized (artifact=%s)", artifact_version(config))
    return rag


async def _get_rag(config: Settings):
    global _RAG
    if _RAG is None:
        _RAG = await _create_rag(config)
    return _RAG


def query_param(request: schemas.EmotionRequest, config: Settings):
    """Build the QueryParam. Kept separate so the regression test can assert it.

    ``hl_keywords`` / ``ll_keywords`` are supplied directly from the stored
    Gemini result, which is what suppresses LightRAG's keyword-extraction LLM
    call.
    """
    from lightrag import QueryParam

    return QueryParam(
        mode=config.mode,
        top_k=config.top_k,
        chunk_top_k=config.chunk_top_k,
        enable_rerank=config.enable_rerank,
        response_type="JSON object",
        hl_keywords=list(request.major),
        ll_keywords=list(request.detected),
        user_prompt=prompts.USER_PROMPT,
    )


def parse_minor_categories(raw: str) -> list[Any] | None:
    """Extract ``minor_categories`` from the model's reply, or None on failure."""
    stripped = (raw or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    payload: Any = None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = _OBJECT.search(stripped)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    if not isinstance(payload, dict):
        return None
    values = payload.get("minor_categories")
    return values if isinstance(values, list) else None


def _echo_input(
    request: schemas.EmotionRequest, config: Settings, reason: str
) -> dict[str, Any]:
    """Defined fallback: return the caller's own emotions, flagged in meta."""
    return schemas.success_response(
        request,
        schemas.normalize_minor_categories(
            [{"code": code, "confidence": score}
             for code, score in request.confidences.items()]
        ),
        mode=config.mode,
        model=config.llm_model,
        fallback=reason,
    )


async def aselect(request: schemas.EmotionRequest) -> dict[str, Any]:
    config = settings()
    rag = await _get_rag(config)
    query = prompts.build_query(request)
    try:
        raw = await rag.aquery(query, param=query_param(request, config))
    except Exception as exc:  # provider/network/timeouts all land here
        logger.exception("native query failed")
        raise UpstreamError(f"{type(exc).__name__}") from exc

    # LightRAG swallows provider errors inside aquery and returns None rather
    # than raising, so an outage would otherwise be indistinguishable from a
    # model that replied with non-JSON. Treat it as the upstream failure it is.
    if raw is None:
        logger.error("native query returned no result (retrieval or provider failure)")
        raise UpstreamError("native query returned no result")

    text = str(raw).strip()
    if not text:
        logger.error("native query returned an empty response")
        raise UpstreamError("native query returned an empty response")

    # LightRAG's own "no usable context" sentinel. Retrieval ran but found
    # nothing, which is a degraded answer rather than an outage.
    if "[no-context]" in text:
        logger.warning("native query found no usable context")
        return _echo_input(request, config, "no_context")

    values = parse_minor_categories(text)
    if values is None:
        logger.warning("native query response was not parseable JSON")
        return _echo_input(request, config, "parse_failed")

    minor = schemas.normalize_minor_categories(values, request.confidences)
    if not minor:
        logger.warning("native query returned no official taxonomy code")
        return _echo_input(request, config, "no_official_code")
    return schemas.success_response(
        request, minor, mode=config.mode, model=config.llm_model
    )


def select(request: schemas.EmotionRequest) -> dict[str, Any]:
    """Synchronous entry point that reuses one event loop across invocations."""
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_LOOP)
    return _LOOP.run_until_complete(aselect(request))
