"""Pluggable LLM / embedding backends for the emotion reviewer.

The original pipeline (build_layer1.py / infer.py) hard-codes Ollama. This
module keeps that option alive while adding OpenAI and Gemini, selected purely
by environment variables so no code changes when the backend changes:

    LLM_BINDING        = openai | gemini | ollama
    EMBEDDING_BINDING  = openai | gemini | ollama

Everything is read from the repository-root ``.env`` (not the example folder),
which is where this project already keeps its credentials.

Two things are exported:

  * ``make_embedding_func()``  -> ``EmbeddingFunc`` for LightRAG construction.
  * ``chat(prompt, json_mode=True)`` -> raw model text, used by the judge.

The judge deliberately talks to the provider SDK directly rather than through
``rag.llm_model_func``: LightRAG's wrapper expects an injected ``hashing_kv``
and applies its own caching/prompt scaffolding, neither of which we want for a
measured experiment.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from lightrag.utils import EmbeddingFunc

# Repository root = .../lightRAG (this file lives in examples/emotion_graphrag/)
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

_ENV_LOADED = False


def load_env() -> None:
    """Load the repo-root .env once. Existing env vars win (override=False)."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(dotenv_path=ENV_PATH, override=False)
        _ENV_LOADED = True


def _env(key: str, default: str = "") -> str:
    load_env()
    return os.getenv(key, default)


def llm_binding() -> str:
    return _env("LLM_BINDING", "openai").strip().lower()


def embedding_binding() -> str:
    return _env("EMBEDDING_BINDING", "openai").strip().lower()


def llm_model() -> str:
    return _env("LLM_MODEL", "gpt-5-mini").strip()


def embedding_model() -> str:
    return _env("EMBEDDING_MODEL", "text-embedding-3-large").strip()


def embedding_dim() -> int:
    return int(_env("EMBEDDING_DIM", "3072"))


def describe() -> dict:
    """Backend summary for run provenance (written into every result file)."""
    return {
        "llm_binding": llm_binding(),
        "llm_model": llm_model(),
        "embedding_binding": embedding_binding(),
        "embedding_model": embedding_model(),
        "embedding_dim": embedding_dim(),
    }


# --- embeddings ---------------------------------------------------------------
def make_embedding_func() -> EmbeddingFunc:
    """Build the EmbeddingFunc for the configured EMBEDDING_BINDING.

    Base functions decorated with @wrap_embedding_func_with_attrs are unwrapped
    via ``.func`` so the outer EmbeddingFunc settings are not overridden by the
    inner wrapper (see EmbeddingFunc docstring in lightrag/utils.py).
    """
    binding = embedding_binding()
    dim = embedding_dim()
    max_tokens = int(_env("MAX_EMBED_TOKENS", "8192"))

    if binding == "openai":
        from lightrag.llm.openai import openai_embed

        func = partial(
            openai_embed.func,
            model=embedding_model(),
            api_key=_env("EMBEDDING_BINDING_API_KEY") or _env("OPENAI_API_KEY"),
            base_url=_env("EMBEDDING_BINDING_HOST") or None,
        )
    elif binding == "gemini":
        from lightrag.llm.gemini import gemini_embed

        func = partial(
            gemini_embed.func,
            model=embedding_model(),
            api_key=_env("EMBEDDING_BINDING_API_KEY") or _env("GEMINI_API_KEY"),
        )
    elif binding == "ollama":
        from lightrag.llm.ollama import ollama_embed

        func = partial(
            ollama_embed.func,
            embed_model=embedding_model(),
            host=_env("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
        )
    else:
        raise ValueError(f"unsupported EMBEDDING_BINDING: {binding!r}")

    return EmbeddingFunc(embedding_dim=dim, max_token_size=max_tokens, func=func)


# --- LLM placeholder for LightRAG construction --------------------------------
async def noop_llm(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    """LightRAG requires an llm_model_func at construction time, but this
    pipeline never uses it: custom_kg insertion bypasses LLM extraction and all
    reads are deterministic graph / vector queries. The judge calls ``chat``
    below directly. Kept explicit so an accidental LLM path fails loudly."""
    raise RuntimeError(
        "LightRAG's llm_model_func was called, but this pipeline should never "
        "use it (custom_kg insert + deterministic reads only). Check the call site."
    )


# --- judge chat ---------------------------------------------------------------
def _omits_temperature(model: str) -> bool:
    """Newer OpenAI reasoning models reject an explicit temperature."""
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


async def chat(prompt: str, *, json_mode: bool = True,
               temperature: float = 0.2, timeout: int = 120) -> str:
    """Send one prompt, return raw text. Provider chosen by LLM_BINDING."""
    binding = llm_binding()
    model = llm_model()

    if binding == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=_env("LLM_BINDING_API_KEY") or _env("OPENAI_API_KEY"),
            base_url=_env("LLM_BINDING_HOST") or None,
            timeout=timeout,
        )
        kwargs: dict = {"model": model,
                        "messages": [{"role": "user", "content": prompt}]}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if not _omits_temperature(model):
            kwargs["temperature"] = temperature
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    if binding == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=_env("LLM_BINDING_API_KEY") or _env("GEMINI_API_KEY")
        )
        cfg = types.GenerateContentConfig(temperature=temperature)
        if json_mode:
            cfg.response_mime_type = "application/json"
        resp = await client.aio.models.generate_content(
            model=model, contents=prompt, config=cfg
        )
        return resp.text or ""

    if binding == "ollama":
        import ollama  # optional dependency; only imported on this path

        client = ollama.AsyncClient(
            host=_env("LLM_BINDING_HOST", "http://localhost:11434")
        )
        resp = await client.generate(
            model=model,
            prompt=prompt,
            format="json" if json_mode else None,
            options={"temperature": temperature,
                     "num_ctx": int(_env("OLLAMA_LLM_NUM_CTX", "32768"))},
        )
        return resp.get("response", "")

    raise ValueError(f"unsupported LLM_BINDING: {binding!r}")
