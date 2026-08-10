"""Request validation and response construction for the emotion API.

Pure module: no LightRAG, no network. Everything here is unit testable.

Request contract (see ``deploy/lambda_native/events/``)::

    {"request_id": "231436",
     "gemini_result": {"transcript": "...",
                       "major": ["HAPPY"],
                       "detected": [{"code": "JOY", "confidence": 0.75}]}}

``major_category`` / ``minor_categories`` are accepted as aliases for ``major``
/ ``detected`` because earlier callers used those names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from emotion_api import taxonomy

ENGINE = "lightrag_native_query"


class RequestError(ValueError):
    """Client-side (4xx) validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class EmotionRequest:
    request_id: str
    transcript: str
    major: list[str]
    detected: list[str]
    confidences: dict[str, float] = field(default_factory=dict)
    summary: str = ""
    prosody: str = ""


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _code_of(item: Any) -> str:
    if isinstance(item, str):
        return item.strip().upper()
    if isinstance(item, dict):
        return str(item.get("code") or item.get("name") or "").strip().upper()
    return ""


def parse_request(event: dict[str, Any]) -> EmotionRequest:
    """Validate the incoming event, or raise :class:`RequestError` (4xx)."""
    if not isinstance(event, dict):
        raise RequestError("INVALID_REQUEST", "request body must be a JSON object")

    request_id = event.get("request_id")
    if not isinstance(request_id, (str, int)) or not str(request_id).strip():
        raise RequestError("MISSING_REQUEST_ID", "request_id is required")

    gemini = event.get("gemini_result")
    if not isinstance(gemini, dict):
        raise RequestError(
            "INVALID_GEMINI_RESULT", "gemini_result must be a JSON object"
        )

    transcript = gemini.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise RequestError(
            "MISSING_TRANSCRIPT", "gemini_result.transcript must be a non-empty string"
        )

    raw_major = gemini.get("major", gemini.get("major_category"))
    major = sorted({
        code for code in (_code_of(item) for item in _as_list(raw_major))
        if taxonomy.is_known_major(code)
    })
    if not major:
        raise RequestError(
            "INVALID_MAJOR",
            "gemini_result.major must contain at least one official major category",
        )

    raw_detected = gemini.get("detected", gemini.get("minor_categories"))
    confidences: dict[str, float] = {}
    for item in _as_list(raw_detected):
        code = _code_of(item)
        if not taxonomy.is_known_sub(code):
            continue
        score = _clamp(item.get("confidence") if isinstance(item, dict) else 0.0)
        confidences[code] = max(confidences.get(code, 0.0), score)
    if not confidences:
        raise RequestError(
            "INVALID_DETECTED",
            "gemini_result.detected must contain at least one official minor category",
        )

    return EmotionRequest(
        request_id=str(request_id).strip(),
        transcript=transcript.strip(),
        major=major,
        detected=sorted(confidences),
        confidences=confidences,
        summary=str(gemini.get("summary") or "").strip(),
        prosody=str(gemini.get("prosody") or "").strip(),
    )


def normalize_minor_categories(
    items: Any, fallback_confidences: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    """Keep official codes only, drop duplicates, clamp confidence to [0, 1].

    A code the model returned without a usable confidence inherits the caller's
    original Gemini confidence when there is one, so an omitted field never
    silently collapses a real detection to zero.
    """
    fallback = fallback_confidences or {}
    scores: dict[str, float] = {}
    for item in _as_list(items):
        code = _code_of(item)
        if not taxonomy.is_known_sub(code):
            continue
        raw = item.get("confidence") if isinstance(item, dict) else None
        default = fallback.get(code, 0.5)
        score = _clamp(raw, default) if raw is not None else default
        scores[code] = max(scores.get(code, 0.0), score)
    return [
        {"code": code, "confidence": round(score, 4)}
        for code, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def success_response(
    request: EmotionRequest,
    minor_categories: list[dict[str, Any]],
    *,
    mode: str,
    model: str,
    fallback: str = "",
) -> dict[str, Any]:
    meta: dict[str, Any] = {"engine": ENGINE, "mode": mode, "model": model}
    if fallback:
        meta["fallback"] = fallback
    return {
        "request_id": request.request_id,
        "minor_categories": minor_categories,
        "meta": meta,
    }


def error_response(
    status: int, code: str, message: str, request_id: str = ""
) -> dict[str, Any]:
    """Build an error body. Never includes prompts, context, paths, or keys."""
    return {
        "request_id": request_id,
        "error": {"code": code, "message": message},
        "status": status,
    }
