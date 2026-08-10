"""AWS Lambda entry point for the emotion selection service.

Supports both invocation shapes:

* direct ``Invoke`` — the event *is* the request body; the response is the
  plain JSON body (errors carry a ``status`` field).
* Function URL / API Gateway proxy — the request body arrives as a JSON string
  in ``event["body"]``; the response is a proxy object with ``statusCode``.

Error bodies never contain prompts, retrieval context, file paths, or keys.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

from emotion_api import schemas, service
from emotion_api.config import ConfigError, settings

logger = logging.getLogger()


def _configure_logging() -> None:
    try:
        level = settings().log_level
    except ConfigError:
        level = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    logger.setLevel(getattr(logging, level, logging.INFO))


_configure_logging()


def _is_proxy_event(event: dict[str, Any]) -> bool:
    return isinstance(event, dict) and "requestContext" in event and "body" in event


def _proxy_body(event: dict[str, Any]) -> dict[str, Any]:
    body = event.get("body")
    if body is None:
        return {}
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    if isinstance(body, dict):
        return body
    try:
        parsed = json.loads(body)
    except (TypeError, json.JSONDecodeError) as exc:
        raise schemas.RequestError("MALFORMED_JSON", "request body is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise schemas.RequestError("MALFORMED_JSON", "request body must be a JSON object")
    return parsed


def _authorized(event: dict[str, Any]) -> bool:
    """Shared-secret check, enabled only when EMOTION_API_TOKEN is configured."""
    try:
        expected = settings().api_token
    except ConfigError:
        return True
    if not expected:
        return True
    headers = event.get("headers") or {}
    presented = ""
    if isinstance(headers, dict):
        presented = str(
            headers.get("x-emotion-api-token") or headers.get("X-Emotion-Api-Token") or ""
        )
    return presented == expected


def _health() -> dict[str, Any]:
    status = service.verify_artifact()
    return {"ok": True, "service": "safori-emotion-select", **status}


def _handle(event: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if event.get("action") == "health":
        return 200, _health()

    request = schemas.parse_request(event)
    return 200, service.select(request)


def lambda_handler(event: Any, _context: Any = None) -> dict[str, Any]:
    proxy = _is_proxy_event(event) if isinstance(event, dict) else False
    request_id = ""
    try:
        if not isinstance(event, dict):
            raise schemas.RequestError("INVALID_REQUEST", "event must be a JSON object")
        if proxy and not _authorized(event):
            status, body = 401, schemas.error_response(
                401, "UNAUTHORIZED", "missing or invalid API token"
            )
        else:
            payload = _proxy_body(event) if proxy else event
            request_id = str(payload.get("request_id") or "") if isinstance(payload, dict) else ""
            status, body = _handle(payload)
    except schemas.RequestError as exc:
        logger.warning("rejected request: %s", exc.code)
        status, body = 400, schemas.error_response(400, exc.code, exc.message, request_id)
    except service.ArtifactError as exc:
        logger.error("artifact error: %s", exc)
        status, body = 500, schemas.error_response(
            500, "ARTIFACT_ERROR", "graph artifact is unavailable", request_id
        )
    except ConfigError as exc:
        logger.error("configuration error: %s", exc)
        status, body = 500, schemas.error_response(
            500, "CONFIG_ERROR", "service is not configured correctly", request_id
        )
    except service.UpstreamError:
        status, body = 502, schemas.error_response(
            502, "UPSTREAM_ERROR", "emotion model backend failed", request_id
        )
    except Exception:
        logger.exception("unhandled error")
        status, body = 500, schemas.error_response(
            500, "INTERNAL_ERROR", "internal error", request_id
        )

    if not proxy:
        return body
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(body, ensure_ascii=False),
    }
