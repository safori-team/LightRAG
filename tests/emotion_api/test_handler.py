"""Handler behaviour with the LightRAG query mocked out.

No Gemini call, no graph load. The real end-to-end path is exercised by
``deploy/lambda_native/test-local.sh`` (billable) instead.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from emotion_api import config, handler, schemas, service

# Every test here is offline: no Gemini call, no graph load, no Docker.
pytestmark = pytest.mark.offline

VALID_EVENT = {
    "request_id": "231436",
    "gemini_result": {
        "transcript": "오늘 시험 결과가 생각보다 잘 나와서 기분이 좋아요.",
        "major": ["HAPPY"],
        "detected": [{"code": "JOY", "confidence": 0.75}],
    },
}


@pytest.fixture
def configured(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    storage = artifact / service.STORAGE_DIRNAME
    storage.mkdir(parents=True)
    for name in service.REQUIRED_STORAGE_FILES:
        (storage / name).write_text("{}", encoding="utf-8")
    (artifact / "build_info.json").write_text(
        json.dumps({"run": "split455_train273_case_w1_gemini_1536"}), encoding="utf-8"
    )
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_DIR", str(artifact))
    monkeypatch.setenv("LIGHTRAG_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("EMOTION_API_TOKEN", raising=False)
    config.settings.cache_clear()
    yield artifact
    config.settings.cache_clear()


@pytest.fixture
def fake_query(monkeypatch):
    """Replace only the LightRAG call; validation and normalization stay real."""
    calls: list[dict] = []

    def install(reply):
        async def fake_aselect(request):
            settings = config.settings()
            calls.append(
                {"query": request.transcript, "param": service.query_param(request, settings)}
            )
            if isinstance(reply, Exception):
                raise service.UpstreamError("simulated") from reply
            values = service.parse_minor_categories(reply)
            if values is None:
                return schemas.success_response(
                    request,
                    schemas.normalize_minor_categories(
                        [{"code": c, "confidence": s} for c, s in request.confidences.items()]
                    ),
                    mode=settings.mode,
                    model=settings.llm_model,
                    fallback="parse_failed",
                )
            minor = schemas.normalize_minor_categories(values, request.confidences)
            return schemas.success_response(
                request, minor, mode=settings.mode, model=settings.llm_model
            )

        monkeypatch.setattr(
            service, "select", lambda request: _run(fake_aselect(request))
        )
        return calls

    return install


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_successful_selection_returns_contract_shape(configured, fake_query):
    fake_query(
        '{"minor_categories":[{"code":"JOY","confidence":0.75},'
        '{"code":"SATISFACTION","confidence":0.63}]}'
    )
    response = handler.lambda_handler(VALID_EVENT, None)
    assert response == {
        "request_id": "231436",
        "minor_categories": [
            {"code": "JOY", "confidence": 0.75},
            {"code": "SATISFACTION", "confidence": 0.63},
        ],
        "meta": {
            "engine": "lightrag_native_query",
            "mode": "hybrid",
            "model": "gemini-2.5-flash",
        },
    }


def test_request_id_survives_the_round_trip(configured, fake_query):
    fake_query('{"minor_categories":[{"code":"JOY","confidence":0.5}]}')
    response = handler.lambda_handler({**VALID_EVENT, "request_id": "abc-999"}, None)
    assert response["request_id"] == "abc-999"


def test_codes_outside_the_official_taxonomy_are_dropped(configured, fake_query):
    fake_query(
        '{"minor_categories":[{"code":"JOY","confidence":0.7},'
        '{"code":"HAPPINESS_MADE_UP","confidence":0.9},'
        '{"code":"HAPPY","confidence":0.8}]}'  # major code, not a minor code
    )
    response = handler.lambda_handler(VALID_EVENT, None)
    assert [item["code"] for item in response["minor_categories"]] == ["JOY"]


def test_code_fence_wrapped_reply_is_parsed(configured, fake_query):
    fake_query('```json\n{"minor_categories":[{"code":"PRIDE","confidence":0.6}]}\n```')
    response = handler.lambda_handler(VALID_EVENT, None)
    assert [item["code"] for item in response["minor_categories"]] == ["PRIDE"]


def test_unparseable_reply_falls_back_to_the_input_emotions(configured, fake_query):
    fake_query("모델이 설명 문장만 돌려주었습니다.")
    response = handler.lambda_handler(VALID_EVENT, None)
    assert response["minor_categories"] == [{"code": "JOY", "confidence": 0.75}]
    assert response["meta"]["fallback"] == "parse_failed"


def test_query_uses_the_stored_gemini_result_as_predefined_keywords(configured, fake_query):
    calls = fake_query('{"minor_categories":[{"code":"JOY","confidence":0.5}]}')
    handler.lambda_handler(VALID_EVENT, None)
    param = calls[0]["param"]
    assert param.hl_keywords == ["HAPPY"]
    assert param.ll_keywords == ["JOY"]
    assert param.mode == "hybrid"


@pytest.mark.parametrize(
    "event, code",
    [
        ({"gemini_result": VALID_EVENT["gemini_result"]}, "MISSING_REQUEST_ID"),
        ({"request_id": "1"}, "INVALID_GEMINI_RESULT"),
        (
            {"request_id": "1", "gemini_result": {"major": ["HAPPY"], "detected": [{"code": "JOY"}]}},
            "MISSING_TRANSCRIPT",
        ),
        (
            {"request_id": "1", "gemini_result": {"transcript": "x", "major": [], "detected": []}},
            "INVALID_MAJOR",
        ),
        (
            {"request_id": "1", "gemini_result": {"transcript": "x", "major": ["HAPPY"], "detected": []}},
            "INVALID_DETECTED",
        ),
        (
            {
                "request_id": "1",
                "gemini_result": {
                    "transcript": "x",
                    "major": ["HAPPY"],
                    "detected": [{"code": "NOPE", "confidence": 0.9}],
                },
            },
            "INVALID_DETECTED",
        ),
    ],
)
def test_input_errors_are_400_class(configured, event, code):
    response = handler.lambda_handler(event, None)
    assert response["status"] == 400
    assert response["error"]["code"] == code


def test_gemini_failure_is_5xx_class(configured, fake_query):
    fake_query(RuntimeError("gemini exploded"))
    response = handler.lambda_handler(VALID_EVENT, None)
    assert response["status"] == 502
    assert response["error"]["code"] == "UPSTREAM_ERROR"


def test_missing_artifact_is_5xx_class(monkeypatch, tmp_path):
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_DIR", str(tmp_path / "absent"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config.settings.cache_clear()
    try:
        response = handler.lambda_handler({"action": "health"}, None)
        assert response["status"] == 500
        assert response["error"]["code"] == "ARTIFACT_ERROR"
    finally:
        config.settings.cache_clear()


def test_error_bodies_never_leak_internals(configured, fake_query):
    fake_query(RuntimeError("key=AIzaSyEXAMPLE at /var/task/emotion_api/service.py"))
    body = json.dumps(handler.lambda_handler(VALID_EVENT, None))
    for secret in ("AIzaSy", "/var/task", "taxonomy", "user_prompt", "test-key"):
        assert secret not in body


def test_health_action_reports_the_bundled_artifact(configured):
    response = handler.lambda_handler({"action": "health"}, None)
    assert response["ok"] is True
    assert response["artifact_version"] == "split455_train273_case_w1_gemini_1536"
    assert (response["mode"], response["top_k"], response["chunk_top_k"]) == ("hybrid", 8, 8)


def test_function_url_proxy_event_returns_status_code(configured, fake_query):
    fake_query('{"minor_categories":[{"code":"JOY","confidence":0.9}]}')
    response = handler.lambda_handler(
        {"requestContext": {"http": {"method": "POST"}}, "body": json.dumps(VALID_EVENT)}, None
    )
    assert response["statusCode"] == 200
    assert json.loads(response["body"])["request_id"] == "231436"


def test_function_url_malformed_body_is_400(configured):
    response = handler.lambda_handler(
        {"requestContext": {"http": {"method": "POST"}}, "body": "{not json"}, None
    )
    assert response["statusCode"] == 400
    assert json.loads(response["body"])["error"]["code"] == "MALFORMED_JSON"


def test_api_token_is_enforced_when_configured(configured, monkeypatch, fake_query):
    fake_query('{"minor_categories":[{"code":"JOY","confidence":0.9}]}')
    monkeypatch.setenv("EMOTION_API_TOKEN", "s3cret")
    config.settings.cache_clear()
    event = {"requestContext": {"http": {}}, "body": json.dumps(VALID_EVENT)}
    assert handler.lambda_handler(event, None)["statusCode"] == 401
    event["headers"] = {"x-emotion-api-token": "s3cret"}
    assert handler.lambda_handler(event, None)["statusCode"] == 200


def test_production_path_never_imports_the_experiment_judge():
    """The separate judge is not part of the native query path."""
    root = Path(__file__).resolve().parents[2] / "emotion_api"
    forbidden = re.compile(r"^\s*(?:from|import)\s+[\w.]*judge", re.MULTILINE)
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not forbidden.search(source), f"{path.name} imports the judge module"
    assert "judge" not in sys.modules
