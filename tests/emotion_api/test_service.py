"""Service-level handling of what LightRAG's aquery actually returns.

Regression coverage for a real defect: ``aquery`` swallows provider errors and
returns ``None`` instead of raising, which an earlier version stringified into
``"None"`` and reported as a successful 200 response with a ``parse_failed``
fallback. A Gemini outage must surface as an upstream error, not as a
confident-looking answer echoing the caller's own input.
"""

from __future__ import annotations

import json

import pytest

from emotion_api import config, handler, schemas, service

pytestmark = pytest.mark.offline

REQUEST = schemas.EmotionRequest(
    request_id="231436",
    transcript="오늘 기분이 좋아요.",
    major=["HAPPY"],
    detected=["JOY"],
    confidences={"JOY": 0.7},
)

EVENT = {
    "request_id": "231436",
    "gemini_result": {
        "transcript": "오늘 기분이 좋아요.",
        "major": ["HAPPY"],
        "detected": [{"code": "JOY", "confidence": 0.7}],
    },
}


@pytest.fixture
def stub_rag(monkeypatch, tmp_path):
    """Configure the service and stub only ``rag.aquery``'s return value."""
    artifact = tmp_path / "artifact"
    storage = artifact / service.STORAGE_DIRNAME
    storage.mkdir(parents=True)
    for name in service.REQUIRED_STORAGE_FILES:
        (storage / name).write_text("{}", encoding="utf-8")
    (artifact / "build_info.json").write_text(json.dumps({"run": "test"}), encoding="utf-8")
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_DIR", str(artifact))
    monkeypatch.setenv("LIGHTRAG_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("EMOTION_API_TOKEN", raising=False)
    config.settings.cache_clear()

    def install(reply):
        class FakeRag:
            async def aquery(self, _query, param=None):
                if isinstance(reply, Exception):
                    raise reply
                return reply

        async def fake_get_rag(_config):
            return FakeRag()

        monkeypatch.setattr(service, "_get_rag", fake_get_rag)

    yield install
    config.settings.cache_clear()


async def test_none_result_is_an_upstream_error_not_a_fallback(stub_rag):
    stub_rag(None)
    with pytest.raises(service.UpstreamError):
        await service.aselect(REQUEST)


async def test_empty_result_is_an_upstream_error(stub_rag):
    stub_rag("   ")
    with pytest.raises(service.UpstreamError):
        await service.aselect(REQUEST)


async def test_raised_provider_error_is_an_upstream_error(stub_rag):
    stub_rag(RuntimeError("gemini refused the api key"))
    with pytest.raises(service.UpstreamError):
        await service.aselect(REQUEST)


async def test_no_context_sentinel_falls_back_instead_of_erroring(stub_rag):
    stub_rag("Sorry, I'm not able to provide an answer to that question.[no-context]")
    response = await service.aselect(REQUEST)
    assert response["meta"]["fallback"] == "no_context"
    assert response["minor_categories"] == [{"code": "JOY", "confidence": 0.7}]


async def test_valid_json_reply_is_returned_normally(stub_rag):
    stub_rag('{"minor_categories":[{"code":"SATISFACTION","confidence":0.62}]}')
    response = await service.aselect(REQUEST)
    assert response["minor_categories"] == [{"code": "SATISFACTION", "confidence": 0.62}]
    assert "fallback" not in response["meta"]


def test_provider_outage_reaches_the_caller_as_502(stub_rag):
    """End-to-end through the handler: an outage must never look like success."""
    stub_rag(None)
    response = handler.lambda_handler(EVENT, None)
    assert response["status"] == 502
    assert response["error"]["code"] == "UPSTREAM_ERROR"
    assert "minor_categories" not in response
