"""Regression guard for the native query configuration.

The 91-record evaluation in ``results/split455_native_query/summary.json`` was
produced with an exact QueryParam and prompt. These tests fail if production
drifts away from it, because such a drift silently invalidates that result.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from emotion_api import config, prompts, schemas, service, taxonomy

# Every test here is offline: no Gemini call, no graph load, no Docker.
pytestmark = pytest.mark.offline

EXPERIMENT = (
    Path(__file__).resolve().parents[2]
    / "examples/emotion_graphrag/local_experiments/native_query_455.py"
)

REQUEST = schemas.EmotionRequest(
    request_id="231436",
    transcript="오늘 시험 결과가 생각보다 잘 나왔어요.",
    major=["HAPPY"],
    detected=["JOY", "PRIDE"],
    confidences={"JOY": 0.75, "PRIDE": 0.6},
)


def _settings(monkeypatch) -> config.Settings:
    monkeypatch.setenv("LIGHTRAG_ARTIFACT_DIR", "/nonexistent")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    config.settings.cache_clear()
    return config.settings()


def test_default_query_param_matches_the_evaluated_configuration(monkeypatch):
    param = service.query_param(REQUEST, _settings(monkeypatch))
    assert param.mode == "hybrid"
    assert param.top_k == 8
    assert param.chunk_top_k == 8
    assert param.enable_rerank is False
    assert param.response_type == "JSON object"
    assert param.user_prompt == prompts.USER_PROMPT


def test_keywords_are_predefined_so_no_extraction_llm_call_is_made(monkeypatch):
    param = service.query_param(REQUEST, _settings(monkeypatch))
    assert param.hl_keywords == ["HAPPY"]
    assert param.ll_keywords == ["JOY", "PRIDE"]


EXPERIMENT_SCHEMA = EXPERIMENT.parents[1] / "schema.py"


def test_taxonomy_has_the_official_shape():
    assert len(taxonomy.ALL_SUBS) == 48
    assert len(taxonomy.ALL_MAJORS) == 6
    assert set(taxonomy.SUB_KR) == set(taxonomy.SUB2MAJOR)


@pytest.mark.skipif(
    not EXPERIMENT_SCHEMA.is_file(), reason="experiment schema not on this branch"
)
def test_taxonomy_matches_the_experiment_codebook():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_exp_schema", EXPERIMENT_SCHEMA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert taxonomy.TAXONOMY == module.TAXONOMY


@pytest.mark.skipif(not EXPERIMENT.is_file(), reason="experiment script not on this branch")
def test_user_prompt_is_byte_identical_to_the_experiment():
    source = EXPERIMENT.read_text(encoding="utf-8")
    body = re.search(r'USER_PROMPT = f"""(.*?)"""', source, re.DOTALL).group(1)
    expected = body.replace("{taxonomy_text()}", taxonomy.taxonomy_text()).replace(
        "{{", "{"
    ).replace("}}", "}")
    assert prompts.USER_PROMPT == expected


def test_query_states_the_gemini_result_is_already_stored():
    query = prompts.build_query(REQUEST)
    assert query.startswith("아래는 이미 저장되어 있는 Gemini 음성 분석 결과다.")
    assert "새 음성을 분석하는 요청이 아니다" in query
    assert REQUEST.transcript in query
    assert "Gemini 대분류: HAPPY" in query
    assert "Gemini 소분류: JOY, PRIDE" in query


def test_prompt_restricts_output_to_official_codes():
    assert "아래 공식 taxonomy 코드만 출력한다" in prompts.USER_PROMPT
    assert "개수를 억지로 3~5개에 맞추지 않는다" in prompts.USER_PROMPT
    for code in taxonomy.ALL_SUBS:
        assert code in prompts.USER_PROMPT
