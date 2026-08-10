"""Request validation and response normalization for the emotion API."""

from __future__ import annotations

import pytest

from emotion_api import schemas

# Every test here is offline: no Gemini call, no graph load, no Docker.
pytestmark = pytest.mark.offline


def _event(**overrides):
    gemini = {
        "transcript": "오늘 시험 결과가 생각보다 잘 나와서 기분이 좋아요.",
        "major": ["HAPPY"],
        "detected": [{"code": "JOY", "confidence": 0.75}],
    }
    gemini.update(overrides.pop("gemini_result", {}))
    event = {"request_id": "231436", "gemini_result": gemini}
    event.update(overrides)
    return event


def test_parses_documented_request_shape():
    request = schemas.parse_request(_event())
    assert request.request_id == "231436"
    assert request.major == ["HAPPY"]
    assert request.detected == ["JOY"]
    assert request.confidences == {"JOY": 0.75}


def test_accepts_legacy_field_aliases():
    request = schemas.parse_request(
        {
            "request_id": 42,
            "gemini_result": {
                "transcript": "괜찮아요.",
                "major_category": "NEUTRAL",
                "minor_categories": [{"code": "calmness", "confidence": "0.4"}],
            },
        }
    )
    assert request.request_id == "42"
    assert request.major == ["NEUTRAL"]
    assert request.detected == ["CALMNESS"]


@pytest.mark.parametrize(
    "event, code",
    [
        ({"gemini_result": {}}, "MISSING_REQUEST_ID"),
        ({"request_id": "1"}, "INVALID_GEMINI_RESULT"),
    ],
)
def test_top_level_validation_errors(event, code):
    with pytest.raises(schemas.RequestError) as excinfo:
        schemas.parse_request(event)
    assert excinfo.value.code == code


@pytest.mark.parametrize(
    "gemini_override, code",
    [
        ({"transcript": ""}, "MISSING_TRANSCRIPT"),
        ({"transcript": None}, "MISSING_TRANSCRIPT"),
        ({"major": []}, "INVALID_MAJOR"),
        ({"major": ["NOT_A_MAJOR"]}, "INVALID_MAJOR"),
        ({"detected": []}, "INVALID_DETECTED"),
        ({"detected": [{"code": "NOT_AN_EMOTION", "confidence": 0.9}]}, "INVALID_DETECTED"),
    ],
)
def test_gemini_result_validation_errors(gemini_override, code):
    with pytest.raises(schemas.RequestError) as excinfo:
        schemas.parse_request(_event(gemini_result=gemini_override))
    assert excinfo.value.code == code


def test_unknown_detected_codes_are_dropped_not_fatal():
    request = schemas.parse_request(
        _event(
            gemini_result={
                "detected": [
                    {"code": "JOY", "confidence": 0.7},
                    {"code": "MADE_UP", "confidence": 0.9},
                ]
            }
        )
    )
    assert request.detected == ["JOY"]


def test_normalize_filters_dedupes_and_clamps():
    result = schemas.normalize_minor_categories(
        [
            {"code": "JOY", "confidence": 0.4},
            {"code": "joy", "confidence": 1.9},        # duplicate, out of range
            {"code": "FAKE_CODE", "confidence": 0.9},  # not in the taxonomy
            {"code": "SATISFACTION", "confidence": -3},
        ]
    )
    assert result == [
        {"code": "JOY", "confidence": 1.0},
        {"code": "SATISFACTION", "confidence": 0.0},
    ]


def test_normalize_inherits_input_confidence_when_model_omits_it():
    result = schemas.normalize_minor_categories(
        [{"code": "JOY"}, {"code": "PRIDE"}], {"JOY": 0.82}
    )
    assert result == [
        {"code": "JOY", "confidence": 0.82},
        {"code": "PRIDE", "confidence": 0.5},
    ]


def test_success_response_shape_matches_contract():
    request = schemas.parse_request(_event())
    response = schemas.success_response(
        request,
        [{"code": "JOY", "confidence": 0.75}],
        mode="hybrid",
        model="gemini-2.5-flash",
    )
    assert response == {
        "request_id": "231436",
        "minor_categories": [{"code": "JOY", "confidence": 0.75}],
        "meta": {
            "engine": "lightrag_native_query",
            "mode": "hybrid",
            "model": "gemini-2.5-flash",
        },
    }
