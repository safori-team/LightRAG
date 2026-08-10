"""Official emotion taxonomy: 6 major categories / 48 minor categories.

Pure module — no LightRAG or provider imports — so the taxonomy filter can be
unit tested without touching the graph artifact or any network call.

This is the operational copy of the codebook that
``examples/emotion_graphrag/schema.py`` uses for the offline experiments. The
codes must stay byte-identical to that file: the shipped graph artifact stores
entity names from this exact list, and the native query prompt restricts the
model to it.
"""

from __future__ import annotations

TAXONOMY: dict[str, dict] = {
    "HAPPY": {
        "kr": "기쁨",
        "subs": {
            "JOY": "기쁨", "ECSTASY": "황홀", "CONTENTMENT": "흐뭇함",
            "SATISFACTION": "만족", "AMUSEMENT": "즐거움", "EXCITEMENT": "신남",
            "PRIDE": "자부심", "TRIUMPH": "승리감", "RELIEF": "안도",
            "ADMIRATION": "감탄", "ADORATION": "흠모", "LOVE": "사랑",
            "ROMANCE": "설렘", "ENTRANCEMENT": "매혹",
            "AESTHETIC_APPRECIATION": "심미적 감상", "DETERMINATION": "결의",
        },
    },
    "SAD": {
        "kr": "슬픔",
        "subs": {
            "SADNESS": "슬픔", "DISTRESS": "괴로움", "DISAPPOINTMENT": "실망",
            "GUILT": "죄책감", "SHAME": "수치심", "EMBARRASSMENT": "민망함",
            "EMPATHIC_PAIN": "공감적 고통", "SYMPATHY": "연민", "LONELINESS": "외로움",
        },
    },
    "ANGRY": {
        "kr": "분노",
        "subs": {
            "ANGER": "분노", "CONTEMPT": "경멸", "DISGUST": "혐오",
            "FRUSTRATION": "좌절", "ENVY": "시기", "CRAVING": "갈망",
        },
    },
    "ANXIETY": {
        "kr": "불안",
        "subs": {"FEAR": "두려움", "ANXIETY_GENERAL": "불안", "HORROR": "공포"},
    },
    "SURPRISE": {
        "kr": "놀람",
        "subs": {
            "SURPRISE_POSITIVE": "긍정적 놀람", "SURPRISE_NEGATIVE": "부정적 놀람",
            "AWE": "경외", "AWKWARDNESS": "어색함",
        },
    },
    "NEUTRAL": {
        "kr": "중립",
        "subs": {
            "CALMNESS": "평온", "CONTEMPLATION": "사색", "CONCENTRATION": "집중",
            "INTEREST": "흥미", "REALIZATION": "깨달음", "BOREDOM": "지루함",
            "TIREDNESS": "피로", "CONFUSION": "혼란", "DOUBT": "의심",
            "NOSTALGIA": "향수",
        },
    },
}

ALL_MAJORS: list[str] = list(TAXONOMY.keys())
SUB2MAJOR: dict[str, str] = {
    sub: major for major, data in TAXONOMY.items() for sub in data["subs"]
}
ALL_SUBS: list[str] = list(SUB2MAJOR.keys())
SUB_KR: dict[str, str] = {
    sub: kr for data in TAXONOMY.values() for sub, kr in data["subs"].items()
}


def is_known_major(code: str) -> bool:
    return code in TAXONOMY


def is_known_sub(code: str) -> bool:
    return code in SUB2MAJOR


def taxonomy_text() -> str:
    """Render the taxonomy for the native query prompt.

    Identical to the rendering used by the 91-record native query experiment;
    changing it changes the prompt and invalidates that measurement.
    """
    return "\n".join(
        f"- {major}: " + ", ".join(
            f"{code}({name})" for code, name in data["subs"].items()
        )
        for major, data in TAXONOMY.items()
    )
