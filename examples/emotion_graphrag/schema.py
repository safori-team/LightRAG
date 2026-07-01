"""Fixed emotion taxonomy and custom_kg dict builders.

Nodes are fixed by the official codebook (6 major / 48 minor) and are NEVER
derived from data. This module is pure (no LightRAG import) so it can be unit
tested and mirrors the design doc's "노드=분류표 고정" principle.

See DESIGN.md sections 2 and 7.
"""

from __future__ import annotations

# --- Node / edge type tags (encoded into entity_type / relationship keywords) --
NODE_MAJOR = "EmotionMajor"
NODE_MINOR = "EmotionMinor"

EDGE_BELONGS_TO = "belongs_to"   # minor -> major (taxonomy, weight 1.0)
EDGE_CO_OCCURS = "co_occurs"     # minor <-> minor (data aggregate, wilson weight)

# source_id partitions (single-workspace isolation, see DESIGN.md section 6)
SRC_TAXONOMY = "taxonomy"
SRC_TRAIN_AGG = "train_agg"


def _utt_source_id(clip_id: str) -> str:
    return f"utt_{clip_id}"


# --- Official taxonomy: major -> {kr, subs: {code: kr}} -----------------------
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

# Derived lookups
ALL_MAJORS: list[str] = list(TAXONOMY.keys())
SUB2MAJOR: dict[str, str] = {
    sub: major for major, d in TAXONOMY.items() for sub in d["subs"]
}
ALL_SUBS: list[str] = list(SUB2MAJOR.keys())
SUB_KR: dict[str, str] = {
    sub: kr for d in TAXONOMY.values() for sub, kr in d["subs"].items()
}


def is_known_sub(code: str) -> bool:
    return code in SUB2MAJOR


# --- custom_kg dict builders --------------------------------------------------
def taxonomy_entities() -> list[dict]:
    """Return the fixed 54 node dicts (6 major + 48 minor)."""
    entities: list[dict] = []
    for major, d in TAXONOMY.items():
        entities.append({
            "entity_name": major,
            "entity_type": NODE_MAJOR,
            "description": f"{d['kr']} (대분류)",
            "source_id": SRC_TAXONOMY,
        })
        for sub, kr in d["subs"].items():
            entities.append({
                "entity_name": sub,
                "entity_type": NODE_MINOR,
                "description": f"{kr} (대분류: {d['kr']} {major})",
                "source_id": SRC_TAXONOMY,
            })
    return entities


def belongs_to_edges() -> list[dict]:
    """Return the 48 minor->major hierarchy edges (deterministic, weight 1.0)."""
    edges: list[dict] = []
    for sub, major in SUB2MAJOR.items():
        edges.append({
            "src_id": sub,
            "tgt_id": major,
            "keywords": EDGE_BELONGS_TO,
            "weight": 1.0,
            "description": f"{SUB_KR[sub]}은(는) {TAXONOMY[major]['kr']}의 하위감정",
            "source_id": SRC_TAXONOMY,
        })
    return edges


def cooccur_edge(src: str, tgt: str, weight: float, evidence: str) -> dict:
    """Build one co_occurs edge dict (weight + evidence string from rules.py)."""
    return {
        "src_id": src,
        "tgt_id": tgt,
        "keywords": EDGE_CO_OCCURS,
        "weight": round(weight, 4),
        "description": evidence,
        "source_id": SRC_TRAIN_AGG,
    }


def utterance_chunk(clip_id: str, transcript: str, order: int = 0) -> dict:
    """Build one exemplar/utterance chunk (situational-similarity index)."""
    return {
        "content": transcript,
        "source_id": _utt_source_id(clip_id),
        "chunk_order_index": order,
    }
