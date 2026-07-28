"""LLM re-judgement: request format, response contract, and parsing.

The graph proposes candidates; this module is where an LLM decides which of
them the utterance actually supports. Two variants share one prompt skeleton so
the ablation isolates a single variable:

  with_evidence=True   candidates carry their graph / situational grounding
                       (arm A5 — the full GraphRAG reviewer)
  with_evidence=False  the same instructions, but the candidate list is the
                       bare taxonomy with no grounding (arm A4 — the control
                       that answers "isn't this just the LLM being clever?")

Response contract (json_mode is requested from the provider):

    {"judgements": [
        {"code": "EMPATHIC_PAIN", "confidence": 0.8, "evidence": "..."},
        ...
    ]}

Everything the model can get wrong is handled explicitly and counted, because
a silent parse failure looks exactly like a low-recall result:

    fenced output            -> fences stripped before parsing
    unparseable JSON         -> one retry, then confidence 0.0 + parse_fail
    code not among candidates-> dropped, counted as hallucinated
    code outside the 48      -> dropped, counted as hallucinated
    candidate never mentioned-> confidence 0.0
    confidence out of [0,1]  -> clamped
    duplicate codes          -> max kept

The accept decision is NOT made here: the caller applies its own tau so the
threshold can be swept without re-running the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import providers
import schema

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class CandidateView:
    """One candidate as the prompt should present it."""
    code: str
    graph_line: str = ""     # e.g. "SADNESS와 19회 동반, P=0.42, lift 1.6"
    sit_line: str = ""       # e.g. "유사 발화 \"...\"(유사도 0.71)에 라벨됨"


@dataclass
class Judgement:
    code: str
    confidence: float
    evidence: str = ""


@dataclass
class JudgeOutcome:
    confidences: dict[str, float] = field(default_factory=dict)
    judgements: list[Judgement] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)
    parse_fail: bool = False
    retries: int = 0
    raw: str = ""


_RULES = """[규칙]
- 그래프/상황 근거는 "일반적 경향"일 뿐 이 발화의 증거가 아니다. 통계만으로 채택하지 마라.
- 발화에 단서가 없으면 confidence를 0.3 미만으로 줘라.
- 1차 결과와 대분류가 달라도 동등하게 평가하라. 복합감정은 정상이다.
- evidence는 발화에서 인용하거나 발화 내용을 근거로 써라. 못 쓰면 confidence를 낮춰라.
- 후보 목록에 없는 감정은 절대 출력하지 마라."""

_OUTPUT = """[출력] 아래 형태의 JSON 객체만. 마크다운 코드펜스와 설명 문장 금지.
{"judgements":[{"code":"<후보코드>","confidence":<0.0~1.0>,"evidence":"<발화 근거 한 문장>"}]}
후보 %d개 전부에 대해 한 항목씩. 채택 여부는 호출측이 정한다."""


def build_prompt(*, transcript: str, summary: str, prosody: str,
                 detected: set[str], candidates: list[CandidateView],
                 with_evidence: bool) -> str:
    det = " · ".join(
        f"{c}({schema.SUB_KR.get(c, c)})" for c in sorted(detected)
    ) or "없음"

    lines = []
    for i, c in enumerate(candidates, 1):
        kr = schema.SUB_KR.get(c.code, c.code)
        if not with_evidence:
            lines.append(f"{i}. {c.code} ({kr})")
            continue
        lines.append(f"{i}. {c.code} ({kr})")
        lines.append(f"   그래프: {c.graph_line or '근거 없음'}")
        lines.append(f"   상황: {c.sit_line or '근거 없음'}")

    header = (
        "[역할] 너는 감정 검수관이다. 1차 분석기가 놓쳤을 수 있는 감정을 "
        "아래 후보 목록 중에서만 골라낸다.\n\n"
        f"[발화]\n전사: {transcript}\n"
        f"요약: {summary}\n"
        f"운율: {prosody}\n\n"
        f"[1차 분석 결과 — 확정, 판정 대상 아님]\n{det}\n\n"
        f"[후보 {len(candidates)}개 — 각각 판정하라]\n"
    )
    return f"{header}{chr(10).join(lines)}\n\n{_RULES}\n\n{_OUTPUT % len(candidates)}"


def parse_response(text: str, allowed: set[str]) -> tuple[list[Judgement], list[str], bool]:
    """Return (judgements, hallucinated_codes, parse_failed)."""
    stripped = _FENCE.sub("", (text or "").strip())
    payload = None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        # Some models wrap the object in prose; take the outermost {...}.
        m = _OBJECT.search(stripped)
        if m:
            try:
                payload = json.loads(m.group(0))
            except json.JSONDecodeError:
                payload = None
    if payload is None:
        return [], [], True

    items = payload.get("judgements") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return [], [], True

    best: dict[str, Judgement] = {}
    hallucinated: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip().upper()
        if code not in allowed or not schema.is_known_sub(code):
            if code:
                hallucinated.append(code)
            continue
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(1.0, max(0.0, conf))
        evidence = str(item.get("evidence", "")).strip()
        prev = best.get(code)
        if prev is None or conf > prev.confidence:
            best[code] = Judgement(code=code, confidence=conf, evidence=evidence)
    return list(best.values()), hallucinated, False


async def judge(*, transcript: str, summary: str, prosody: str,
                detected: set[str], candidates: list[CandidateView],
                with_evidence: bool, max_retries: int = 1) -> JudgeOutcome:
    """One LLM call per utterance (all candidates judged together).

    Judging candidates jointly rather than one-by-one gives the model a
    comparative context, which suppresses blanket acceptance and costs one
    request instead of len(candidates).
    """
    if not candidates:
        return JudgeOutcome()

    allowed = {c.code for c in candidates}
    prompt = build_prompt(
        transcript=transcript, summary=summary, prosody=prosody,
        detected=detected, candidates=candidates, with_evidence=with_evidence,
    )

    outcome = JudgeOutcome()
    text = ""
    for attempt in range(max_retries + 1):
        ask = prompt if attempt == 0 else (
            prompt + "\n\n주의: 이전 응답이 JSON 파싱에 실패했다. "
            "설명 없이 JSON 객체 하나만 출력하라."
        )
        text = await providers.chat(ask, json_mode=True)
        judgements, hallucinated, failed = parse_response(text, allowed)
        outcome.retries = attempt
        if not failed:
            outcome.judgements = judgements
            outcome.hallucinated = hallucinated
            outcome.raw = text
            break
    else:
        outcome.parse_fail = True
        outcome.raw = text

    # Candidates the model ignored are treated as rejected, not as missing data.
    outcome.confidences = {c.code: 0.0 for c in candidates}
    for j in outcome.judgements:
        outcome.confidences[j.code] = j.confidence
    return outcome
