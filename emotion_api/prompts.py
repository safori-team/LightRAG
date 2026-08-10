"""Native LightRAG query text and emotion-selection prompt.

Both strings are ported verbatim from the 91-record native query experiment
(``examples/emotion_graphrag/local_experiments/native_query_455.py``). Any edit
here changes the measured behaviour, so ``tests/emotion_api/test_query_config.py``
pins them.
"""

from __future__ import annotations

from emotion_api import taxonomy
from emotion_api.schemas import EmotionRequest

USER_PROMPT = f"""감정 선택 전용 규칙:
1. 입력의 Gemini 감정을 무조건 유지하지 말고 검색 근거와 실제 발화를 함께 검토한다.
2. 검색된 감정을 무조건 추가하지 않는다.
3. 아래 공식 taxonomy 코드만 출력한다.
4. 근거가 있는 감정을 모두 선택하되 개수를 억지로 3~5개에 맞추지 않는다.
5. JSON 객체 외에는 아무것도 출력하지 않는다.

공식 taxonomy:
{taxonomy.taxonomy_text()}

출력 형식:
{{"minor_categories":[{{"code":"CALMNESS","confidence":0.7}}]}}
"""


def build_query(request: EmotionRequest) -> str:
    """Compose the retrieval query from the stored Gemini result.

    The opening line states that the payload is an existing analysis, which is
    what keeps the model from re-running emotion detection on the transcript.
    """
    existing = ", ".join(request.detected) or "없음"
    major = ", ".join(request.major) or "없음"
    return (
        "아래는 이미 저장되어 있는 Gemini 음성 분석 결과다. 새 음성을 분석하는 요청이 아니다.\n"
        f"전사: {request.transcript}\n"
        f"요약: {request.summary}\n"
        f"운율: {request.prosody}\n"
        f"Gemini 대분류: {major}\n"
        f"Gemini 소분류: {existing}\n\n"
        "지식 그래프와 검색된 사례를 함께 사용하여 이 발화에서 실제로 지지되는 "
        "최종 소분류 감정 집합을 선택하라."
    )
