# emotion_layer1 — 1층 감정 개념 그래프

음성 감정분석 보정용 GraphRAG의 **1층(오프라인 배치)**. 노드는 분류표에서
정의하고, 엣지는 gold 라벨에서 **세서** 만든다. LightRAG core는 건드리지 않고
공개 API `ainsert_custom_kg` / `BaseGraphStorage`만 쓴다.

## 설계 원칙
- **노드 = 분류표(taxonomy.json)에서 정의.** 데이터에서 추출 안 함.
- **엣지 = 라벨 집계 + 임계값.** 약한 선은 버림(`min_count`).
- **train만 그래프에 주입.** holdout은 검증 전용(데이터 누수 금지).
- 자동추출에 라벨 안 맡김 — `custom_kg`로 직접 주입.

## 모듈
| 파일 | 역할 |
|------|------|
| `taxonomy.json` | 분류표(노드 정의). **교체지점** — 실제 분류표로 갈아끼움 |
| `taxonomy.py` | 분류표 로드 + 노드 id(타입 프리픽스) + 라벨 해석 |
| `schema.py` | 라벨 레코드 컨트랙트 + JSONL 로더 + 검증 + split |
| `nodes.py` | 분류표 → custom_kg entities + 소분류→대분류 BELONGS_TO |
| `aggregate.py` | 선1(감정 공존) + cue→감정 집계, count+P(B\|A), 임계값 |
| `build_kg.py` | nodes + edges + source chunk → custom_kg dict |
| `inject.py` | LightRAG 파일백엔드 + mock 임베딩(키 없이 오프라인) 주입 |
| `infer.py` | 추론 루프 스텁: 후보 조회 + clue_check 스텁 + suggest_missing |
| `smoke.py` | end-to-end 동작 확인 |

## 라벨 컨트랙트 (gold, JSONL 한 줄=한 레코드)
```json
{
  "record_id": "r001",
  "speaker_id": null,
  "transcript": "...",
  "split": "train",
  "gold_labels": {
    "emotions": [{"major": "슬픔", "minor": "사랑", "confidence": 0.9}],
    "nonverbal_raw": "라벨러 자유서술",
    "nonverbal_tags": ["voice_tremor"],
    "context": ""
  },
  "model_outputs": {"gemini": {}}
}
```
- 감정명 = 한글, 분류표 기준(Gemini 카테고리 아님). 멀티라벨.
- `nonverbal_tags` = 통제 어휘(cue 노드 id). `nonverbal_raw`는 집계 안 함.
- `context` = 자유 문자열, **선2(상황→감정)는 후순위**.
- `model_outputs.gemini`는 1층 빌드에서 **무시**(예측 분리 저장).

## 실행
```bash
python -m emotion_layer1.smoke
```

## 현재 상태 / 미구현(다음 단계)
- 화자 split: 지금은 레코드 단위(speaker_id=null). 화자 정보 확보 시 화자
  단위로 전환(`schema.split_records` 교체).
- 선2(상황→감정): context 통제 어휘 정해지면 추가.
- exemplar chunks, holdout 보정 정확도 측정, 2층 incremental insert: 미구현.
- `clue_check`는 스텁(항상 True) — LLM/키워드 단서확인으로 교체 예정.
- 분류표가 임시로 emotion_engine rules.py(한글)에서 시드됨. **실제 분류표로
  `taxonomy.json` 교체** 필요(섭섭류 복합감정은 지금은 선만, 노드 승격은 나중).
