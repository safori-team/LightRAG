# 감정 GraphRAG 재설계안 — 검수관 1층 + 하이브리드 후보생성

> LightRAG + Ollama 기반. 그래프는 감정을 **판정하지 않고**, Gemini 1차 결과에서 **누락 감정을 찾아주는 검수관**이자 CBT 개인화 근거 저장소.

---

## 0. 한 줄 요지

음성 → **Gemini 1차 감정 분류(신호)** → **GraphRAG 보정(검수관)** → CBT 상담 개인화.
판정·설명은 LLM, 그래프는 **후보·근거·가중치**만 공급한다.

---

## 1. 데이터 실측 (근거)

| 항목 | 값 | 시사점 |
|---|---|---|
| 총 레코드 | 100 | 소량 |
| 화자 수 | 57 (대부분 1~2건) | 2층 개인화·화자 holdout 불안정 |
| 대분류 | 6종 (ANGRY 31 / NEUTRAL 21 / ANXIETY 18 / SAD 17 / HAPPY 14 / SURPRISE 11) | 노드로 적합 |
| 소분류(다수결) | 33종, 롱테일 | — |
| 어노테이터 sub 리딩 | **497개** | co_occurs 집계원 |
| `original_major_code` | 50% 공백 + 오타 | triggered_by(선2) 지금 부적합 |
| 비언어 cue | 없음 (억양기호 HL/LH/M만) | cue_indicates 불가 |

**분류표 정합성 점검**: 데이터 소분류가 전부 공식 분류표 48개 안에 있음(오타·미등록 0). 분류표 중 데이터에 없는 cold 노드는 ECSTASY 1개뿐. → "노드=분류표 고정" 정제 없이 적용 가능.

---

## 2. 분류 체계 (고정 노드, 대6 / 소48)

| 대분류 | 소분류 |
|---|---|
| **HAPPY 기쁨** (16) | JOY, ECSTASY*, CONTENTMENT, SATISFACTION, AMUSEMENT, EXCITEMENT, PRIDE, TRIUMPH, RELIEF, ADMIRATION, ADORATION, LOVE, ROMANCE, ENTRANCEMENT, AESTHETIC_APPRECIATION, DETERMINATION |
| **SAD 슬픔** (9) | SADNESS, DISTRESS, DISAPPOINTMENT, GUILT, SHAME, EMBARRASSMENT, EMPATHIC_PAIN, SYMPATHY, LONELINESS |
| **ANGRY 분노** (6) | ANGER, CONTEMPT, DISGUST, FRUSTRATION, ENVY, CRAVING |
| **ANXIETY 불안** (3) | FEAR, ANXIETY_GENERAL, HORROR |
| **SURPRISE 놀람** (4) | SURPRISE_POSITIVE, SURPRISE_NEGATIVE, AWE, AWKWARDNESS |
| **NEUTRAL 중립** (10) | CALMNESS, CONTEMPLATION, CONCENTRATION, INTEREST, REALIZATION, BOREDOM, TIREDNESS, CONFUSION, DOUBT, NOSTALGIA |

`*` ECSTASY = 데이터 미출현 cold 노드(생성만, 엣지 없음). 한글 이름은 노드 `description`에 넣어 임베딩·CBT 설명 근거로 사용.

---

## 3. 전체 아키텍처 (하이브리드)

### 빌드 (오프라인) — 두 인덱스를 만든다

| 산출물 | 내용 | 역할 |
|---|---|---|
| 노드 54 | 분류표 고정 | 골격 |
| belongs_to 48 | 소→대 (분류표, w=1.0) | 계층 |
| **co_occurs ~298** | 라벨 집계, w=wilson 하한 | **전역 prior (소스1)** |
| **발화 chunk 100** | transcript 임베딩 + 라벨 연결 | **상황 유사도 인덱스 (소스2)** |

빌드는 `ainsert_custom_kg`로 직접 주입 → LightRAG 기본 LLM 추출을 우회해 **라벨 희석 방지**. 이 단계 Ollama는 **임베딩만** 사용.

### 추론 (실시간) — 두 소스 융합

```
발화 → ┌ 소스1: get_node_edges(Gemini감정) → co_occurs 이웃  (전역 prior, 결정론)
        └ 소스2: 발화 임베딩 → 유사 발화 top-K → 이웃 라벨 집계 → sit_score  (맥락)
   → rules.fuse(graph_w·boost, sit_score) → 후보 → judge(단서 확인) → 누락 추가
   → CBT 근거 문장(LLM)
```

- 소스1 = "일반적으로 이 감정과 붙는 것"(안정적, 소량 데이터에 강함).
- 소스2 = "이 상황과 비슷한 발화에서 실제로 붙은 것"(맥락 반영). → 전역 통계의 "상황 장님" 약점 보완.
- LightRAG **mix 모드(그래프+벡터)** 철학과 일치.

---

## 4. co_occurs 판정 근거 — 4단계 깔때기

각 단계가 하나의 반론을 막는 관문. 모두 통과해야 엣지가 된다.

| 근거 | 답하는 반론 | 기준 | 데이터 예시 |
|---|---|---|---|
| **support** | 우연 1회 아닌가 | count ≥ 2 | count=1 쌍 270개 컷 |
| **lift** | B가 흔해서 아닌가 | P(B\|A)/P(B) ≥ 1.3 | FRUSTRATION→EMBARRASSMENT(c=19)도 lift 0.75로 **기각** |
| **Wilson 하한** | 표본 적어 뻥튀기 아닌가 | weight = wilson_lb(P(B\|A)) | LONELINESS→SADNESS 0.80 → 0.51 수축 |
| **방향성** | 방향 반대 아닌가 | P(B\|A) ≠ P(A\|B), 방향별 엣지 | 외로움→슬픔 0.80 vs 슬픔→외로움 0.10 |

**lift의 위력**: FRUSTRATION은 리딩의 28%에 등장(초다빈도). count만 보면 FRUSTRATION과 붙는 쌍이 다 강해 보이지만 lift가 우연분을 걷어내 기각한다. = "우연히 붙는 쌍은 버린다"의 실제 구현.

**감사 가능성**: 채택된 엣지마다 판정 수치를 `description`에 저장.
```
LONELINESS → SADNESS  weight=0.51
description: "외로움→슬픔 동반. count=4, P=0.80, lift=10.2, pmi=2.32"
```
규칙이 결정론(rules.py)이라 누가 돌려도 같은 근거·결론.

**추론 시 2차 근거**: co_occurs 엣지는 통계적 후보일 뿐. 실제 추가는 exemplar 유사도 + LLM judge로 발화에서 재확인 → 통계만으로 감정 추가 안 함.

---

## 5. 결정론 계층 분리 (R1)

임계값·weight·boost·융합계수는 전부 `rules.py` **코드 상수**. 그래프엔 계산된 숫자만 나감. RAG는 재해석하지 않는다.

```python
# rules.py 핵심 상수 (전부 holdout 튜닝 대상, 확정값 아님)
SUPPORT_FLOOR = 2       # 단발 노이즈 제거
TAU_LIFT      = 1.3     # 그래프 후보 컷
WILSON_Z      = 1.28    # 80% 하한(소량이라 관대)
LAMBDA_SIT    = 0.5     # 상황 신호 가중
SIGMA_SIT     = 0.25    # 상황 후보 컷
SIT_TOPK      = 8       # 유사 발화 이웃 수
```

> **소량 데이터 원칙**: 절대 임계값을 지금 확정하지 않는다. 빌드는 support 바닥값만, 실제 컷은 rules.py 런타임(재빌드 불필요) + holdout 회수율/오탐율 곡선으로 선택. 통계가 약한 만큼 **judge 게이트 + 상황 유사도**가 오탐을 막는다.

---

## 6. 저장 구조

단일 workspace + `source_id` 파티션:

| 대상 | source_id | 비고 |
|---|---|---|
| 1층 노드 | `taxonomy` | 읽기 전용 |
| 1층 엣지(co_occurs) | `train_agg` | |
| 발화 chunk | `utt_{clip_id}` | 라벨은 `labels_map.json` 사이드 룩업 |
| 2층 인스턴스(보류) | `spk_{speaker_id}` | 개인화 |

후보 조회는 `get_node_edges` + `get_edges_batch`(weight/keywords)로 결정론 수행. LLM 미개입.

---

## 7. 산출물(파일) 목록 — `examples/emotion_graphrag/`

| 파일 | 역할 |
|---|---|
| `schema.py` | 분류표 54노드 상수(TAXONOMY), node/edge/chunk dict 빌더 |
| `rules.py` | wilson/lift/support + sit_score/fuse + 근거 문자열 |
| `aggregate.py` | CSV → co_occurs 엣지 집계 |
| `build_layer1.py` | custom_kg 조립(노드+belongs_to+co_occurs+발화chunk) → `ainsert_custom_kg` |
| `infer.py` | 소스1+소스2 융합 → judge → 누락 추가 → CBT 근거 |
| `labels_map.json` | `utt_{clip_id} → [labels]` 사이드 매핑 |

---

## 8. 평가

- **신호 정확도**(Gemini 1차) vs **해석 정확도**(보정 후) 분리 측정.
- **검수관 메트릭**: 레코드 단위 20% holdout(대분류 층화)에서 누락 회수율 / 오탐율.
- **하이브리드 효과**: 그래프-only vs 그래프+상황 유사도 비교로 `LAMBDA_SIT` 결정.
- holdout은 그래프·검색 인덱스에 **절대 미투입**(누수 금지).

---

## 9. Ollama 운영

| 항목 | 값/주의 |
|---|---|
| 임베딩 | `bge-m3` (다국어, 1024d). 모델 변경 시 rag_storage 초기화 필수 |
| LLM | judge/CBT용 한국어 모델(exaone/qwen2.5/gemma2), temperature 0.2~0.4 |
| 컨텍스트 | `OLLAMA_LLM_NUM_CTX` > MAX_TOTAL_TOKENS+2000 |
| 부하 | 빌드는 임베딩만(LLM 추출 안 함) → 빠르고 환각 없음. `EMBEDDING_FUNC_MAX_ASYNC` 상향 |

---

## 10. 진행 순서 & 보류 항목

**진행(점진)**
1. 최소 빌드: 노드 + belongs_to + co_occurs + 발화 chunk 주입.
2. holdout으로 그래프-only 회수율/오탐율 측정.
3. 상황 유사도(소스2) 융합 + 재평가 → `LAMBDA_SIT` 확정.

**보류(데이터 사정)**

| 항목 | 사유 | 재개 조건 |
|---|---|---|
| triggered_by(선2) | original_major_code 50% 공백+오타 | 상황 필드 정제 |
| cue_indicates | cue 데이터 없음 | 비언어 cue 라벨 확보 |
| 2층 개인화 | 화자 대부분 1~2건(cold start) | 반복 화자/세션 누적 |

---

## 불변 원칙

노드=분류표 고정 · 엣지=train 집계(근거 저장) · holdout 미투입 · 1층은 custom_kg 직접 주입(기본 추출로 라벨 희석 금지) · 정밀 가중치·규칙은 코드 결정론, 그래프는 개인화·후보 공급.
