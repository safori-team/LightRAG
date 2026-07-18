# 감정 GraphRAG 설계 — 검수관(소분류·복합감정) + 하이브리드 후보 + LLM 재판정

> LightRAG + Ollama 기반. 그래프는 감정을 **판정하지 않고**, Gemini 1차 결과에서 **누락된 소분류·복합감정을 찾아주는 검수관**이자 CBT 개인화 근거 저장소.
> (본 문서는 W6~W7 구현·측정을 반영한 현행판. 이전 초안 대비 judge v2·평가 방법론·무향 그래프·객관 검증·A/B/C 어블레이션이 추가됨.)

---

## 0. 요지 & 스코프

**파이프라인**: 음성 → **Gemini 대분류(신호)** → **GraphRAG 소분류·복합감정 보완(검수관)** → CBT 개인화.
판정·설명은 LLM, 그래프는 **후보·근거·가중치**만 공급.

**스코프 (공모전: GraphRAG 사용은 필수 조건)**
- **대분류 = Gemini** 담당(음성 기반, 이미 우수 → 그래프로 개선 안 함).
- **GraphRAG = 소분류 + 복합감정(cross-major)** 에 집중. 그다음 개인화(2층).
- 철학: "언어는 그 사람의 세계를 담는다 — 같은 음성도 사람마다 다르게 필터된다." 대분류는 보편, 그래프는 개인화된 세계의 결과.
- **현재 단계: cross-major 복합감정만 우선** (2층 개인화는 데이터 확보 후).

근거: 객관 검증(§10)에서 그래프의 고유 우위 영역이 **대분류를 넘는 복합감정**임을 확인 → 거기 특화.

---

## 1. 데이터 실측

| 항목 | 값 | 시사점 |
|---|---|---|
| 총 레코드 | 100 (train 69 / holdout 31, 화자 단위 분리) | 소량 → 방향성 지표 |
| 화자 수 | 57 (대부분 1~2건) | 2층 개인화·화자 holdout 불안정 |
| 대분류 | 6종 (ANGRY 31 / NEUTRAL 21 / ANXIETY 18 / SAD 17 / HAPPY 14 / SURPRISE 11) | Gemini 담당 |
| 소분류(다수결) | 33종, 롱테일 | 검수관 타깃 |
| 어노테이터 sub 리딩 | 497개 | **co_occurs 집계원**(다수결 아님) |
| `original_major_code` | 50% 공백 + 오타 | triggered_by(선2) 부적합 |
| 비언어 cue | 없음(억양기호만) | cue_indicates 불가 |

- **정답(gold) = `human_majority_sub_codes`**(다수결). 단, 평가에선 소수지지도 반영(§9).
- 분류표 정합성: 데이터 소분류 전부 공식 분류표 48개 내(오타·미등록 0). cold 노드 ECSTASY 1개뿐.

---

## 2. 분류 체계 (고정 노드, 대6 / 소48)

| 대분류 | 소분류 |
|---|---|
| HAPPY 기쁨 (16) | JOY, ECSTASY*, CONTENTMENT, SATISFACTION, AMUSEMENT, EXCITEMENT, PRIDE, TRIUMPH, RELIEF, ADMIRATION, ADORATION, LOVE, ROMANCE, ENTRANCEMENT, AESTHETIC_APPRECIATION, DETERMINATION |
| SAD 슬픔 (9) | SADNESS, DISTRESS, DISAPPOINTMENT, GUILT, SHAME, EMBARRASSMENT, EMPATHIC_PAIN, SYMPATHY, LONELINESS |
| ANGRY 분노 (6) | ANGER, CONTEMPT, DISGUST, FRUSTRATION, ENVY, CRAVING |
| ANXIETY 불안 (3) | FEAR, ANXIETY_GENERAL, HORROR |
| SURPRISE 놀람 (4) | SURPRISE_POSITIVE, SURPRISE_NEGATIVE, AWE, AWKWARDNESS |
| NEUTRAL 중립 (10) | CALMNESS, CONTEMPLATION, CONCENTRATION, INTEREST, REALIZATION, BOREDOM, TIREDNESS, CONFUSION, DOUBT, NOSTALGIA |

`*` ECSTASY = 데이터 미출현 cold 노드. 한글 이름은 노드 `description`에 넣어 임베딩·설명 근거로 사용.

---

## 3. 전체 아키텍처

### 빌드 (오프라인) — `build_layer1.py`
`ainsert_custom_kg`로 직접 주입(LLM 추출 우회 → 라벨 희석 방지). Ollama는 임베딩만 사용.

| 산출물 | 내용 | 실제 규모(train 69) |
|---|---|---|
| 노드 | 분류표 고정 | 54 |
| belongs_to | 소→대 (w=1.0) | 48 |
| co_occurs | 라벨 집계, 무향+대칭 weight | 117 |
| 발화 chunk | transcript 임베딩 + 라벨 연결 | 68 |

### 추론 (실시간) — `infer.py`
```
발화 + Gemini 감정(detected)
  │ ① 후보 생성(hybrid)
  │   source1 graph:  get_node_edges(detected) → co_occurs 이웃 (방향 weight)
  │   source2 situational: 발화 임베딩 → 유사 발화 top-K → 이웃 라벨 집계
  │   → rules.fuse → 상위 TOP_K 후보
  │ ② LLM 재판정 (judge v2): 근거주입 + 선택형 + 확신도(tau)
  ▼ 통과한 것만 추가 → (CBT 근거 문장)
```

---

## 4. co_occurs 판정 근거 — 4단계 깔때기 + 무향 처리

| 근거 | 반론 | 기준 | 데이터 예 |
|---|---|---|---|
| support | 우연 1회? | count ≥ 2 | count=1 쌍 대량 컷 |
| **lift** | B가 흔해서? | P(B\|A)/P(B) ≥ 1.3 | FRUSTRATION→EMBARRASSMENT(c19) lift0.75 기각 |
| **Wilson 하한** | 표본 적어 뻥튀기? | weight = wilson_lb(P(B\|A)) | LONELINESS→SADNESS 0.80→0.51 |
| 방향성 | 방향 반대? | P(B\|A) ≠ P(A\|B) | 외로움→슬픔 0.80 vs 역 0.10 |

**엣지 결정 = "임계값"이 아니라 lift(우연 보정) + Wilson(표본 신뢰).** 절대 count 임계값 하드코딩 안 함(§7).

**무향 그래프 처리(중요)**: LightRAG 기본 그래프(NetworkX)는 무향이라 `a→b`/`b→a`를 두 엣지로 넣으면 덮어써짐. 따라서:
- 그래프 엣지 = **무향 1개 + 대칭 weight**(양방향 wilson의 max). lift는 원래 대칭이라 컷 판정 무관.
- 방향별 P(B|A) weight = **사이드 테이블 `cooccur_stats.json`**(코드가 읽음). → R1 정합(정밀 가중치는 코드/사이드, 그래프는 신호).

**근거 저장(감사가능성)**: 엣지 `description`은 의미 문장만(수치는 사이드 테이블). ※ 수치를 임베딩 텍스트에 넣으면 bge-m3가 부하 시 NaN 반환(§12).

---

## 5. 후보 생성 (하이브리드 retrieval)

- **source1 graph** (결정론): `get_node_edges`+`get_edges_batch`로 detected의 co_occurs 이웃 조회, `cooccur_stats.json`의 방향 weight로 스코어.
- **source2 situational** (벡터 RAG): `chunks_vdb.query`로 유사 발화 top-K(=8) → 반환 chunk의 `full_doc_id`(=utt_{clip_id})로 `labels_map.json` 룩업 → 유사도 가중 라벨 집계(`sit_score`).
- **fuse**: 후보 = graph_weight>0 OR sit≥SIGMA_SIT; 점수 = graph + LAMBDA_SIT·sit; 상위 **CANDIDATE_TOP_K(=8)**.

→ source1="일반적 동반", source2="이 상황에서 실제 동반". source2가 대분류 넘는 복합감정 회수를 보강.

---

## 6. LLM 재판정 (judge v2) — `infer.llm_judge_batch`

v1(맨 LLM, 후보 하나씩 YES/NO, 문맥 0)의 병목을 개선:

| 개선 | 내용 |
|---|---|
| 선택형 | 후보 전체를 한 번에 제시, 비교 맥락에서 판정 → 과다 추가 억제, 호출 감소 |
| 근거주입(RAG) | 각 후보에 그래프 동시발생·상황 유사 근거 제시 → 감이 아닌 근거 기반 |
| 확신도 점수 | 이진 대신 0~1 → 임계값 **tau**로 회수율↔오탐 조절 |
| 모델 상향 | exaone3.5 7.8B → **32B** |

**응답 가드레일(부분)**: tau 미만 후보 차단 + 고정 분류표(48코드) 출력 제약(범주 밖 환각 구조적 차단). 정식 스키마 검증·거부 응답은 후속.

**정직한 천장**: 정답은 음성 기반인데 judge는 텍스트만 봄 → 어떤 텍스트 방법도 완전히는 못 넘음. 장기적으론 Gemini 음성 신호를 판정에 결합해야 함.

---

## 7. 결정론 계층 (`rules.py` 상수)

모든 정밀 가중치·임계·융합계수는 코드 상수. 그래프엔 계산된 숫자만.

```python
SUPPORT_FLOOR = 2       # 단발 노이즈 제거(유일한 하드 컷)
TAU_LIFT      = 1.3     # co_occurs 채택 컷(우연 보정)
WILSON_Z      = 1.28    # weight = wilson 하한(표본 신뢰)
SIT_TOPK      = 8       # 상황 검색 이웃 수
SIGMA_SIT     = 0.25    # 상황 후보 컷
LAMBDA_SIT    = 0.5     # 상황 신호 가중
CANDIDATE_TOP_K = 8     # 후보 폭(5→8 확대, §10)
MIN_FUSED_TO_JUDGE = 0.05
# judge tau(확신도 임계)는 평가 스윕으로 선택 (권장 0.5~0.7)
```

> 소량 데이터 원칙: 절대 임계값 확정하지 않음. 빌드는 support 바닥값만, 실제 컷은 런타임(재빌드 불필요) + holdout 튜닝.

---

## 8. 저장 구조

단일 workspace + `source_id` 파티션. 후보 조회는 결정론(get_node_edges).

| 대상 | source_id | 비고 |
|---|---|---|
| 노드 | taxonomy | 읽기 전용 |
| co_occurs 엣지 | train_agg | 무향+대칭 weight |
| 발화 chunk | utt_{clip_id} | 라벨은 `labels_map.json` |
| 방향 통계 | (사이드) | `cooccur_stats.json` |
| 2층 인스턴스(보류) | spk_{speaker} | 개인화 |

---

## 9. 평가 방법론

- **신호 vs 해석 분리**: 후보 회수율(LLM 무관) / 최종 회수율(judge 후).
- **2단계 정답 (핵심)**:
  - **target** = 다수결 소분류 (반드시 회수)
  - **acceptable** = ≥2 어노테이터가 준 소분류 (추가해도 오탐 아님) — "감정은 주관적·복합" 전제와 정합
  - **오탐** = 아무도 (≥2) 안 준 것만. `build_accept_map`으로 CSV에서 계산.
- **cross-major 분리**: 같은 대분류 회수는 baseline도 가능 → **대분류 넘는 복합감정 회수를 별도 측정**(그래프 고유 가치).
- **Gemini 시뮬**: `detected = gold_sub[0]`(첫 다수결 소분류), `missing = 나머지`. ⚠ 순서 임의·음성 미반영 한계.
- holdout은 그래프·검색 인덱스에 절대 미투입(누수 금지).

---

## 10. 측정 결과 (holdout 8케이스 / 누락 18 = same 11 + cross 7; 소량→방향성)

**그래프 가치 객관 검증** (LLM 무관, `verify_value.py`)

| 방식 | 그래프? | 회수율@5 |
|---|---|---|
| B1 global-freq | ❌ | 0.444 |
| B2 same-major(계층만) | ❌ | 0.556 |
| B3 co_occur-1hop(그래프) | ✅ | 0.444 |

→ 전체 회수율에선 그래프가 baseline 못 이김. **단 cross-major(계층 구조상 0%)에서 그래프만 회수 가능** → 프레임을 cross-major로 잡아야 그래프가 이김.

**후보 회수율** (검색 폭 확대)

| TOP_K | overall | cross-major |
|---|---|---|
| 5 | 0.444 | 0.429 |
| 8 | 0.611 | 0.429 |

**LLM 재판정** (32B, hybrid, target/acceptable 잣대)

| 설정 | target 회수율 | 오탐[엄격] | 오탐[공정≥2] |
|---|---|---|---|
| v1 이진 7.8B | 0.167 | — | 0.750 |
| v1 이진 32B | 0.278 | 0.583 | 0.333 |
| **v2 tau0.7** | 0.389 | — | **0.250** |
| **v2 tau0.5** | **0.500** | — | 0.448 |

→ judge v2가 v1을 두 축 모두 지배(파레토). tau가 운영 손잡이. 오탐 엄격 0.583 vs 공정 0.333 = "틀렸다"의 절반이 실제 소수지지 감정.

---

## 11. 지식 소스 어블레이션 (A/B/C)

복합감정 정의를 **데이터 기반 + 논문/자료**로 확장하고 소스별로 검증.

| 조건 | 소스 | 상태 |
|---|---|---|
| **A** | 데이터 기반 co_occurs만 | ✅ 완료(§10) |
| **B** | 논문·자료 유래 복합감정만 | 논문 확보 후 |
| **C** | A+B (하이브리드) | 논문 확보 후 |

논문 주입 경로: (a) 구조화 표 → custom_kg(`source_id=lit_*`), (b) 원문 → 네이티브 인제스트, (c) RAG 근거 코퍼스. 삼각 검증(데이터+문헌+발화)으로 신뢰도 차등.

---

## 12. Ollama 운영 (실전 이슈 포함)

| 항목 | 값/주의 |
|---|---|
| 임베딩 | `bge-m3:latest`(1024d). 모델 변경 시 rag_storage 초기화 필수 |
| LLM(judge) | `exaone3.5:32b-instruct-q4_K_M`, temperature 0.2 |
| 컨텍스트 | `OLLAMA_LLM_NUM_CTX` > MAX_TOTAL_TOKENS+2000 |
| **bge-m3 NaN** | 임베딩 텍스트에 수치 근거 넣으면 부하 시 NaN → description은 의미문장만, `EMBEDDING_FUNC_MAX_ASYNC=2` |
| **상황검색 매핑** | query 결과에 source_id 없음 → `full_doc_id` 사용, `distance`=코사인유사도 |
| **judge 호출** | `rag.llm_model_func` 직접호출은 hashing_kv 필요 → `ollama.AsyncClient` 직접 사용 |
| CSV | BOM 있어 `utf-8-sig`로 읽음 |

**WebUI 그래프 뷰**: 예제 폴더 `.env` + `lightrag-server` 실행 → `http://localhost:9621` Knowledge Graph 탭(노드 54개 확인 가능).

---

## 13. 산출물 (`examples/emotion_graphrag/`)

| 파일 | 역할 |
|---|---|
| `schema.py` | 분류표 54노드 상수 + custom_kg 빌더 |
| `rules.py` | wilson/lift/support + fuse/sit_score + 상수(R1) |
| `aggregate.py` | CSV → co_occurs(무향+방향) + chunk + labels_map |
| `build_layer1.py` | custom_kg 조립 → `ainsert_custom_kg` |
| `infer.py` | 후보 생성 + judge v1/v2(batch) |
| `evaluate.py` | target/acceptable 평가 + cross-major 분리 + tau 스윕 |
| `verify_value.py` | 그래프 가치 객관 검증(baseline 대비) |
| `labels_map.json` / `cooccur_stats.json` / `holdout.json` | 사이드 산출물 |

---

## 14. 진행 / 보류 / 다음

**완료**: 노드·엣지·chunk 빌드 / 하이브리드 후보 / judge v2 / target-acceptable 평가 / 객관 검증 / 조건 A.

**다음**: tau 운영점 확정 · 경로(멀티홉) 스코어링 · 정식 가드레일 · Gemini 시뮬 현실화 · 조건 B·C(논문).

**보류(데이터 사정)**

| 항목 | 사유 | 재개 조건 |
|---|---|---|
| triggered_by(선2) | original_major_code 오염 | 상황 필드 정제 |
| cue_indicates | cue 데이터 없음 | 비언어 cue 확보 |
| 2층 개인화 | 화자 대부분 1~2건 | 반복 화자/세션 누적 |

---

## 불변 원칙

노드=분류표 고정 · 엣지=train 집계(lift+wilson, 근거 저장) · holdout 미투입 · custom_kg 직접 주입(라벨 희석 금지) · 정밀 가중치·규칙은 코드 결정론, 그래프는 후보·근거 공급 · 평가는 target(다수결)+acceptable(≥2)로 주관·복합성 반영.
