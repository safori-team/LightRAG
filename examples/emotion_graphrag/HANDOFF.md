# 감정 GraphRAG — 핸드오프 문서 (전체 맥락)

> 다른 환경/세션에서 작업을 이어가기 위한 종합 인수인계 문서.
> 목적: (1) 지금까지의 설계 진화(무엇이 왜 바뀌었나), (2) 현재 설계, (3) 구현된 것, (4) 측정 결과, (5) 다음에 할 것을 최대한 많은 맥락과 함께 남긴다.
> 현행 설계 상세는 `DESIGN.md`, 주차 보고서는 `REPORT_W6.md`/`REPORT_W7.md` 참조. 이 문서는 **결정의 히스토리와 근거**까지 포함한 상위 맥락 문서다.

---

## 0. 빠른 시작 (다른 환경에서 재현)

```bash
# 전제: Ollama 실행중(localhost:11434), bge-m3 / exaone3.5:32b pull, venv(lightrag)
cd examples/emotion_graphrag
# 공통 env (또는 이 폴더의 .env 사용)
export EMBEDDING_MODEL=bge-m3:latest EMBEDDING_DIM=1024
export LLM_MODEL=exaone3.5:32b-instruct-q4_K_M
export EMBEDDING_FUNC_MAX_ASYNC=2   # bge-m3 NaN 회피(중요)

python build_layer1.py            # 그래프 빌드 → rag_storage/ + 사이드 JSON
python evaluate.py                # 후보 회수율(LLM 무관) + cross-major
python evaluate.py --judge2       # judge v2 + 확신도 tau 스윕
python verify_value.py            # 그래프 가치 객관 검증(baseline 대비)
python render_graph.py            # 그래프 → emotion_graph.html(전달용)
```

- venv: 저장소의 `.venv` (lightrag 1.5.3). pip 없음(uv 생성) → 의존성은 `uv pip install`.
- 모델(이미 로컬에 있던 것): 임베딩 `bge-m3:latest`(1024d), judge `exaone3.5:32b-instruct-q4_K_M`(7.8b·qwen3:30b도 있음).
- 데이터: `C:\Users\windowadmin6\Desktop\safori\data\emotion_experiment_dataset.csv` (100건).

---

## 1. 프로젝트 한눈에

- **목표**: 음성 감정분석 → CBT 상담 개인화. 파이프라인 = 음성 → **Gemini 대분류** → **GraphRAG 소분류·복합감정 보완** → CBT 설명.
- **핵심 가설**: "감정은 복합적이고 상관되어 있다." → 감정 간 공존/상관 구조를 그래프로 모델링해 설명.
- **철학**: "언어는 그 사람의 세계를 담는다 — 같은 음성도 사람마다 다르게 필터된다." 대분류=보편, 그래프=개인화된 세계.
- **제약(공모전)**: 메인 계획에 GraphRAG를 쓰기로 했으므로 **GraphRAG 사용은 필수**. (성능이 아니라 조건.)

---

## 2. 설계 진화 히스토리 (무엇이 왜 바뀌었나) — 가장 중요한 맥락

초기 설계는 "검수관 1층(co_occurs) + 하이브리드 후보 + 2층 개인화"였다. 데이터·측정·조건에 부딪히며 아래처럼 크게 바뀌었다.

| # | 이전 | 현재 | 바뀐 이유 |
|---|---|---|---|
| A | 노드·엣지 전부 활용(선1/선2/cue), 2층 개인화 포함 | co_occurs(선1)만 우선, 나머지 보류 | 데이터 실측: 100건/57화자. 2층은 화자 대부분 1~2건(cold start). `original_major_code` 50% 공백+오타 → triggered_by(선2) 불가. cue 라벨 없음 → cue_indicates 불가 |
| B | 다수결 소분류로 공존 집계 | **어노테이터 5인 개별 리딩(497개)**로 집계 | 다수결은 71/100이 코드 1개 → 공존 신호 붕괴. 리딩은 5배·블렌드 관측 단위 |
| C | 엣지를 count/확률 **임계값**으로 컷 | **lift(우연 보정) + Wilson 하한(표본 신뢰)** | 소량 데이터에서 절대 임계값은 취약. lift가 "흔해서 붙는 쌍" 걸러냄. Wilson이 표본 적은 엣지 자동 감쇠 |
| C' | 방향별 co_occurs 엣지 2개 | **무향 1개 + 대칭 weight**, 방향은 사이드 테이블 | LightRAG 기본 그래프(NetworkX)가 무향 → 방향별 엣지가 덮어써짐 |
| D | 그래프가 회수 성능을 낸다고 가정 | **객관 검증: 전체 회수율은 baseline과 동률**, cross-major에서만 우위 | `verify_value.py`: 계층 휴리스틱(0.556)이 그래프(0.444)보다 나음. 누락의 61%가 같은 대분류라 계층으로 잡힘 |
| E | "GraphRAG를 쓴다" | 실제론 그래프+벡터 **저장·검색만** 씀(aquery 미사용), judge는 문맥 없는 **맨 LLM** | 재검토 중 발견. judge가 회수율을 44%→17%로 깎는 병목 |
| F | 도구 적합성 재고 | **GraphRAG 필수(조건)** → "쓸지"가 아니라 "어떻게 최대화"로 전환. 대분류=Gemini, 그래프=**소분류+복합감정(cross-major)**, 개인화 나중 | 공모전 조건. + cross-major가 그래프 유일 우위 영역이라 거기 특화 |
| G | 다수결만 정답 | **target(다수결) + acceptable(≥2 어노테이터)** 2단계 | 다수결-only는 소수지지(복합·주관) 감정을 과잉 처벌 → "감정은 주관적" 전제와 모순. 오탐율 0.583→0.333로 실제 판명 |
| H | judge = 맨 LLM 이진 YES/NO | **judge v2**: 선택형 + 근거주입(RAG) + 확신도 점수 + 모델 32B | 병목 해소. v2가 v1을 두 축(회수·오탐) 모두 지배 |
| I | 복합감정 정의 미정 | **데이터 기반(A)** 기본 + 논문/자료(B) + 하이브리드(C) 어블레이션 | 논문 확보 시 소스별 검증 계획. 현재 A만 완료(논문 대기) |

**요약 서사**: 데이터가 작아서 야심찬 초기 계획(2층·선2·cue)을 접고 co_occurs에 집중 → 통계 처리를 임계값에서 lift+Wilson으로 교체 → 객관 검증으로 "그래프의 진짜 가치는 cross-major 복합감정"임을 확인 → 공모전 조건상 GraphRAG를 필수로 두고 그 니치에 특화 → 평가 잣대를 target/acceptable로 바로잡고 judge를 v2로 개선. 가설("감정은 복합적")은 데이터로 이미 지지됨.

---

## 3. 현재 설계 (요약)

- **빌드(오프라인)**: 라벨을 직접 집계해 `ainsert_custom_kg`로 주입(LLM 추출 우회, 라벨 희석 방지). 노드=분류표 고정 54개, belongs_to 48, co_occurs(무향+대칭 weight), 발화 chunk.
- **후보 생성(하이브리드)**: ① graph(get_node_edges + 방향 weight) + ② situational(발화 임베딩→유사 발화 라벨). `rules.fuse`로 융합, 상위 TOP_K(=8).
- **재판정(judge v2)**: 후보 전체를 근거와 함께 한 번에 LLM에 제시 → 0~1 확신도 → tau 임계로 추가.
- **평가**: target(다수결 회수) / acceptable(≥2 어노테이터는 오탐 아님) / cross-major 분리 / holdout 누수 차단.
- **결정론 분리(R1)**: 모든 임계·가중치·융합계수는 `rules.py` 상수. 그래프엔 계산된 숫자만.

상세는 `DESIGN.md` 참조.

---

## 4. 구현된 것 (파일별 · 동작 검증 상태)

`examples/emotion_graphrag/`

| 파일 | 역할 | 상태 |
|---|---|---|
| `schema.py` | 분류표 54노드 상수(TAXONOMY) + custom_kg dict 빌더 | ✅ 검증 |
| `rules.py` | wilson/lift/support, fuse, sit_score, 상수(TOP_K=8, tau 등) | ✅ 검증 |
| `aggregate.py` | CSV→어노테이터 리딩→co_occurs(무향+방향 사이드) + chunk + labels_map + 화자 split | ✅ 검증 |
| `build_layer1.py` | custom_kg 조립 → `ainsert_custom_kg`(Ollama 임베딩) | ✅ 실행 성공 |
| `infer.py` | 후보 생성(그래프+상황) + judge v1(YES/NO) + **judge v2(batch, 근거·확신도)** | ✅ 실행 성공 |
| `evaluate.py` | target/acceptable 평가 + cross-major 분리 + `--judge`(v1) / `--judge2`(v2 tau 스윕) | ✅ 실행 성공 |
| `verify_value.py` | 그래프 가치 객관 검증(baseline B1/B2/B3) | ✅ 실행 성공 |
| `render_graph.py` | graphml → `emotion_graph.html`(전달용 인터랙티브) | ✅ 실행 성공 |
| `DESIGN.md` / `README.md` | 현행 설계 / 실행법 | — |
| `REPORT_W6.md` / `REPORT_W7.md` | 주차 보고서 | — |
| `.env` | 로컬 서버/실행용 설정 | — |

**산출물(빌드 결과, git 비추천)**: `rag_storage/`(그래프+벡터), `labels_map.json`, `cooccur_stats.json`(방향 통계), `holdout.json`(평가셋), `emotion_graph.html`.

**빌드 규모(train 69)**: 노드 54, belongs_to 48, co_occurs 117, chunk 68.

**WebUI 확인**: `lightrag-server`(예제 폴더 `.env` 필요) → `http://localhost:9621` Knowledge Graph 탭.

---

## 5. 측정 결과 (전부, holdout 8케이스 / 누락 18 = same 11 + cross 7 · 소량→방향성)

**그래프 가치 객관 검증 (LLM 무관)**
| 방식 | 그래프? | 회수율@5 |
|---|---|---|
| B1 global-freq | ❌ | 0.444 |
| B2 same-major(계층만) | ❌ | 0.556 |
| B3 co_occur-1hop(그래프) | ✅ | 0.444 |
→ 전체 회수율은 그래프 우위 없음. **cross-major(계층 구조상 0%)에서만 그래프가 회수** = 고유 가치.

**후보 회수율(검색 폭)**
| TOP_K | overall | cross-major |
|---|---|---|
| 5 | 0.444 | 0.429 |
| 8 | 0.611 | 0.429 |

**LLM 재판정(32B, hybrid, target/acceptable)**
| 설정 | target 회수율 | 오탐[엄격] | 오탐[공정≥2] |
|---|---|---|---|
| v1 이진 7.8B | 0.167 | — | 0.750 |
| v1 이진 32B | 0.278 | 0.583 | 0.333 |
| v2 tau0.7 | 0.389 | — | **0.250** |
| v2 tau0.5 | **0.500** | — | 0.448 |
| v2 tau0.3 | 0.556 | — | 0.571 |
→ judge v2가 v1을 파레토 지배. tau가 회수↔오탐 손잡이. 잣대 엄격 0.583 vs 공정 0.333 = "틀림"의 절반이 실제 소수지지 감정.

---

## 6. 핵심 의사결정과 근거 (요약)

- **노드=분류표 고정**: 데이터 추출 금지, custom_kg 직접 주입 → 라벨 희석 방지, 결정론·감사가능성.
- **co_occurs=어노테이터 리딩**: 다수결은 공존 신호 붕괴, 리딩이 블렌드 관측 단위.
- **엣지=lift+Wilson(임계값 아님)**: 소량 데이터에 강건, 우연/표본 보정.
- **무향+사이드 테이블**: NetworkX 무향 제약 회피 + 방향 정보 보존(R1).
- **스코프=cross-major 복합감정**: 그래프 유일 우위 영역 + 가설과 직결.
- **평가=target/acceptable**: 주관·복합 감정을 오탐 처벌하지 않음.
- **judge v2**: 근거 없는 맨 LLM 병목을, 근거주입+선택형+확신도로 해소.

---

## 7. 알려진 한계 / 열린 질문

- **표본 소량(n=100, holdout cross 7건)** → 모든 수치는 통계적 결론이 아닌 **방향성**. 데이터 확보 시 재측정 필수.
- **텍스트 천장**: gold 라벨은 **음성 기반**인데 judge는 텍스트만 봄 → 어떤 텍스트 방법도 완전히 못 넘음. 장기적으로 Gemini 음성 신호를 판정에 결합해야.
- **Gemini 시뮬 조악**: 현재 평가는 `detected = gold_sub[0]`(순서 임의). 실제(음성+대분류→소분류) 파이프라인과 간극.
- **그래프 활용도**: 현재 1-hop co_occurs 조회만 = 사실상 "공존 사전". 멀티홉·커뮤니티 등 그래프 고유 기능 미사용.
- **열린 결정**: tau 운영점(0.5 회수 우선 vs 0.7 정밀 우선), 복합감정 노드 정의(데이터 기반 확정).

---

## 8. 다음에 진행할 것 (로드맵 · 우선순위)

**바로 이어갈 후보**
1. **tau 운영점 확정** — 용도(CBT 검수관=회수 우선)에 맞춰 0.5 근방 채택 여부.
2. **judge v2 고도화** — exemplar 발화까지 근거로 주입, 프롬프트 few-shot, 가드레일(출력 스키마 검증·거부 응답) 정식화.
3. **경로 기반(멀티홉) 스코어링** — belongs_to·co_occurs 2홉 순회로 간접 연관 반영(진짜 그래프 활용).
4. **cross-major 특화 그래프 보강** — CulturalComplex 노드(섭섭류) + `blends_into`(대분류 넘는) 엣지.
5. **Gemini 시뮬 현실화** — 실제 대분류→소분류 흐름 반영한 평가.

**지식 소스 어블레이션 (논문 확보 후)**
- 조건 A(데이터)=완료. B(논문 단독)·C(하이브리드)는 논문 필요.
- 논문 주입 경로: (a) 구조화 표→custom_kg(`source_id=lit_*`), (b) 원문→네이티브 인제스트, (c) RAG 근거 코퍼스.
- **소스 태깅** 후 `evaluate`에서 `--sources data|lit|data,lit` 토글로 A/B/C 비교. 삼각 검증(데이터+문헌+발화)으로 신뢰 차등.
- 필요 입력: 논문/자료(표 형태가 최선, PDF면 추출 단계 추가).

**데이터가 쌓이면**
- **2층 개인화**: 화자별 workspace + EmotionInstance + instance_of + boost 수식(폭주 상한). 반복 화자/세션 필요.
- **cross-major 통계 안정화**: n=100은 부족. 얼마나 더 모아야 하는지 수집 설계 필요.
- (선택) **네트워크 심리측정(partial correlation/graphical LASSO)**을 엣지 선택기로 → 간접경로 제거, 임의 lift 컷 대체. 단 데이터 hungry.

**보류(데이터 사정)**: triggered_by(선2, original 필드 오염), cue_indicates(cue 없음), 2층 개인화(cold start).

---

## 9. 실행 환경 상세

| 항목 | 값 |
|---|---|
| venv | 저장소 `.venv` (lightrag 1.5.3), pip 없음 → `uv pip install` |
| 임베딩 | `bge-m3:latest` (1024d). 모델 변경 시 rag_storage 초기화 필수 |
| judge LLM | `exaone3.5:32b-instruct-q4_K_M` (7.8b·qwen3:30b도 로컬 존재) |
| Ollama | `http://localhost:11434` |
| 핵심 env | `EMBEDDING_FUNC_MAX_ASYNC=2`(NaN 회피), `OLLAMA_LLM_NUM_CTX>=32768`, `WORKSPACE=emotion` |
| 실행 위치 | `examples/emotion_graphrag/`에서 실행(형제 모듈 import) |

---

## 10. 데이터 지도

파일: `C:\Users\windowadmin6\Desktop\safori\data\emotion_experiment_dataset.csv` (100건, BOM有→utf-8-sig).

| 컬럼 | 용도 |
|---|---|
| `human_majority_major_codes` | 대분류 골드(Gemini 담당 영역) |
| `human_majority_sub_codes` | 소분류 골드 = **target** |
| `annotator_labels`(5인 major/sub) | **co_occurs 집계원** + acceptable(≥2) 계산원 |
| `transcript` | 발화 chunk(상황 검색 인덱스) |
| `filename` 접두사(F0001..) | speaker_id(화자 split/미래 2층) |
| `original_major_code` | 원본 태그(오염) — 선2 보류 사유 |

분류표: 대분류 6 / 소분류 48. 데이터 정합(오타·미등록 0), cold 노드 ECSTASY 1개.

---

## 11. 실전 이슈 (gotchas) — 재현 시 반드시 참고

- **bge-m3 NaN**: 임베딩 텍스트에 수치(lift/P 등) 넣으면 부하 시 NaN 반환 → description은 의미문장만, `EMBEDDING_FUNC_MAX_ASYNC=2`.
- **무향 그래프**: co_occurs 방향별 2엣지는 덮어써짐 → 무향+대칭 weight, 방향은 `cooccur_stats.json`.
- **CSV BOM**: 첫 컬럼 `﻿clip_id` → `utf-8-sig`로 읽음.
- **chunks_vdb.query 결과**: `source_id` 없음 → `full_doc_id`(=utt_{clip_id}) 사용, `distance`=코사인유사도.
- **judge LLM 직접호출**: `rag.llm_model_func`는 `hashing_kv` 필요 → `ollama.AsyncClient` 직접.
- **fuse 비결정성**: 동점 정렬 랜덤 → 2차 정렬키(emotion) 고정으로 재현성 확보.
- **서버**: 시작 디렉터리에 `.env` 없으면 대화형 프롬프트로 멈춤(비대화 환경 EOF) → 예제 폴더 `.env` 필요.
- **pip 없음**: venv가 uv 생성 → `uv pip install` 사용.

---

## 12. 관련 문서·메모리 포인터

- `DESIGN.md` — 현행 설계 상세(섹션별).
- `REPORT_W6.md` / `REPORT_W7.md` — 주차 보고서(계획 대비 수행, 정확 표기).
- `emotion_graph.html` — 그래프 시각화(전달용, vis-network CDN).
- 프로젝트 메모리(`~/.claude/projects/.../memory/`): `emotion-graphrag-design`, `emotion-dataset-facts`, `emotion-graphrag-value-check` — 설계 결정·데이터 실측·객관 검증 결과.

---

## 부록: 현재 성능 스냅샷 (한 줄)

조건 A(데이터), 하이브리드, 32B judge v2, tau0.5 기준 — **후보 회수율 0.611(cross-major 0.429), 최종 target 회수율 0.500, 공정 오탐율 0.448**. 모두 holdout 8케이스 기반 방향성 지표.
