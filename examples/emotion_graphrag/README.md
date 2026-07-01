# emotion_graphrag

Emotion GraphRAG reviewer on LightRAG + Ollama. Full design: [DESIGN.md](DESIGN.md).

The graph does not classify emotions; it reviews Gemini's 1st-pass output to
surface **missing** emotions, using a global co_occurs prior + situational
similarity, and supplies CBT-personalization evidence.

## Modules

| file | role | needs |
|---|---|---|
| `schema.py` | fixed taxonomy (6 major / 48 minor) + custom_kg builders | pure |
| `rules.py` | wilson/lift/support, fuse, sit_score, judge gate (R1 constants) | pure |
| `aggregate.py` | CSV → co_occurs edges + directional stats + chunks + labels | pure |
| `build_layer1.py` | assemble custom_kg → `ainsert_custom_kg` (Ollama embeddings) | LightRAG + Ollama |
| `infer.py` | reviewer loop: graph + situational → fuse → judge | LightRAG + Ollama |

Pure modules are validated with real data; `build`/`infer` need a running
Ollama server. Run scripts **from this directory** (they import siblings).

## Run

```bash
# 1) point to the labels CSV + Ollama models (or edit ../.env)
export EMOTION_CSV="C:/Users/windowadmin6/Desktop/safori/data/emotion_experiment_dataset.csv"
export EMBEDDING_MODEL=bge-m3:latest   EMBEDDING_DIM=1024
export LLM_MODEL=exaone3.5:7.8b
ollama pull bge-m3 && ollama pull exaone3.5:7.8b

# 2) build layer 1 (train split only; writes rag_storage/ + side JSONs)
python build_layer1.py

# 3) smoke-test the reviewer against holdout
python infer.py
```

## Outputs (git-ignored working data)

- `rag_storage/` — LightRAG graph + vector index (workspace `emotion`)
- `labels_map.json` — `utt_{clip_id} → {major, sub}` for situational lookup
- `cooccur_stats.json` — directional `A|B → P(B|A)`-based weight (side table)
- `holdout.json` — speaker-disjoint eval set (never injected)

## Notes

- Nodes are fixed by the codebook; edges are aggregated from **per-annotator**
  sub-code readings (not the majority label). See DESIGN.md §1, §4.
- co_occurs graph edges are **undirected with a symmetric weight** (NetworkX is
  undirected); asymmetric P(B|A) lives in `cooccur_stats.json`.
- Thresholds in `rules.py` are starting points — tune on the holdout
  recall/false-add tradeoff, not frozen.
- Deferred (data reasons): triggered_by (선2), cue_indicates, layer-2
  personalization. See DESIGN.md §10.
