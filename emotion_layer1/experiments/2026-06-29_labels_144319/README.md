# labels-20260628-144319 Gold Extract

Local experiment extract for Gemini hallucination and layer-1 GraphRAG
candidate-recovery tests.

Files:

- `source_manifest.json`: source dump path, checksum, split counts, extraction notes.
- `gold.jsonl`: 500 annotation-level gold records. This file contains transcript
  text and is ignored by git.
- `gold_text.jsonl`: transcript-present subset for text-only Gemini tests. In
  this dump only 250 HUMAN annotation records have transcript text.
- `with_transcripts.jsonl`: same transcript-present subset with a clearer name
  for immediate tests.
- `missing_transcripts.jsonl`: 250 records needing transcript backfill. Fill
  `transcript_to_fill` for each record.

Extraction command:

```bash
python3 emotion_layer1/experiments/extract_gold_from_pg_dump.py \
  /Users/hann/Downloads/labels-20260628-144319.sql \
  emotion_layer1/experiments/2026-06-29_labels_144319 \
  --limit 500
```

Text-only extract:

```bash
python3 emotion_layer1/experiments/extract_gold_from_pg_dump.py \
  /Users/hann/Downloads/labels-20260628-144319.sql \
  emotion_layer1/experiments/2026-06-29_labels_144319 \
  --limit 500 \
  --require-transcript \
  --output-name gold_text.jsonl \
  --manifest-name source_manifest_text.json
```

Current split policy is deterministic annotation order: first 80% train, last
20% holdout. Speaker-level split is not available yet.

Missing transcript export:

```bash
python3 emotion_layer1/experiments/export_missing_transcripts.py \
  emotion_layer1/experiments/2026-06-29_labels_144319/gold.jsonl \
  emotion_layer1/experiments/2026-06-29_labels_144319/missing_transcripts.jsonl
```
