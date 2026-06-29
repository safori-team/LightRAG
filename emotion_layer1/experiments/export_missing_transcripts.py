"""Export records that need transcript backfill from an experiment gold JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_jsonl", type=Path)
    parser.add_argument("out_jsonl", type=Path)
    args = parser.parse_args()

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.gold_jsonl.open(encoding="utf-8") as src, args.out_jsonl.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("transcript"):
                continue

            missing = {
                "record_id": record["record_id"],
                "annotation_id": record["annotation_id"],
                "clip_id": record["clip_id"],
                "storage_key": record.get("storage_key"),
                "transcript_to_fill": "",
                "gold_labels": {
                    "emotions": record["gold_labels"].get("emotions", []),
                    "major_categories": record["gold_labels"].get(
                        "major_categories", []
                    ),
                    "original_label": record["gold_labels"].get("original_label", {}),
                },
            }
            out.write(json.dumps(missing, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")
            count += 1

    print(f"wrote {count} missing transcript records to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

