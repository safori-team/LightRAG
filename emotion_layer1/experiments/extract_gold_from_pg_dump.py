"""Extract annotation-level gold JSONL from the labeler PostgreSQL dump.

The dump is parsed directly from pg_dump COPY sections so a local PostgreSQL
restore is not required. Output is intended as local experiment data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

NULL = r"\N"


def _copy_unescape(value: str) -> str | None:
    if value == NULL:
        return None
    out: list[str] = []
    i = 0
    while i < len(value):
        if value[i] != "\\" or i + 1 >= len(value):
            out.append(value[i])
            i += 1
            continue
        nxt = value[i + 1]
        if nxt == "n":
            out.append("\n")
        elif nxt == "r":
            out.append("\r")
        elif nxt == "t":
            out.append("\t")
        elif nxt == "\\":
            out.append("\\")
        else:
            out.append(nxt)
        i += 2
    return "".join(out)


def _read_copy_sections(path: Path) -> dict[str, list[list[str | None]]]:
    wanted = {
        "annotations",
        "annotation_selections",
        "clips",
        "emotion_categories",
        "original_labels",
    }
    sections: dict[str, list[list[str | None]]] = {name: [] for name in wanted}
    current: str | None = None

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("COPY public."):
                table = line.split()[1].removeprefix("public.")
                current = table if table in wanted else None
                continue
            if current and line == "\\.\n":
                current = None
                continue
            if current:
                sections[current].append(
                    [_copy_unescape(part) for part in line.rstrip("\n").split("\t")]
                )
    return sections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(
    sql_path: Path, limit: int, *, require_transcript: bool = False
) -> tuple[list[dict], dict]:
    data = _read_copy_sections(sql_path)

    categories: dict[int, dict] = {}
    for row in data["emotion_categories"]:
        cid, parent_id, level, code, name, display_order = row
        categories[int(cid)] = {
            "id": int(cid),
            "parent_id": None if parent_id is None else int(parent_id),
            "level": level,
            "code": code,
            "name": name,
            "display_order": None if display_order is None else int(display_order),
        }

    transcripts: dict[int, dict] = {}
    for row in data["original_labels"]:
        clip_id, major_code, sub_tags, transcript, raw = row
        transcripts[int(clip_id)] = {
            "major_code": major_code,
            "sub_tags": sub_tags,
            "transcript": transcript or "",
            "raw": raw,
        }

    clips: dict[int, dict] = {}
    for row in data["clips"]:
        clip_id, storage_key, duration_sec, status, claimed_by, claimed_at, created_at, is_shared = row
        clips[int(clip_id)] = {
            "storage_key": storage_key,
            "duration_sec": None if duration_sec is None else float(duration_sec),
            "status": status,
            "is_shared": is_shared == "t",
        }

    annotations: list[dict] = []
    for row in data["annotations"]:
        ann_id, clip_id, source, annotator_id, model_version, raw_response, created_at, updated_at = row
        if source != "HUMAN":
            continue
        annotations.append({
            "annotation_id": int(ann_id),
            "clip_id": int(clip_id),
            "source": source,
            "annotator_id": None if annotator_id is None else int(annotator_id),
            "model_version": model_version,
            "created_at": created_at,
            "updated_at": updated_at,
        })
    annotations.sort(key=lambda item: item["annotation_id"])

    selections: dict[int, list[dict]] = defaultdict(list)
    for row in data["annotation_selections"]:
        _sel_id, ann_id, category_id, confidence = row
        cat = categories[int(category_id)]
        selections[int(ann_id)].append({
            "category_id": cat["id"],
            "level": cat["level"],
            "code": cat["code"],
            "name": cat["name"],
            "confidence": None if confidence is None else float(confidence),
        })

    records: list[dict] = []
    for ann in annotations:
        original = transcripts.get(ann["clip_id"], {})
        if require_transcript and not original.get("transcript"):
            continue
        if len(records) >= limit:
            break
        idx = len(records)
        clip = clips.get(ann["clip_id"], {})
        selected = sorted(
            selections.get(ann["annotation_id"], []),
            key=lambda item: (item["level"] != "MAJOR", item["category_id"]),
        )
        major_categories = [item for item in selected if item["level"] == "MAJOR"]
        emotions: list[dict] = []
        for item in selected:
            if item["level"] != "SUB":
                continue
            sub = categories[item["category_id"]]
            major = categories[sub["parent_id"]]
            emotions.append({
                "major": major["name"],
                "major_code": major["code"],
                "major_category_id": major["id"],
                "minor": sub["name"],
                "minor_code": sub["code"],
                "minor_category_id": sub["id"],
                "confidence": 1.0 if item["confidence"] is None else item["confidence"],
            })

        records.append({
            "record_id": f"ann_{ann['annotation_id']}",
            "annotation_id": ann["annotation_id"],
            "clip_id": ann["clip_id"],
            "storage_key": clip.get("storage_key"),
            "duration_sec": clip.get("duration_sec"),
            "clip_status": clip.get("status"),
            "is_shared": clip.get("is_shared"),
            "speaker_id": None,
            "annotator_id": ann["annotator_id"],
            "transcript": original.get("transcript", ""),
            "split": "train",
            "gold_labels": {
                "emotions": emotions,
                "major_categories": major_categories,
                "nonverbal_raw": "",
                "nonverbal_tags": [],
                "context": "",
                "original_label": {
                    "major_code": original.get("major_code"),
                    "sub_tags": original.get("sub_tags"),
                },
            },
            "model_outputs": {"gemini": {}},
        })

    split_at = int(len(records) * 0.8)
    for idx, record in enumerate(records):
        record["split"] = "train" if idx < split_at else "holdout"

    manifest = {
        "source_sql": str(sql_path),
        "source_sha256": _sha256(sql_path),
        "record_unit": "annotation",
        "selection_source": "annotations.source=HUMAN joined through annotation_selections",
        "requested_limit": limit,
        "require_transcript": require_transcript,
        "records": len(records),
        "train_records": sum(1 for r in records if r["split"] == "train"),
        "holdout_records": sum(1 for r in records if r["split"] == "holdout"),
        "records_with_transcript": sum(1 for r in records if r["transcript"]),
        "records_without_transcript": sum(1 for r in records if not r["transcript"]),
        "notes": [
            "Gold labels are annotation-level, not clip-level consensus.",
            "Only SUB category selections are converted into gold_labels.emotions.",
            "MAJOR selections are preserved under gold_labels.major_categories.",
            "The raw SQL dump is not copied into the repository.",
        ],
    }
    return records, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sql_path", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-name", default="gold.jsonl")
    parser.add_argument("--manifest-name", default="source_manifest.json")
    parser.add_argument("--require-transcript", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records, manifest = build_records(
        args.sql_path, args.limit, require_transcript=args.require_transcript
    )

    with (args.out_dir / args.output_name).open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    with (args.out_dir / args.manifest_name).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
