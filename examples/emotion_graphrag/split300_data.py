"""Load the source-prefix-disjoint 300-record split and Gemini analyses."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import labels_kr
import schema

HERE = Path(__file__).resolve().parent
DEFAULT_SPLIT_DIR = HERE / "results/source_prefix_split_300"
W30_AUDIO = Path(
    "/Users/hann/Project/SAFORI/labeled-data/exports/emotion_relabeling/"
    "weekly/2026-W30/snapshot_2026-07-26/audio"
)
W31_GEMINI = Path(
    "/Users/hann/Project/SAFORI/data/"
    "gemini-2026-08-02-team-reference-pending-100"
)


@dataclass
class SplitRecord:
    clip_id: str
    stem: str
    source_prefix: str
    source_batch: str
    split: str
    transcript: str
    summary: str
    prosody: str
    detected: set[str]
    major: set[str]
    readings: dict[str, set[str]]

    def votes(self) -> Counter:
        result = Counter()
        for labels in self.readings.values():
            result.update(labels)
        return result

    def gold(self, threshold: int = 2) -> set[str]:
        return {code for code, count in self.votes().items() if count >= threshold}


def _emotion_code(name: str) -> str:
    value = name.strip().lower()
    return "ANXIETY_GENERAL" if value == "anxiety" else value.upper()


def _w31_by_clip_id() -> dict[str, dict]:
    output = {}
    for path in W31_GEMINI.glob("*.json"):
        if path.name == "batch-summary.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        output[str(data["clipId"])] = data
    return output


def _analysis(row: dict, w31: dict[str, dict]) -> dict:
    stem = Path(row["storage_key"]).stem
    if row["source_batch"] == "w30_legacy":
        path = W30_AUDIO / "analysis_legacy_labeled_100" / f"{stem}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
    elif row["source_batch"] == "w30_pending":
        path = W30_AUDIO / "analysis_team_reference_pending_100" / f"{stem}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
    elif row["source_batch"] == "w31_pending":
        data = w31[row["clip_id"]]
    else:
        raise ValueError(f"unknown source batch: {row['source_batch']}")
    return data.get("result") or data


def load_split(split: str, split_dir: Path = DEFAULT_SPLIT_DIR) -> list[SplitRecord]:
    w31 = _w31_by_clip_id()
    with (split_dir / f"{split}.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    records = []
    for row in rows:
        result = _analysis(row, w31)
        detected, majors, notes = set(), set(), []
        for segment in result.get("segments", []):
            category = (segment.get("category") or "").strip().upper()
            if category in schema.ALL_MAJORS:
                majors.add(category)
            for emotion in segment.get("emotions", []):
                code = _emotion_code(emotion.get("name") or "")
                if schema.is_known_sub(code):
                    detected.add(code)
            note = (segment.get("prosody_notes") or "").strip()
            if note:
                notes.append(note)
        if not majors:
            majors = {schema.SUB2MAJOR[code] for code in detected}
        readings = {}
        for annotator in labels_kr.ANNOTATORS:
            labels = set()
            for value in (row.get(f"{annotator}_sub_tags") or "").split(";"):
                code = labels_kr.SUB_KR2CODE.get(value.strip())
                if code:
                    labels.add(code)
            major = (row.get(f"{annotator}_major") or "").strip()
            if labels or major:
                readings[annotator] = labels
        records.append(SplitRecord(
            clip_id=row["clip_id"],
            stem=Path(row["storage_key"]).stem,
            source_prefix=row["source_prefix"],
            source_batch=row["source_batch"],
            split=split,
            transcript=(result.get("transcript") or "").strip(),
            summary=(result.get("summary") or "").strip(),
            prosody=" ".join(notes),
            detected=detected,
            major=majors,
            readings=readings,
        ))
    return records


def coverage(records: list[SplitRecord]) -> dict:
    prefixes = {record.source_prefix for record in records}
    return {
        "records": len(records),
        "source_prefixes": len(prefixes),
        "missing_transcript": sum(not record.transcript for record in records),
        "reading_counts": dict(sorted(Counter(
            len(record.readings) for record in records
        ).items())),
        "avg_gold": round(sum(len(record.gold()) for record in records) / len(records), 3),
        "avg_detected": round(sum(len(record.detected) for record in records) / len(records), 3),
    }
