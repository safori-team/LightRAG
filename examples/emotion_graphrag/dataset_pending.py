"""Adapter: pending_100 manifest + Gemini analysis JSON -> evaluation records.

The evaluation set is split across two files per clip and neither matches the
schema the training pipeline expects:

  manifests/team_reference_pending_100.csv
      human labels as "ANGER;SAD;DISTRESS", column `human_subs`
      `original_transcript` is EMPTY for all 100 rows
  audio/analysis_team_reference_pending_100/<clip>.json
      Gemini's output: transcript, summary, per-segment emotions in lowercase

This module joins them on the audio filename and emits one flat record per
clip so the rest of the evaluation never touches file formats.

Gemini's emotion names are already the official sub codes in lowercase; the
only mismatch in the data is `anxiety` -> `ANXIETY_GENERAL` (verified across
both analysis folders: every other name maps by uppercasing).
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import schema

SNAPSHOT = Path(
    "/Users/hann/Project/SAFORI/labeled-data/exports/emotion_relabeling/"
    "weekly/2026-W30/snapshot_2026-07-26"
)

GROUPS = {
    "pending100": (
        SNAPSHOT / "manifests/team_reference_pending_100.csv",
        SNAPSHOT / "audio/analysis_team_reference_pending_100",
    ),
    "legacy100": (
        SNAPSHOT / "manifests/legacy_labeled_100.csv",
        SNAPSHOT / "audio/analysis",
    ),
}

# Gemini emotion name -> official sub code, for the names that are not a plain
# uppercase of the code.
NAME_FIX = {"anxiety": "ANXIETY_GENERAL"}


@dataclass
class Record:
    clip: str                       # F2001_000002 (manifest filename stem)
    clip_id: str                    # server clip id, for traceability
    speaker: str                    # F2001
    transcript: str                 # from Gemini (manifest has none)
    summary: str
    prosody: str
    detected: set[str] = field(default_factory=set)   # Gemini 1st pass
    gold: set[str] = field(default_factory=set)       # human labels
    unknown_detected: set[str] = field(default_factory=set)
    unknown_gold: set[str] = field(default_factory=set)


def _to_code(name: str) -> str:
    return NAME_FIX.get(name.strip().lower(), name.strip().upper())


def _load_gold(manifest: Path) -> dict[str, dict]:
    """filename stem -> {clip_id, subs}. utf-8-sig: the CSV carries a BOM."""
    out: dict[str, dict] = {}
    with open(manifest, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = os.path.splitext(row["audio_filename"])[0]
            out[stem] = {
                "clip_id": row["clip_id"],
                "subs": {s for s in (row.get("human_subs") or "").split(";") if s},
            }
    return out


def _load_gemini(analysis_dir: Path) -> dict[str, dict]:
    """filename stem -> {transcript, summary, prosody, emotions}.

    Gold labels are per clip while Gemini labels are per segment, so the
    segment emotions are unioned up to clip level to make the two comparable.
    """
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(str(analysis_dir / "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        emotions: set[str] = set()
        notes: list[str] = []
        for seg in j.get("segments", []):
            for e in seg.get("emotions", []):
                name = e.get("name")
                if name:
                    emotions.add(_to_code(name))
            note = (seg.get("prosody_notes") or "").strip()
            if note:
                notes.append(note)
        out[stem] = {
            "transcript": (j.get("transcript") or "").strip(),
            "summary": (j.get("summary") or "").strip(),
            "prosody": " ".join(notes),
            "emotions": emotions,
        }
    return out


def load_records(group: str = "pending100") -> list[Record]:
    """Join manifest gold labels with Gemini predictions for one group.

    Codes outside the official 48 are dropped from the working sets but kept in
    ``unknown_*`` so a silent vocabulary drift shows up in the report instead of
    quietly deflating recall.
    """
    manifest, analysis_dir = GROUPS[group]
    gold = _load_gold(manifest)
    gemini = _load_gemini(analysis_dir)

    records: list[Record] = []
    for stem in sorted(gold):
        g = gemini.get(stem)
        if g is None:
            continue  # manifest row without a Gemini analysis file
        gold_subs = gold[stem]["subs"]
        detected = g["emotions"]
        records.append(Record(
            clip=stem,
            clip_id=gold[stem]["clip_id"],
            speaker=stem.split("_")[0],
            transcript=g["transcript"],
            summary=g["summary"],
            prosody=g["prosody"],
            detected={c for c in detected if schema.is_known_sub(c)},
            gold={c for c in gold_subs if schema.is_known_sub(c)},
            unknown_detected={c for c in detected if not schema.is_known_sub(c)},
            unknown_gold={c for c in gold_subs if not schema.is_known_sub(c)},
        ))
    return records


def coverage_report(records: list[Record]) -> dict:
    """Sanity numbers to print before an evaluation run."""
    n = len(records) or 1
    unknown_d = sorted({c for r in records for c in r.unknown_detected})
    unknown_g = sorted({c for r in records for c in r.unknown_gold})
    return {
        "records": len(records),
        "missing_transcript": sum(1 for r in records if not r.transcript),
        "avg_gold": round(sum(len(r.gold) for r in records) / n, 2),
        "avg_detected": round(sum(len(r.detected) for r in records) / n, 2),
        "unknown_detected_codes": unknown_d,
        "unknown_gold_codes": unknown_g,
        "gold_codes_never_predicted": sorted(
            {c for r in records for c in r.gold}
            - {c for r in records for c in r.detected}
        ),
    }


if __name__ == "__main__":
    for group in GROUPS:
        recs = load_records(group)
        print(f"== {group}")
        print(json.dumps(coverage_report(recs), ensure_ascii=False, indent=2))
        if recs:
            r = recs[0]
            print("sample:", json.dumps({
                "clip": r.clip, "transcript": r.transcript[:60],
                "detected": sorted(r.detected), "gold": sorted(r.gold),
            }, ensure_ascii=False))
