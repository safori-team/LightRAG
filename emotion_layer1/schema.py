"""Label record contract + JSONL loader + validation.

Gold-label record (one JSON object per line). Minimal labeling schema:
sentence + gold emotions (+ nonverbal cue). Gemini's rich prediction is kept
SEPARATE under ``model_outputs.gemini`` and is NOT used for layer-1 build.

    {
      "record_id": "r001",
      "speaker_id": null,                 # filled later for speaker-wise split
      "transcript": "...",
      "split": "train",                   # "train" | "holdout"
      "gold_labels": {
        "emotions": [
          {"major": "슬픔", "minor": "사랑", "confidence": 0.9}
        ],
        "nonverbal_raw": "목소리가 떨림",  # labeler free text (not aggregated)
        "nonverbal_tags": ["voice_tremor"],# controlled vocab -> CUE nodes
        "context": ""                      # free string; 선2 deferred
      },
      "model_outputs": {"gemini": { ... }} # ignored by layer-1
    }

Only ``gold_labels`` drives the layer-1 graph. Validation reports unknown
labels against the taxonomy instead of silently dropping them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .taxonomy import Taxonomy


@dataclass
class Emotion:
    major: str
    minor: str
    confidence: float = 1.0


@dataclass
class Record:
    record_id: str
    transcript: str
    emotions: list[Emotion]
    nonverbal_tags: list[str] = field(default_factory=list)
    nonverbal_raw: str = ""
    context: str = ""
    speaker_id: str | None = None
    split: str = "train"


@dataclass
class ValidationReport:
    n_records: int = 0
    unknown_majors: dict[str, int] = field(default_factory=dict)
    unknown_minors: dict[str, int] = field(default_factory=dict)
    unknown_cues: dict[str, int] = field(default_factory=dict)
    minor_major_mismatch: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.unknown_majors
            or self.unknown_minors
            or self.unknown_cues
            or self.minor_major_mismatch
        )

    def summary(self) -> str:
        return (
            f"records={self.n_records} "
            f"unknown_major={dict(self.unknown_majors)} "
            f"unknown_minor={dict(self.unknown_minors)} "
            f"unknown_cue={dict(self.unknown_cues)} "
            f"minor!=major={self.minor_major_mismatch}"
        )


def parse_record(obj: dict) -> Record:
    gold = obj.get("gold_labels", {})
    emotions = [
        Emotion(
            major=e["major"],
            minor=e["minor"],
            confidence=float(e.get("confidence", 1.0)),
        )
        for e in gold.get("emotions", [])
    ]
    return Record(
        record_id=obj["record_id"],
        transcript=obj.get("transcript", ""),
        emotions=emotions,
        nonverbal_tags=list(gold.get("nonverbal_tags", [])),
        nonverbal_raw=gold.get("nonverbal_raw", ""),
        context=gold.get("context", ""),
        speaker_id=obj.get("speaker_id"),
        split=obj.get("split", "train"),
    )


def load_jsonl(path: str) -> list[Record]:
    records: list[Record] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(parse_record(json.loads(line)))
    return records


def validate(records: list[Record], tax: Taxonomy) -> ValidationReport:
    rep = ValidationReport(n_records=len(records))
    for r in records:
        for e in r.emotions:
            if not tax.has_major(e.major):
                rep.unknown_majors[e.major] = rep.unknown_majors.get(e.major, 0) + 1
            if not tax.has_minor(e.minor):
                rep.unknown_minors[e.minor] = rep.unknown_minors.get(e.minor, 0) + 1
            elif tax.major_of(e.minor) != e.major:
                rep.minor_major_mismatch.append(
                    f"{r.record_id}: {e.minor}->{e.major} (taxonomy={tax.major_of(e.minor)})"
                )
        for c in r.nonverbal_tags:
            if not tax.has_cue(c):
                rep.unknown_cues[c] = rep.unknown_cues.get(c, 0) + 1
    return rep


def split_records(records: list[Record], which: str = "train") -> list[Record]:
    """Record-level split for now (speaker_id is null at this stage).

    When speaker_id is populated, switch this to speaker-wise grouping so a
    speaker never appears in both train and holdout (no data leak).
    """
    return [r for r in records if r.split == which]
