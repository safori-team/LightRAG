"""Weekly 5-annotator labeling exports (Korean labels) -> codes and records.

Two batches, disjoint clips, same five annotators:

    labeling_before_last_week_100.csv   100 clips  -> builds the graph
    labeling_last_week_100.csv          100 clips  -> the test set

Both store one column pair per annotator (``<name>_major`` / ``<name>_sub_tags``)
with semicolon-separated Korean labels. The Korean->code mapping is not
invented here: it is the one recorded in the production label dump
(emotion_layer1/experiments/2026-06-29_labels_144319/gold.jsonl, where each
label carries both its Korean name and its official code). Only ECSTASY never
appeared in that dump and is filled in from the taxonomy.

Having five readings per clip restores the two-tier gold standard the design
calls for (DESIGN.md §9): a majority label the reviewer must recover, and a
wider "at least two annotators agreed" set whose members are not false adds.
Utterance text comes from the Gemini analysis JSON, which is also the query
side at inference.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import schema

LABELED_DATA = Path("/Users/hann/Project/SAFORI/labeled-data")
SNAPSHOT = LABELED_DATA / (
    "exports/emotion_relabeling/weekly/2026-W30/snapshot_2026-07-26"
)

ANNOTATORS = ["박진하", "이승연", "이정한", "공윤서", "박준혁"]

BATCHES = {
    # batch name -> (label csv, gemini analysis dir)
    "before": (LABELED_DATA / "labeling_before_last_week_100.csv",
               SNAPSHOT / "audio/analysis"),
    "last": (LABELED_DATA / "labeling_last_week_100.csv",
             SNAPSHOT / "audio/analysis_team_reference_pending_100"),
}

MAJOR_KR2CODE = {
    "기쁨": "HAPPY", "슬픔": "SAD", "분노": "ANGRY",
    "불안": "ANXIETY", "놀람": "SURPRISE", "중립": "NEUTRAL",
}

SUB_KR2CODE = {
    "갈망": "CRAVING", "감탄": "ADMIRATION", "경멸": "CONTEMPT",
    "경애": "ADORATION", "경외": "AWE", "고통": "DISTRESS",
    "공감적 아픔": "EMPATHIC_PAIN", "공포": "HORROR",
    "긍정적 놀람": "SURPRISE_POSITIVE", "기쁨": "JOY", "깨달음": "REALIZATION",
    "낭만": "ROMANCE", "당혹": "EMBARRASSMENT", "두려움": "FEAR",
    "만족": "CONTENTMENT", "만족감": "SATISFACTION", "매혹": "ENTRANCEMENT",
    "무료함": "BOREDOM", "부정적 놀람": "SURPRISE_NEGATIVE", "분노": "ANGER",
    "불안": "ANXIETY_GENERAL", "사랑": "LOVE", "사색": "CONTEMPLATION",
    "설렘": "EXCITEMENT", "성취감": "TRIUMPH", "수치심": "SHAME",
    "슬픔": "SADNESS", "실망": "DISAPPOINTMENT",
    "심미적 감동": "AESTHETIC_APPRECIATION", "안도": "RELIEF",
    "어색함": "AWKWARDNESS", "연민": "SYMPATHY", "외로움": "LONELINESS",
    "의심": "DOUBT", "의지": "DETERMINATION", "자부심": "PRIDE",
    "죄책감": "GUILT", "즐거움": "AMUSEMENT", "질투": "ENVY",
    "집중": "CONCENTRATION", "짜증": "FRUSTRATION", "평온": "CALMNESS",
    "피로": "TIREDNESS", "향수": "NOSTALGIA", "혐오": "DISGUST",
    "혼란": "CONFUSION", "흥미": "INTEREST",
    "황홀": "ECSTASY",  # absent from the dump; taken from the taxonomy
}

# Minimum annotators for each gold tier (DESIGN.md §9).
TARGET_MIN = 3   # majority of five: what the reviewer must recover
ACCEPT_MIN = 2   # plausible enough that adding it is not a false add


@dataclass
class Reading:
    annotator: str
    major: str | None
    subs: set[str] = field(default_factory=set)


@dataclass
class Clip:
    clip_id: str
    stem: str                     # F0001_102994
    speaker: str                  # F0001
    batch: str
    readings: list[Reading] = field(default_factory=list)
    transcript: str = ""
    summary: str = ""
    prosody: str = ""
    detected: set[str] = field(default_factory=set)   # Gemini 1st pass
    unknown_kr: list[str] = field(default_factory=list)

    # --- gold tiers -----------------------------------------------------
    def sub_votes(self) -> Counter:
        votes: Counter = Counter()
        for r in self.readings:
            for s in r.subs:
                votes[s] += 1
        return votes

    def gold(self, min_votes: int) -> set[str]:
        return {s for s, c in self.sub_votes().items() if c >= min_votes}

    def gold_major(self, min_votes: int = 2) -> set[str]:
        votes: Counter = Counter(
            r.major for r in self.readings if r.major
        )
        return {m for m, c in votes.items() if c >= min_votes}


def _split(value: str) -> list[str]:
    return [t.strip() for t in (value or "").split(";") if t.strip()]


def _load_gemini(analysis_dir: Path) -> dict[str, dict]:
    """stem -> transcript / summary / prosody / clip-level emotion union.

    Gemini labels segments while the human gold is per clip, so segment
    emotions are unioned to make the two comparable.
    """
    out: dict[str, dict] = {}
    for path in sorted(analysis_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        emotions, notes = set(), []
        for seg in j.get("segments", []):
            for e in seg.get("emotions", []):
                name = (e.get("name") or "").strip().lower()
                if not name:
                    continue
                # Gemini emits the official codes in lowercase; the only name
                # that is not a plain uppercase of a code is "anxiety".
                emotions.add("ANXIETY_GENERAL" if name == "anxiety"
                             else name.upper())
            note = (seg.get("prosody_notes") or "").strip()
            if note:
                notes.append(note)
        out[path.stem] = {
            "transcript": (j.get("transcript") or "").strip(),
            "summary": (j.get("summary") or "").strip(),
            "prosody": " ".join(notes),
            "emotions": {c for c in emotions if schema.is_known_sub(c)},
        }
    return out


def load_batch(batch: str) -> list[Clip]:
    csv_path, analysis_dir = BATCHES[batch]
    gemini = _load_gemini(analysis_dir)

    clips: list[Clip] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = Path(row["storage_key"]).stem
            clip = Clip(clip_id=row["clip_id"], stem=stem,
                        speaker=stem.split("_")[0], batch=batch)
            for name in ANNOTATORS:
                krs = _split(row.get(f"{name}_sub_tags", ""))
                codes = set()
                for kr in krs:
                    code = SUB_KR2CODE.get(kr)
                    if code is None:
                        clip.unknown_kr.append(kr)
                    else:
                        codes.add(code)
                major_kr = (row.get(f"{name}_major") or "").strip()
                if not codes and not major_kr:
                    continue  # annotator did not label this clip
                clip.readings.append(Reading(
                    annotator=name,
                    major=MAJOR_KR2CODE.get(major_kr),
                    subs=codes,
                ))
            g = gemini.get(stem)
            if g:
                clip.transcript = g["transcript"]
                clip.summary = g["summary"]
                clip.prosody = g["prosody"]
                clip.detected = g["emotions"]
            clips.append(clip)
    return clips


# --- rows for the existing aggregation pipeline -------------------------------
def to_aggregate_rows(clips: list[Clip]) -> list[dict]:
    """Shape the clips the way aggregate.py already reads.

    Co-occurrence is counted from each annotator's individual reading rather
    than from the majority label: the majority collapses most clips to a single
    code, which destroys the co-occurrence signal the graph is built on
    (DESIGN.md, evolution note B).
    """
    rows: list[dict] = []
    for c in clips:
        rows.append({
            "clip_id": c.clip_id,
            "filename": f"{c.stem}.wav",      # speaker_of() splits on "_"
            "transcript": c.transcript,        # Gemini ASR, already clean
            "annotator_labels": json.dumps(
                {r.annotator: {"major_codes": [r.major] if r.major else [],
                               "sub_codes": sorted(r.subs)}
                 for r in c.readings},
                ensure_ascii=False,
            ),
            "human_majority_major_codes": json.dumps(
                sorted(c.gold_major()), ensure_ascii=False),
            "human_majority_sub_codes": json.dumps(
                sorted(c.gold(TARGET_MIN)), ensure_ascii=False),
        })
    return rows


def to_records(clips: list[Clip], tier: str = "accept"):
    """Evaluation records with the gold tier selected.

    target  majority of five (>=3) — what a reviewer must recover
    accept  >=2 annotators — adding one of these is not a false add
    union   any annotator — the widest reading of "someone saw this"
    """
    from dataset_pending import Record

    min_votes = {"target": TARGET_MIN, "accept": ACCEPT_MIN, "union": 1}[tier]
    return [
        Record(
            clip=c.stem, clip_id=c.clip_id, speaker=c.speaker,
            transcript=c.transcript, summary=c.summary, prosody=c.prosody,
            detected=set(c.detected), gold=c.gold(min_votes),
        )
        for c in clips
    ]


def report(clips: list[Clip]) -> dict:
    n = len(clips) or 1
    unknown = sorted({kr for c in clips for kr in c.unknown_kr})
    return {
        "clips": len(clips),
        "readings": sum(len(c.readings) for c in clips),
        "avg_subs_per_reading": round(
            sum(len(r.subs) for c in clips for r in c.readings)
            / max(1, sum(len(c.readings) for c in clips)), 2),
        "avg_gold_target(>=3)": round(
            sum(len(c.gold(TARGET_MIN)) for c in clips) / n, 2),
        "avg_gold_accept(>=2)": round(
            sum(len(c.gold(ACCEPT_MIN)) for c in clips) / n, 2),
        "avg_gold_union(>=1)": round(sum(len(c.gold(1)) for c in clips) / n, 2),
        "avg_gemini": round(sum(len(c.detected) for c in clips) / n, 2),
        "missing_transcript": sum(1 for c in clips if not c.transcript),
        "unknown_korean_labels": unknown,
    }


if __name__ == "__main__":
    for batch in BATCHES:
        clips = load_batch(batch)
        print(f"== {batch}")
        print(json.dumps(report(clips), ensure_ascii=False, indent=2))
