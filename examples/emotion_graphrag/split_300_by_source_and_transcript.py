"""Group split by source prefix plus exact/near-duplicate transcripts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from split300_data import load_split
from split_300_by_source_prefix import (
    SPLITS, TARGETS, find_assignment, summary,
)

HERE = Path(__file__).resolve().parent
OLD_SPLIT = HERE / "results/source_prefix_split_300"


class DSU:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", text).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity", type=float, default=0.95)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--min-length-ratio", type=float, default=0.8)
    parser.add_argument("--trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument(
        "--out-dir", type=Path,
        default=HERE / "results/source_prefix_transcript_split_300",
    )
    args = parser.parse_args()

    with (OLD_SPLIT / "all_300.csv").open(encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        original_fields = list(reader.fieldnames or [])
        rows = list(reader)
    records = {
        record.clip_id: record
        for split in SPLITS for record in load_split(split, OLD_SPLIT)
    }
    for row in rows:
        row["transcript"] = records[row["clip_id"]].transcript
        row["accepted_labels"] = sorted(records[row["clip_id"]].gold())

    dsu = DSU(len(rows))
    by_prefix = defaultdict(list)
    for index, row in enumerate(rows):
        by_prefix[row["source_prefix"]].append(index)
    for indexes in by_prefix.values():
        for index in indexes[1:]:
            dsu.union(indexes[0], index)

    transcript_edges = []
    normalized = [normalize(row["transcript"]) for row in rows]
    for left, left_text in enumerate(normalized):
        for right in range(left + 1, len(rows)):
            right_text = normalized[right]
            if min(len(left_text), len(right_text)) < args.min_length:
                continue
            if min(len(left_text), len(right_text)) / max(len(left_text), len(right_text)) < args.min_length_ratio:
                continue
            similarity = SequenceMatcher(
                None, left_text, right_text, autojunk=False
            ).ratio()
            if similarity >= args.similarity:
                dsu.union(left, right)
                transcript_edges.append({
                    "left_clip_id": rows[left]["clip_id"],
                    "right_clip_id": rows[right]["clip_id"],
                    "similarity": round(similarity, 6),
                    "exact_normalized": left_text == right_text,
                    "left_transcript": rows[left]["transcript"],
                    "right_transcript": rows[right]["transcript"],
                })

    roots = {dsu.find(index) for index in range(len(rows))}
    component_names = {root: f"component_{rank:03d}" for rank, root in enumerate(sorted(roots), 1)}
    for index, row in enumerate(rows):
        row["component_id"] = component_names[dsu.find(index)]
    groups = defaultdict(list)
    for row in rows:
        groups[row["component_id"]].append(row)

    # find_assignment only depends on source_prefix as the group key. Use the
    # component ID in that slot so connected components remain indivisible.
    optimizer_rows = []
    for row in rows:
        copied = dict(row)
        copied["source_prefix"] = row["component_id"]
        optimizer_rows.append(copied)
    _, assignment, objective = find_assignment(
        optimizer_rows, args.trials, args.seed
    )
    for row in rows:
        row["split"] = assignment[row["component_id"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_fields = [
        "split", "component_id", "transcript", *[
            field for field in original_fields if field not in {"split"}
        ]
    ]
    saved_rows = [{key: value for key, value in row.items()
                   if key != "accepted_labels"} for row in rows]
    saved_rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["storage_key"]))
    for name, selected in (
        ("all_300.csv", saved_rows),
        *((f"{split}.csv", [row for row in saved_rows if row["split"] == split])
          for split in SPLITS),
    ):
        with (args.out_dir / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(selected)

    # summary() expects assignment keyed by the row source_prefix. Feed the
    # optimizer view, then append the transcript-specific audit.
    report = summary(optimizer_rows, assignment, objective)
    report.update({
        "method": "connected components of source prefix + transcript similarity",
        "transcript_rule": {
            "normalized_similarity": args.similarity,
            "min_normalized_length": args.min_length,
            "min_length_ratio": args.min_length_ratio,
        },
        "total_source_prefixes": len({row["source_prefix"] for row in rows}),
        "components": len(groups),
        "largest_component": max(len(group) for group in groups.values()),
        "transcript_edges": len(transcript_edges),
        "exact_normalized_edges": sum(edge["exact_normalized"] for edge in transcript_edges),
        "near_duplicate_edges": sum(not edge["exact_normalized"] for edge in transcript_edges),
        "seed": args.seed,
        "trials": args.trials,
    })
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        report["splits"][split]["source_prefixes"] = len({
            row["source_prefix"] for row in selected
        })
        report["splits"][split]["components"] = len({
            row["component_id"] for row in selected
        })
    # Replace component-key overlap audit with checks over both grouping axes.
    split_prefixes = {
        split: {row["source_prefix"] for row in rows if row["split"] == split}
        for split in SPLITS
    }
    split_components = {
        split: {row["component_id"] for row in rows if row["split"] == split}
        for split in SPLITS
    }
    report["leakage"] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            report["leakage"][f"{left}__{right}_source_prefix_overlap"] = sorted(
                split_prefixes[left] & split_prefixes[right]
            )
            report["leakage"][f"{left}__{right}_component_overlap"] = sorted(
                split_components[left] & split_components[right]
            )
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "transcript_edges.json").write_text(
        json.dumps(transcript_edges, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
