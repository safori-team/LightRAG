"""Create a deterministic 60/20/20 split without source-prefix leakage.

The prefix in storage keys such as ``clips/F2005/F2005_000091.wav`` is not a
verified speaker ID.  It is treated only as a source/recording group so related
clips cannot cross train, validation, and test boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import labels_kr

HERE = Path(__file__).resolve().parent
WEEKLY = Path(
    "/Users/hann/Project/SAFORI/labeled-data/exports/"
    "emotion_relabeling/weekly"
)
DEFAULT_INPUTS = (
    (
        "w30_legacy",
        WEEKLY / "2026-W30/snapshot_2026-07-26/manifests/"
        "legacy_labeled_100_by_annotator.csv",
    ),
    (
        "w30_pending",
        WEEKLY / "2026-W30/snapshot_2026-07-26/manifests/"
        "team_reference_pending_100_by_annotator.csv",
    ),
    (
        "w31_pending",
        WEEKLY / "2026-W31/snapshot_2026-08-02/manifests/"
        "team_reference_pending_100_by_annotator.csv",
    ),
)
SPLITS = ("train", "validation", "test")
TARGETS = {"train": 180, "validation": 60, "test": 60}


def source_prefix(storage_key: str) -> str:
    return Path(storage_key).stem.split("_", 1)[0]


def accepted_labels(row: dict) -> set[str]:
    votes = Counter()
    for annotator in labels_kr.ANNOTATORS:
        values = (row.get(f"{annotator}_sub_tags") or "").split(";")
        for value in {item.strip() for item in values if item.strip()}:
            code = labels_kr.SUB_KR2CODE.get(value)
            if code:
                votes[code] += 1
    return {code for code, count in votes.items() if count >= 2}


def load_rows():
    rows = []
    fieldnames = None
    for batch, path in DEFAULT_INPUTS:
        with path.open(encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"manifest columns differ: {path}")
            for row in reader:
                row["source_batch"] = batch
                row["source_manifest"] = str(path)
                row["source_prefix"] = source_prefix(row["storage_key"])
                row["accepted_labels"] = sorted(accepted_labels(row))
                rows.append(row)
    if len(rows) != 300:
        raise ValueError(f"expected 300 rows, got {len(rows)}")
    clip_ids = [row["clip_id"] for row in rows]
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("clip IDs overlap across the three source manifests")
    return rows, fieldnames or []


def score_assignment(groups, assignment, global_labels, global_batches):
    split_labels = {split: Counter() for split in SPLITS}
    split_batches = {split: Counter() for split in SPLITS}
    for prefix, rows in groups.items():
        split = assignment[prefix]
        for row in rows:
            split_labels[split].update(row["accepted_labels"])
            split_batches[split][row["source_batch"]] += 1

    score = 0.0
    total = sum(TARGETS.values())
    for split in SPLITS:
        ratio = TARGETS[split] / total
        for label, count in global_labels.items():
            expected = count * ratio
            score += ((split_labels[split][label] - expected) / max(1.0, expected)) ** 2
            # When a label occurs at least three times, strongly prefer a split
            # that exposes it to train, validation, and test. Group constraints
            # may make this impossible, so this remains an objective, not a hard
            # validity rule.
            if count >= 3 and split_labels[split][label] == 0:
                score += 5.0
        for batch, count in global_batches.items():
            expected = count * ratio
            score += 2.0 * (
                (split_batches[split][batch] - expected) / max(1.0, expected)
            ) ** 2
    return score


def find_assignment(rows, trials: int, seed: int):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_prefix"]].append(row)
    global_labels = Counter(
        label for row in rows for label in row["accepted_labels"]
    )
    global_batches = Counter(row["source_batch"] for row in rows)
    rng = random.Random(seed)
    best = None

    for _ in range(trials):
        # Larger groups go first; random tie-breaking creates alternative valid
        # assignments while singleton groups make exact target sizes possible.
        prefixes = list(groups)
        rng.shuffle(prefixes)
        prefixes.sort(key=lambda prefix: len(groups[prefix]), reverse=True)
        remaining = dict(TARGETS)
        assignment = {}
        valid = True
        for prefix in prefixes:
            size = len(groups[prefix])
            choices = [split for split in SPLITS if remaining[split] >= size]
            if not choices:
                valid = False
                break
            weights = [max(1, remaining[split]) ** 2 for split in choices]
            split = rng.choices(choices, weights=weights, k=1)[0]
            assignment[prefix] = split
            remaining[split] -= size
        if not valid or any(remaining.values()):
            continue
        score = score_assignment(groups, assignment, global_labels, global_batches)
        if best is None or score < best[0]:
            best = (score, assignment)

    if best is None:
        raise RuntimeError("could not find an exact group assignment")
    return groups, best[1], best[0]


def summary(rows, assignment, objective):
    result = {
        "method": "source-prefix group split (not verified speaker IDs)",
        "targets": TARGETS,
        "objective": round(objective, 6),
        "total_records": len(rows),
        "total_source_prefixes": len(assignment),
        "splits": {},
        "leakage": {},
    }
    prefix_sets = {}
    for split in SPLITS:
        selected = [row for row in rows if assignment[row["source_prefix"]] == split]
        prefixes = {row["source_prefix"] for row in selected}
        prefix_sets[split] = prefixes
        labels = Counter(label for row in selected for label in row["accepted_labels"])
        result["splits"][split] = {
            "records": len(selected),
            "source_prefixes": len(prefixes),
            "source_batches": dict(sorted(Counter(
                row["source_batch"] for row in selected
            ).items())),
            "accepted_label_counts": dict(sorted(labels.items())),
        }
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            result["leakage"][f"{left}__{right}_prefix_overlap"] = sorted(
                prefix_sets[left] & prefix_sets[right]
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--out-dir", type=Path,
        default=HERE / "results/source_prefix_split_300",
    )
    args = parser.parse_args()
    rows, original_fields = load_rows()
    _, assignment, objective = find_assignment(rows, args.trials, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_fields = [
        "split", "source_batch", "source_prefix", "source_manifest",
        *original_fields,
    ]
    combined = []
    for row in rows:
        saved = {key: value for key, value in row.items() if key != "accepted_labels"}
        saved["split"] = assignment[row["source_prefix"]]
        combined.append(saved)
    combined.sort(key=lambda row: (SPLITS.index(row["split"]), row["storage_key"]))

    for name, selected in (
        ("all_300.csv", combined),
        *((f"{split}.csv", [row for row in combined if row["split"] == split])
          for split in SPLITS),
    ):
        with (args.out_dir / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(selected)

    report = summary(rows, assignment, objective)
    report["seed"] = args.seed
    report["trials"] = args.trials
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "prefix_assignment.json").write_text(
        json.dumps(dict(sorted(assignment.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
