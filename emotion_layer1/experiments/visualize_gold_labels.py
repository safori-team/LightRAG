"""Generate an HTML report for multi-annotator gold label inspection."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean


def _labels(record: dict, key: str = "minor") -> set[str]:
    return {e[key] for e in record["gold_labels"].get("emotions", [])}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def _bar(value: float, max_value: float, label: str = "") -> str:
    width = 0 if max_value <= 0 else max(2, int(value / max_value * 100))
    text = html.escape(label or str(value))
    return (
        f'<div class="bar-row"><div class="bar" style="width:{width}%"></div>'
        f'<span>{text}</span></div>'
    )


def _render_matrix(annotators: list[int], values: dict[tuple[int, int], float]) -> str:
    rows = []
    header = "<tr><th></th>" + "".join(f"<th>A{a}</th>" for a in annotators) + "</tr>"
    for a in annotators:
        cells = [f"<th>A{a}</th>"]
        for b in annotators:
            if a == b:
                cells.append('<td class="diag">1.00</td>')
                continue
            v = values.get(tuple(sorted((a, b))), 0.0)
            shade = int(255 - v * 110)
            cells.append(
                f'<td style="background:rgb({shade},{shade},255)">{v:.2f}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<table class="matrix">' + header + "".join(rows) + "</table>"


def build_report(records: list[dict]) -> tuple[str, dict]:
    by_annotator: dict[int, list[dict]] = defaultdict(list)
    by_clip: dict[int, list[dict]] = defaultdict(list)
    major_by_annotator: dict[int, Counter] = defaultdict(Counter)
    minor_by_annotator: dict[int, Counter] = defaultdict(Counter)
    label_count_by_annotator: dict[int, list[int]] = defaultdict(list)

    for record in records:
        annotator = record["annotator_id"]
        by_annotator[annotator].append(record)
        by_clip[record["clip_id"]].append(record)
        emotions = record["gold_labels"].get("emotions", [])
        label_count_by_annotator[annotator].append(len(emotions))
        for emotion in emotions:
            major_by_annotator[annotator][emotion["major"]] += 1
            minor_by_annotator[annotator][emotion["minor"]] += 1

    annotators = sorted(by_annotator)
    all_minors = Counter()
    all_majors = Counter()
    for counter in minor_by_annotator.values():
        all_minors.update(counter)
    for counter in major_by_annotator.values():
        all_majors.update(counter)

    pair_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    clip_summaries = []
    for clip_id, clip_records in by_clip.items():
        rec_by_a = {r["annotator_id"]: r for r in clip_records}
        labels_by_a = {a: _labels(r) for a, r in rec_by_a.items()}
        scores = []
        for a, b in combinations(sorted(labels_by_a), 2):
            score = _jaccard(labels_by_a[a], labels_by_a[b])
            pair_scores[(a, b)].append(score)
            scores.append(score)
        union = set().union(*labels_by_a.values()) if labels_by_a else set()
        vote = Counter(label for labels in labels_by_a.values() for label in labels)
        clip_summaries.append({
            "clip_id": clip_id,
            "storage_key": clip_records[0].get("storage_key"),
            "transcript": clip_records[0].get("transcript", ""),
            "avg_jaccard": mean(scores) if scores else 0.0,
            "union_size": len(union),
            "majority_labels": [k for k, v in vote.items() if v >= 3],
            "labels_by_annotator": {
                str(a): sorted(labels) for a, labels in labels_by_a.items()
            },
        })

    avg_pair = {pair: mean(scores) for pair, scores in pair_scores.items()}
    worst_clips = sorted(clip_summaries, key=lambda item: item["avg_jaccard"])[:12]

    metrics = {
        "records": len(records),
        "clips": len(by_clip),
        "annotators": annotators,
        "records_by_annotator": {str(k): len(v) for k, v in by_annotator.items()},
        "avg_labels_by_annotator": {
            str(k): round(mean(v), 3) for k, v in label_count_by_annotator.items()
        },
        "top_major_labels": all_majors.most_common(),
        "top_minor_labels": all_minors.most_common(),
        "pairwise_minor_jaccard": {
            f"{a}-{b}": round(v, 4) for (a, b), v in avg_pair.items()
        },
        "avg_pairwise_minor_jaccard": round(mean(avg_pair.values()), 4)
        if avg_pair
        else 0.0,
        "lowest_agreement_clips": [
            {
                "clip_id": item["clip_id"],
                "avg_jaccard": round(item["avg_jaccard"], 4),
                "union_size": item["union_size"],
                "majority_labels": item["majority_labels"],
            }
            for item in worst_clips
        ],
    }

    max_records = max(len(v) for v in by_annotator.values())
    max_major = max(all_majors.values()) if all_majors else 1
    max_minor = max(all_minors.values()) if all_minors else 1

    rows = []
    for annotator in annotators:
        rows.append(
            "<tr>"
            f"<td>A{annotator}</td>"
            f"<td>{len(by_annotator[annotator])}</td>"
            f"<td>{mean(label_count_by_annotator[annotator]):.2f}</td>"
            f"<td>{html.escape(', '.join(k for k, _ in minor_by_annotator[annotator].most_common(8)))}</td>"
            "</tr>"
        )

    major_rows = []
    for label, total in all_majors.most_common():
        cells = [f"<td>{html.escape(label)}</td>", f"<td>{total}</td>"]
        for annotator in annotators:
            cells.append(f"<td>{major_by_annotator[annotator][label]}</td>")
        cells.append(f"<td>{_bar(total, max_major, str(total))}</td>")
        major_rows.append("<tr>" + "".join(cells) + "</tr>")

    minor_rows = []
    for label, total in all_minors.most_common(30):
        cells = [f"<td>{html.escape(label)}</td>", f"<td>{total}</td>"]
        for annotator in annotators:
            cells.append(f"<td>{minor_by_annotator[annotator][label]}</td>")
        cells.append(f"<td>{_bar(total, max_minor, str(total))}</td>")
        minor_rows.append("<tr>" + "".join(cells) + "</tr>")

    disagreement_rows = []
    for item in worst_clips:
        label_lines = []
        for annotator in annotators:
            labels = item["labels_by_annotator"].get(str(annotator), [])
            label_lines.append(
                f"<div><b>A{annotator}</b>: {html.escape(', '.join(labels) or '-')}</div>"
            )
        disagreement_rows.append(
            "<tr>"
            f"<td>{item['clip_id']}<br><small>{html.escape(str(item['storage_key']))}</small></td>"
            f"<td>{item['avg_jaccard']:.2f}</td>"
            f"<td>{item['union_size']}</td>"
            f"<td>{html.escape(item['transcript'])}</td>"
            f"<td>{''.join(label_lines)}</td>"
            "</tr>"
        )

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Gold Labeling Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #202124; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 7px 8px; vertical-align: top; }}
    th {{ background: #f4f6f8; text-align: left; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .num {{ font-size: 24px; font-weight: 700; }}
    .bar-row {{ position: relative; height: 22px; background: #f1f3f4; min-width: 120px; }}
    .bar {{ position: absolute; inset: 0 auto 0 0; background: #7aa7ff; }}
    .bar-row span {{ position: relative; display: block; padding: 2px 6px; }}
    .matrix td, .matrix th {{ text-align: center; }}
    .diag {{ background: #eee; color: #777; }}
    small {{ color: #666; }}
  </style>
</head>
<body>
  <h1>Gold Labeling Report</h1>
  <p><small>Generated from gold_text/with_transcripts JSONL. A3/A4/A5/A6/A7 are annotator IDs.</small></p>

  <section class="cards">
    <div class="card"><div>Records</div><div class="num">{len(records)}</div></div>
    <div class="card"><div>Unique Clips</div><div class="num">{len(by_clip)}</div></div>
    <div class="card"><div>Annotators</div><div class="num">{len(annotators)}</div></div>
    <div class="card"><div>Avg Pairwise Jaccard</div><div class="num">{metrics['avg_pairwise_minor_jaccard']:.2f}</div></div>
  </section>

  <section>
    <h2>Annotator Summary</h2>
    <table>
      <tr><th>Annotator</th><th>Records</th><th>Avg Labels / Record</th><th>Top Minor Labels</th></tr>
      {''.join(rows)}
    </table>
  </section>

  <section>
    <h2>Pairwise Agreement: Minor Label Jaccard</h2>
    {_render_matrix(annotators, avg_pair)}
  </section>

  <section>
    <h2>Major Label Distribution</h2>
    <table>
      <tr><th>Major</th><th>Total</th>{''.join(f'<th>A{a}</th>' for a in annotators)}<th>Bar</th></tr>
      {''.join(major_rows)}
    </table>
  </section>

  <section>
    <h2>Top 30 Minor Label Distribution</h2>
    <table>
      <tr><th>Minor</th><th>Total</th>{''.join(f'<th>A{a}</th>' for a in annotators)}<th>Bar</th></tr>
      {''.join(minor_rows)}
    </table>
  </section>

  <section>
    <h2>Lowest Agreement Clips</h2>
    <table>
      <tr><th>Clip</th><th>Avg Jaccard</th><th>Union Size</th><th>Transcript</th><th>Labels by Annotator</th></tr>
      {''.join(disagreement_rows)}
    </table>
  </section>
</body>
</html>
"""
    return html_doc, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("out_html", type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    html_doc, metrics = build_report(records)

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html_doc, encoding="utf-8")
    if args.metrics:
        args.metrics.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"wrote {args.out_html}")
    if args.metrics:
        print(f"wrote {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

