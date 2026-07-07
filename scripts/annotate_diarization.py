#!/usr/bin/env python3
"""
Human annotation -> RTTM converter for diarization ground truth.

A human reviewer should not have to learn the RTTM column format to produce
reference data. This script lets them write a plain CSV instead - one row per
speaker turn, three columns - and converts it to RTTM for
scripts/evaluate_diarization.py / asr_pro/services/diarization_eval.py.

Annotation workflow for a real production call:
  1. Listen to <call_id>.wav in any audio player with a visible timestamp
     (e.g. Audacity, VLC, QuickTime).
  2. For every speaker turn, write one row to <call_id>.csv:
         start_sec,end_sec,speaker
         0.0,3.2,agent
         3.4,7.9,customer
         8.1,8.6,agent
     Speaker labels can be anything consistent within the file (agent/
     customer, or SPEAKER_00/SPEAKER_01) - only the partition into turns
     matters for DER, not the label spelling.
  3. Run:
         python3 scripts/annotate_diarization.py --csv <call_id>.csv \\
             --output benchmarks/diarization_rttm/<call_id>.rttm --uri <call_id>

Repeat for ~20-30 real, representative calls before trusting an aggregate DER
number - a handful of files gives a directional signal, not a stable metric.
"""

import argparse
import csv
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def csv_to_rttm(csv_path: Path, uri: str) -> list[str]:
    """Convert a start_sec,end_sec,speaker CSV into RTTM lines."""
    rows: list[tuple[float, float, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"start_sec", "end_sec", "speaker"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{csv_path} must have header columns: start_sec,end_sec,speaker "
                f"(found: {reader.fieldnames})"
            )
        for line_no, row in enumerate(reader, start=2):
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path}:{line_no}: non-numeric start/end ({row})") from exc
            speaker = (row["speaker"] or "").strip()
            if not speaker:
                raise ValueError(f"{csv_path}:{line_no}: empty speaker label")
            if end <= start:
                raise ValueError(f"{csv_path}:{line_no}: end ({end}) must be after start ({start})")
            rows.append((start, end, speaker))

    if not rows:
        raise ValueError(f"{csv_path} has no annotation rows")

    rows.sort(key=lambda r: r[0])
    lines = []
    for start, end, speaker in rows:
        duration = end - start
        # RTTM: SPEAKER <uri> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
        lines.append(f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a speaker-turn CSV annotation to RTTM.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the annotation CSV.")
    parser.add_argument("--output", type=str, required=True, help="Path to write the .rttm file.")
    parser.add_argument(
        "--uri", type=str, default=None, help="Recording URI (defaults to the CSV file's stem)."
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Annotation CSV not found: {csv_path}")
        sys.exit(1)

    uri = args.uri or csv_path.stem
    lines = csv_to_rttm(csv_path, uri)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} speaker turns to {output_path} (uri={uri})")


if __name__ == "__main__":
    main()
