#!/usr/bin/env python3
"""Measure REAL production behavior on real 8 kHz stereo call recordings.

IMPORTANT — this does NOT compute WER. WER requires human-written reference
transcripts (ground truth), which these calls do not have. What this DOES do is
run the full production ASR pipeline on real telephony audio and report the
operational metrics that are actually measurable without ground truth:
  - ASR decoder self-confidence (exp(avg_logprob)) distribution
  - Heuristic MOS-scale audio quality + SNR + dropout (the fixed estimator)
  - Real-time factor (processing speed)
  - Speaker count / segment counts
  - Full transcripts saved for human qualitative review

Usage:
  python3 scripts/measure_real_calls.py --dir sesler --sample 10 --out data/real_call_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import wave
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def wav_duration(path: str) -> float:
    try:
        with wave.open(path) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sesler")
    ap.add_argument(
        "--sample", type=int, default=10, help="How many calls to process (evenly spaced)."
    )
    ap.add_argument("--out", default="data/real_call_report.json")
    ap.add_argument("--transcripts-dir", default="data/real_transcripts")
    args = ap.parse_args()

    files = sorted(glob.glob(str(WORKSPACE_ROOT / args.dir / "*.wav")))
    if not files:
        print(f"No .wav files in {args.dir}")
        sys.exit(1)

    # Evenly-spaced sample across the whole set (not just the first N) so the
    # sample is representative, not biased to one time window.
    n = min(args.sample, len(files))
    step = max(1, len(files) // n)
    sample = files[::step][:n]

    from asr_pro.services.asr_service import ASRService
    from asr_pro.services.mos_estimator import MOSEstimator

    service = ASRService()

    tdir = WORKSPACE_ROOT / args.transcripts_dir
    tdir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"Processing {len(sample)} real calls (of {len(files)} total)...\n")
    for i, path in enumerate(sample, 1):
        name = os.path.basename(path)
        dur = wav_duration(path)
        t0 = time.perf_counter()
        try:
            segments, _ = service.transcribe(path, language="tr")
        except Exception as e:
            print(f"[{i}/{len(sample)}] {name[:40]} FAILED: {e}")
            rows.append({"file": name, "error": str(e)})
            continue
        elapsed = time.perf_counter() - t0

        confidence = ASRService.compute_confidence(segments)
        mos = MOSEstimator.estimate_mos(path)
        speakers = sorted({s.speaker for s in segments if getattr(s, "speaker", None)})
        full_text = "\n".join(
            f"[{s.start:6.1f}s] {getattr(s, 'speaker', '?') or '?'}: {s.text.strip()}"
            for s in segments
        )
        (tdir / f"{name}.txt").write_text(full_text, encoding="utf-8")

        row = {
            "file": name,
            "duration_s": round(dur, 1),
            "processing_s": round(elapsed, 1),
            "rtf": round(elapsed / dur, 3) if dur else None,
            "asr_confidence_pct": round(confidence * 100, 1),
            "mos_score": mos["mos_score"],
            "mos_grade": mos["quality_grade"],
            "snr_db": mos["snr_db"],
            "dropout_pct": mos["dropout_rate_pct"],
            "clipping_pct": mos["clipping_rate_pct"],
            "noc_alert": bool(mos["noc_alert"]),
            "n_segments": len(segments),
            "speakers": speakers,
        }
        rows.append(row)
        print(
            f"[{i}/{len(sample)}] {name[:36]} | conf %{row['asr_confidence_pct']} | "
            f"MOS {row['mos_score']} | SNR {row['snr_db']}dB | drop %{row['dropout_pct']} | "
            f"{row['n_segments']} seg | {len(speakers)} spk | RTF {row['rtf']}"
        )

    ok = [r for r in rows if "error" not in r]

    def _stats(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if not vals:
            return None
        vals.sort()
        return {
            "mean": round(sum(vals) / len(vals), 2),
            "median": vals[len(vals) // 2],
            "min": min(vals),
            "max": max(vals),
        }

    summary = {
        "total_calls_in_dir": len(files),
        "processed": len(ok),
        "sample_rate_hz": 8000,
        "confidence_pct": _stats("asr_confidence_pct"),
        "mos_score": _stats("mos_score"),
        "snr_db": _stats("snr_db"),
        "dropout_pct": _stats("dropout_pct"),
        "rtf": _stats("rtf"),
        "noc_alerts_fired": sum(1 for r in ok if r.get("noc_alert")),
        "note": "No ground-truth transcripts available -> WER NOT computed. "
        "These are operational metrics on real 8kHz telephony, plus transcripts for human review.",
    }

    out = {"summary": summary, "calls": rows}
    out_path = WORKSPACE_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 64)
    print("REAL-CALL OPERATIONAL SUMMARY (8kHz telephony, no ground truth)")
    print("=" * 64)
    print(f"Processed         : {len(ok)}/{len(sample)} calls")
    if summary["confidence_pct"]:
        c = summary["confidence_pct"]
        print(
            f"ASR confidence %  : mean {c['mean']} | median {c['median']} | min {c['min']} | max {c['max']}"
        )
    if summary["mos_score"]:
        m = summary["mos_score"]
        print(
            f"MOS (heuristic)   : mean {m['mean']} | median {m['median']} | min {m['min']} | max {m['max']}"
        )
    if summary["rtf"]:
        r = summary["rtf"]
        print(f"RTF (speed)       : mean {r['mean']} | median {r['median']}")
    print(f"False-ish NOC?    : {summary['noc_alerts_fired']} alerts across {len(ok)} calls")
    print(f"Transcripts saved : {tdir}")
    print(f"JSON report       : {out_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
