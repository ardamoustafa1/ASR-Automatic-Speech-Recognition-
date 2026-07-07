#!/usr/bin/env python3
"""
ASR-PRO Diarization Error Rate (DER) Evaluation Pipeline.

Scores DiarizationService's speaker turns against human-annotated RTTM
reference files using pyannote.metrics.DiarizationErrorRate - the same metric
used to report published diarization benchmark results, so numbers here are
directly comparable to pyannote's own published DER figures.

Without this, "diarization got more accurate" cannot be verified - only
measured against real call recordings from the target deployment.

Usage:
  python3 scripts/evaluate_diarization.py --audio-dir benchmarks/diarization_audio \\
      --rttm-dir benchmarks/diarization_rttm [--max-der 0.25] [--output-json data/der_report.json]

Expects <name>.wav (or .mp3/.flac/.m4a) in --audio-dir paired with a matching
<name>.rttm reference in --rttm-dir.
"""

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR-PRO DER benchmark suite.")
    parser.add_argument(
        "--audio-dir", type=str, required=True, help="Directory containing reference call audio."
    )
    parser.add_argument(
        "--rttm-dir",
        type=str,
        required=True,
        help="Directory containing matching .rttm ground truth.",
    )
    parser.add_argument(
        "--output-json", type=str, default="data/der_report.json", help="Path to save JSON report."
    )
    parser.add_argument(
        "--max-der",
        type=float,
        default=None,
        help="Maximum acceptable overall DER (0.0-1.0+). Fails with exit code 1 if exceeded.",
    )
    args = parser.parse_args()

    from asr_pro.services.diarization_eval import evaluate_directory

    print(f"Evaluating diarization: audio={args.audio_dir} rttm={args.rttm_dir}")
    result = evaluate_directory(args.audio_dir, args.rttm_dir)

    if not result.per_file:
        print("No audio/RTTM pairs were evaluated - check --audio-dir/--rttm-dir contents.")
        sys.exit(1)

    report = {
        "overall_der": round(result.overall_der, 4),
        "mean_der": round(result.mean_der, 4),
        "files_evaluated": len(result.per_file),
        "per_file": [
            {
                "uri": r.uri,
                "der": round(r.der, 4),
                "reference_speakers": r.reference_speakers,
                "hypothesis_speakers": r.hypothesis_speakers,
                "components": {k: round(v, 4) for k, v in r.components.items()},
            }
            for r in result.per_file
        ],
    }

    output_json_path = WORKSPACE_ROOT / args.output_json
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"JSON report saved at: {output_json_path}")

    print("\n" + "=" * 60)
    print("ASR-PRO DIARIZATION ERROR RATE (DER) SUMMARY")
    print("=" * 60)
    print(f"Files evaluated       : {report['files_evaluated']}")
    print(f"Overall DER (weighted): {report['overall_der'] * 100:.2f}%")
    print(f"Mean per-file DER     : {report['mean_der'] * 100:.2f}%")
    for r in report["per_file"]:
        print(
            f"  - {r['uri']}: DER={r['der'] * 100:.2f}% (ref_spk={r['reference_speakers']}, hyp_spk={r['hypothesis_speakers']})"
        )
    print("=" * 60)

    if args.max_der is not None:
        if report["overall_der"] > args.max_der:
            print(
                f"REGRESSION FAILED: Overall DER ({report['overall_der']:.4f}) exceeds "
                f"maximum allowed threshold ({args.max_der:.4f})."
            )
            sys.exit(1)
        else:
            print(
                f"REGRESSION PASSED: Overall DER ({report['overall_der']:.4f}) is within "
                f"threshold ({args.max_der:.4f})."
            )


if __name__ == "__main__":
    main()
