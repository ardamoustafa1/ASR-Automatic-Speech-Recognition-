#!/usr/bin/env python3
"""
Build a synthetic 2-speaker diarization benchmark from the existing single-speaker
WER benchmark clips (benchmarks/audio/*.wav), with exact ground-truth RTTM.

IMPORTANT - what this is and isn't:
  This concatenates real recorded voices (not sine waves) into synthetic
  two-party "calls" with turns alternating strictly, separated by silence -
  there is no real crosstalk, channel bleed, or natural conversational
  overlap in it. It is an honest smoke test that the full pipeline (real
  audio -> DiarizationService.diarize -> DER scoring) works end-to-end and
  gives a rough sanity signal, NOT a substitute for DER measured on real
  production call recordings (see scripts/annotate_diarization.py for that).
  Report any DER from this benchmark as "synthetic smoke-test DER", never as
  a production accuracy claim.

  All 15 source clips in benchmarks/audio/ are the SAME single narrator
  (verified via ECAPA-TDNN cosine similarity ~0.90-0.91 between any two
  clips - the same range as a genuine same-speaker match). Alternating them
  as "agent"/"customer" without alteration would build a benchmark with only
  one real voice in it, and a diarization system correctly reporting "1
  speaker" on that would score ~50% DER against a 2-speaker ground truth -
  that is a benchmark defect, not a diarization failure. To get genuine
  acoustic speaker separation for the "customer" role, this script applies a
  resample-based pitch/tempo shift (a classic, dependency-free semitone
  shift: resampling to length/2**(semitones/12) before playback at the
  original rate) - verified via ECAPA-TDNN to bring cross-speaker cosine
  similarity down to ~0.15-0.30 (well below the ~0.90 same-speaker
  baseline), a level a real diarization system should resolve as two
  distinct speakers.

Usage:
  python3 scripts/build_synthetic_diarization_benchmark.py \\
      --audio-dir benchmarks/audio \\
      --output-audio-dir benchmarks/diarization_audio \\
      --output-rttm-dir benchmarks/diarization_rttm \\
      --num-calls 5
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

SAMPLE_RATE = 16000
GAP_SEC_MIN = 0.3
GAP_SEC_MAX = 0.9
# Verified via ECAPA-TDNN cosine similarity (see module docstring) to create
# genuine acoustic speaker separation from a single-narrator source.
CUSTOMER_PITCH_SHIFT_SEMITONES = -6.0


def _pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """Dependency-free semitone shift via resampling (changes tempo too, which is fine here)."""
    factor = 2.0 ** (semitones / 12.0)
    n_new = max(1, int(len(audio) / factor))
    return resample(audio, n_new).astype(np.float32)


def build_synthetic_call(
    clip_paths: list[Path], rng: random.Random
) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """Concatenate clips alternating agent/customer with silence gaps.

    Returns (audio, turns) where turns is [(start, end, speaker), ...] - the
    exact ground truth, since we control every gap and clip boundary.
    """
    audio_chunks: list[np.ndarray] = []
    turns: list[tuple[float, float, str]] = []
    cursor = 0.0

    for idx, clip_path in enumerate(clip_paths):
        speaker = "agent" if idx % 2 == 0 else "customer"
        data, sr = sf.read(str(clip_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{clip_path} is {sr}Hz, expected {SAMPLE_RATE}Hz")

        if speaker == "customer":
            data = _pitch_shift(data, CUSTOMER_PITCH_SHIFT_SEMITONES)

        duration = len(data) / SAMPLE_RATE
        audio_chunks.append(data)
        turns.append((round(cursor, 3), round(cursor + duration, 3), speaker))
        cursor += duration

        if idx < len(clip_paths) - 1:
            gap = rng.uniform(GAP_SEC_MIN, GAP_SEC_MAX)
            audio_chunks.append(np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32))
            cursor += gap

    return np.concatenate(audio_chunks), turns


def write_rttm(turns: list[tuple[float, float, str]], uri: str, output_path: Path) -> None:
    lines = [
        f"SPEAKER {uri} 1 {start:.3f} {end - start:.3f} <NA> <NA> {speaker} <NA> <NA>"
        for start, end, speaker in turns
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a synthetic 2-speaker diarization benchmark."
    )
    parser.add_argument("--audio-dir", type=str, default="benchmarks/audio")
    parser.add_argument("--output-audio-dir", type=str, default="benchmarks/diarization_audio")
    parser.add_argument("--output-rttm-dir", type=str, default="benchmarks/diarization_rttm")
    parser.add_argument(
        "--num-calls", type=int, default=5, help="Number of synthetic calls to build."
    )
    parser.add_argument(
        "--clips-per-call", type=int, default=4, help="Number of alternating turns per call."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)  # noqa: S311
    audio_dir = WORKSPACE_ROOT / args.audio_dir
    clip_paths = sorted(audio_dir.glob("*.wav"))
    if len(clip_paths) < args.clips_per_call:
        print(f"Not enough source clips in {audio_dir} (found {len(clip_paths)}).")
        sys.exit(1)

    output_audio_dir = WORKSPACE_ROOT / args.output_audio_dir
    output_rttm_dir = WORKSPACE_ROOT / args.output_rttm_dir
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_rttm_dir.mkdir(parents=True, exist_ok=True)

    built = 0
    for call_idx in range(args.num_calls):
        selected = rng.sample(clip_paths, args.clips_per_call)
        uri = f"synthetic_call_{call_idx:02d}"

        audio, turns = build_synthetic_call(selected, rng)
        sf.write(str(output_audio_dir / f"{uri}.wav"), audio, SAMPLE_RATE)
        write_rttm(turns, uri, output_rttm_dir / f"{uri}.rttm")
        built += 1
        print(f"Built {uri}: {len(selected)} turns, {len(audio) / SAMPLE_RATE:.1f}s")

    print(f"\nDone. {built} synthetic calls written to {output_audio_dir} / {output_rttm_dir}.")
    print(
        "Run: python3 scripts/evaluate_diarization.py "
        f"--audio-dir {args.output_audio_dir} --rttm-dir {args.output_rttm_dir}"
    )


if __name__ == "__main__":
    main()
