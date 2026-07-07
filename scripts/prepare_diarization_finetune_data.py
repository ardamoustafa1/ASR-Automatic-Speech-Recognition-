#!/usr/bin/env python3
"""
Turn a directory of RTTM reference files (built via scripts/annotate_diarization.py
or scripts/build_synthetic_diarization_benchmark.py) into a pyannote.database
custom protocol, ready for scripts/finetune_diarization.py.

pyannote.audio's fine-tuning API requires a registered "protocol" (train/
development file splits described by .lst/.rttm/.uem files + a database.yml
pointing at them) - this generates all of that automatically instead of
requiring a human to hand-write pyannote's database config format.

Usage:
  python3 scripts/prepare_diarization_finetune_data.py \\
      --audio-dir benchmarks/diarization_audio_real \\
      --rttm-dir benchmarks/diarization_rttm_real \\
      --output-dir data/diarization_finetune \\
      --dev-fraction 0.2
"""

import argparse
import random
import sys
import wave
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _audio_duration_sec(audio_path: Path) -> float:
    try:
        with wave.open(str(audio_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return info.frames / float(info.samplerate)


def _find_audio(audio_dir: Path, uri: str) -> Path | None:
    for ext in ("wav", "flac", "mp3", "m4a"):
        candidate = audio_dir / f"{uri}.{ext}"
        if candidate.exists():
            return candidate
    return None


def _write_split(
    uris: list[str], audio_dir: Path, rttm_dir: Path, output_dir: Path, split_name: str
) -> int:
    lst_lines, rttm_lines, uem_lines = [], [], []
    written = 0
    for uri in uris:
        audio_path = _find_audio(audio_dir, uri)
        rttm_path = rttm_dir / f"{uri}.rttm"
        if audio_path is None or not rttm_path.exists():
            continue
        duration = _audio_duration_sec(audio_path)
        lst_lines.append(uri)
        rttm_lines.append(rttm_path.read_text(encoding="utf-8").rstrip("\n"))
        uem_lines.append(f"{uri} 1 0.000 {duration:.3f}")
        written += 1

    (output_dir / f"{split_name}.lst").write_text("\n".join(lst_lines) + "\n", encoding="utf-8")
    (output_dir / f"{split_name}.rttm").write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")
    (output_dir / f"{split_name}.uem").write_text("\n".join(uem_lines) + "\n", encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a pyannote.database custom protocol for diarization fine-tuning."
    )
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--rttm-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/diarization_finetune")
    parser.add_argument(
        "--dev-fraction", type=float, default=0.2, help="Fraction of files held out for validation."
    )
    parser.add_argument("--protocol-name", type=str, default="ASRPro.SpeakerDiarization.Custom")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audio_dir = (WORKSPACE_ROOT / args.audio_dir).resolve()
    rttm_dir = (WORKSPACE_ROOT / args.rttm_dir).resolve()
    output_dir = (WORKSPACE_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    uris = sorted(p.stem for p in rttm_dir.glob("*.rttm"))
    if len(uris) < 2:
        print(
            f"Need at least 2 annotated files in {rttm_dir} to form train/dev splits (found {len(uris)})."
        )
        sys.exit(1)

    rng = random.Random(args.seed)  # noqa: S311
    rng.shuffle(uris)
    n_dev = max(1, round(len(uris) * args.dev_fraction))
    dev_uris, train_uris = uris[:n_dev], uris[n_dev:]
    if not train_uris:
        print(
            "Not enough files left for a training split after holding out dev - add more annotated calls."
        )
        sys.exit(1)

    n_train_written = _write_split(train_uris, audio_dir, rttm_dir, output_dir, "train")
    n_dev_written = _write_split(dev_uris, audio_dir, rttm_dir, output_dir, "development")

    db_name, task_name, protocol_short = args.protocol_name.split(".")
    database_yml = f"""\
Databases:
  {db_name}: {audio_dir}/{{uri}}.wav

Protocols:
  {db_name}:
    {task_name}:
      {protocol_short}:
        train:
          uri: {output_dir}/train.lst
          annotation: {output_dir}/train.rttm
          annotated: {output_dir}/train.uem
        development:
          uri: {output_dir}/development.lst
          annotation: {output_dir}/development.rttm
          annotated: {output_dir}/development.uem
"""
    (output_dir / "database.yml").write_text(database_yml, encoding="utf-8")

    print(f"Train split: {n_train_written} files -> {output_dir}/train.*")
    print(f"Dev split  : {n_dev_written} files -> {output_dir}/development.*")
    print(f"Protocol config written to {output_dir}/database.yml")
    print(
        f"\nNext step: python3 scripts/finetune_diarization.py "
        f"--database-yml {args.output_dir}/database.yml --protocol {args.protocol_name}"
    )


if __name__ == "__main__":
    main()
