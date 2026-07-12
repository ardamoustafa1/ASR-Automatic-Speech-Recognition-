#!/usr/bin/env python3
"""
Prepare Whisper fine-tuning data from your own call recordings (semi-supervised).

Real WER gains on 8kHz Turkish call-center audio come from fine-tuning on
in-domain data - but fine-tuning needs (audio, correct text) pairs. This
script builds them from unlabeled recordings with a confidence-split
pseudo-labeling workflow:

  1. Every call is transcribed with the current production pipeline
     (channel-isolated stereo, VAD gating, second-pass rescue).
  2. Each transcript segment is cut into its own 16kHz mono WAV clip
     (from the correct stereo channel, so the clip contains one speaker).
  3. Segments are split by decoder confidence:
       - avg_logprob >= --auto-label-threshold  -> train_auto.jsonl
         (pseudo-labels: the model is near-certain, safe to train on as-is)
       - below threshold                        -> review_queue.jsonl
         (a human corrects the prefilled text - these are the segments
          where fine-tuning helps most)
  4. After review, merge the corrected rows:
         python3 scripts/prepare_whisper_finetune_data.py \\
             --merge-reviewed data/whisper_finetune/review_queue.jsonl
     which appends human-verified rows to train_reviewed.jsonl.

Then train with scripts/finetune_whisper_lora.py.

Usage:
  python3 scripts/prepare_whisper_finetune_data.py \\
      --audio-dir sesler \\
      --output-dir data/whisper_finetune \\
      --max-files 0            # 0 = all files

Manifest row format (JSONL, one segment per line):
  {"audio": "clips/D101001_0007.wav", "text": "840 TL bir yıl boyunca.",
   "avg_logprob": -0.23, "duration": 2.1, "source": "D101001_...wav",
   "speaker": "SPEAKER_00", "review": "auto"|"pending"|"corrected"}

Resume-safe: already-processed source files are skipped on re-run.
"""

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

SAMPLE_RATE = 16000
# Segments shorter than this carry too little acoustic signal to train on;
# longer than 28s exceed Whisper's 30s training window (with prompt margin).
MIN_CLIP_SEC = 0.8
MAX_CLIP_SEC = 28.0


def _write_wav(path: Path, pcm: np.ndarray) -> None:
    pcm_i16 = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_i16.tobytes())


def _load_channels(audio_path: Path) -> dict[str, np.ndarray]:
    """Return per-speaker channel PCM: stereo files map SPEAKER_00/01 to L/R."""
    from faster_whisper import decode_audio

    from asr_pro.services.asr_service import ASRService

    if ASRService._is_stereo_file(str(audio_path)):
        left, right = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE, split_stereo=True)
        return {"SPEAKER_00": left, "SPEAKER_01": right, "__mono__": (left + right) / 2.0}
    mono = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    return {"__mono__": mono}


def process_files(args: argparse.Namespace) -> None:
    from asr_pro.services.asr_service import ASRService

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "processed_files.txt"
    processed = set(state_path.read_text().splitlines()) if state_path.exists() else set()

    auto_path = out_dir / "train_auto.jsonl"
    review_path = out_dir / "review_queue.jsonl"

    files = sorted(
        p for p in audio_dir.iterdir() if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")
    )
    if args.max_files:
        files = files[: args.max_files]

    svc = ASRService.get_instance()
    svc.load_model()

    n_auto = n_review = 0
    for file_idx, audio_path in enumerate(files):
        if audio_path.name in processed:
            continue
        print(f"[{file_idx + 1}/{len(files)}] {audio_path.name}")
        try:
            segments, _dur = svc.transcribe(str(audio_path), language="tr", sector=args.sector)
            channels = _load_channels(audio_path)
        except Exception as exc:
            print(f"  SKIP ({exc})")
            continue

        with (
            auto_path.open("a", encoding="utf-8") as f_auto,
            review_path.open("a", encoding="utf-8") as f_review,
        ):
            for seg_idx, seg in enumerate(segments):
                dur = seg.end - seg.start
                if not seg.text or dur < MIN_CLIP_SEC or dur > MAX_CLIP_SEC:
                    continue
                # IVR/system prompts aren't useful fine-tune targets.
                if seg.speaker and "IVR" in seg.speaker:
                    continue
                pcm = channels.get(seg.speaker or "__mono__", channels["__mono__"])
                clip = pcm[int(seg.start * SAMPLE_RATE) : int(seg.end * SAMPLE_RATE)]
                if clip.size < MIN_CLIP_SEC * SAMPLE_RATE:
                    continue
                clip_name = f"{audio_path.stem}_{seg_idx:04d}.wav"
                _write_wav(clips_dir / clip_name, clip)
                row = {
                    "audio": f"clips/{clip_name}",
                    "text": seg.text,
                    "avg_logprob": round(seg.avg_logprob, 3),
                    "duration": round(dur, 2),
                    "source": audio_path.name,
                    "speaker": seg.speaker,
                }
                if seg.avg_logprob != -1.0 and seg.avg_logprob >= args.auto_label_threshold:
                    row["review"] = "auto"
                    f_auto.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_auto += 1
                else:
                    row["review"] = "pending"
                    f_review.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_review += 1

        with state_path.open("a") as f:
            f.write(audio_path.name + "\n")

    print(
        f"\nDone. {n_auto} auto-labeled segments -> {auto_path}\n"
        f"      {n_review} segments queued for human review -> {review_path}\n"
        f'Next: correct the \'text\' fields in {review_path.name} (set "review": "corrected"),\n'
        f"then: python3 scripts/prepare_whisper_finetune_data.py --merge-reviewed {review_path}"
    )


def merge_reviewed(args: argparse.Namespace) -> None:
    reviewed_path = Path(args.merge_reviewed)
    out_dir = reviewed_path.parent
    merged_path = out_dir / "train_reviewed.jsonl"
    n = skipped = 0
    with (
        reviewed_path.open(encoding="utf-8") as f_in,
        merged_path.open("a", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("review") != "corrected" or not row.get("text", "").strip():
                skipped += 1
                continue
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(
        f"Merged {n} human-corrected rows into {merged_path} "
        f"({skipped} still pending/empty - left in queue)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--audio-dir", type=str, default="sesler")
    parser.add_argument("--output-dir", type=str, default="data/whisper_finetune")
    parser.add_argument("--sector", type=str, default="telecom")
    parser.add_argument("--max-files", type=int, default=0, help="0 = process all files")
    parser.add_argument(
        "--auto-label-threshold",
        type=float,
        default=-0.35,
        help="avg_logprob at or above this = trusted pseudo-label; below = human review queue.",
    )
    parser.add_argument(
        "--merge-reviewed",
        type=str,
        default="",
        help="Path to a corrected review_queue.jsonl; merges 'corrected' rows and exits.",
    )
    args = parser.parse_args()

    if args.merge_reviewed:
        merge_reviewed(args)
    else:
        process_files(args)


if __name__ == "__main__":
    main()
