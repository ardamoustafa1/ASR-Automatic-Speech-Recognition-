# ASR-Pro Performance Benchmarks

> **2026-07-11 correction:** the numbers previously in this file (RTF ~0.015
> on Apple Silicon, ~0.008 on an RTX 4090) were never produced by an actual
> benchmark run - no script in this repo generates them, and they contradict
> real measurements taken this session on real Apple Silicon hardware by
> roughly 30-45x. They have been removed. Everything below is either a real
> measurement (labeled **Measured**, with how to reproduce it) or an explicit
> **Projected** estimate sourced from faster-whisper's own published GPU
> benchmarks - never present a Projected number to a customer as if it were
> Measured on this system.

## 1. Word Error Rate (WER) - Measured

Run via `python3 scripts/evaluate_wer.py --dataset benchmarks/eval_dataset.json --audio-dir benchmarks/audio` against the 15-sample clean-audio reference set (see `.benchmarks/results.md` for full methodology and caveats):

| Model | WER | Accuracy |
|---|---|---|
| `large-v3` | 4.37% | 95.63% |
| `large-v3-turbo` | 1.64% | 98.36% |

This dataset is clean short utterances, not real noisy call-center audio -
treat it as a regression check, not a production accuracy guarantee. Measure
WER on your own recorded calls (with human-transcribed reference text)
before quoting a number to a customer; `docs/WHISPER_FINETUNING.md`'s
review-queue workflow produces exactly that ground truth as a side effect of
preparing fine-tune data.

## 2. Real-Time Factor (RTF) - Apple Silicon (Measured)

Native `python scripts/dev.py` (not Docker - Docker containers are plain
Linux and never get MLX), real ~2.5 minute stereo call recordings,
`ASRService.transcribe()` end to end (VAD-gated decode + second-pass rescue
on low-confidence segments; excludes diarization and the NLP engines, which
run separately in `save_conversation_with_analysis`):

| Model | RTF |
|---|---|
| `large-v3` | 0.5 - 0.7x |
| `large-v3-turbo` | 0.45 - 0.48x |

A real ~2.5 minute call takes roughly 70-95 seconds to transcribe on this
hardware. That's an async/background-pipeline speed (results land in the
dashboard within ~1-2 minutes), not a live/real-time speed.

## 3. NVIDIA CUDA (faster-whisper) - Projected, not yet measured

No NVIDIA GPU was available in this session to measure directly.
faster-whisper's own published benchmarks report roughly 0.05-0.10x RTF for
large-v3 float16 on a T4/A10G-class GPU - if that holds here, the same
~2.5 minute call would take on the order of 10-20 seconds instead of
70-95. **Verify this on your actual target GPU before quoting it** - use
`docker-compose.gpu.yml` (see `docs/DEPLOYMENT.md`) and re-run
`scripts/evaluate_wer.py`, which reports RTF alongside WER.

## 4. NLP pipeline

Not independently benchmarked. `save_conversation_with_analysis` runs
keyword/topic/empathy/churn/compliance analysis sequentially per
conversation, sharing singleton models (sentiment classifier, diarization
pipeline) that have no verified thread-safety for concurrent inference calls
- don't parallelize this stage without adding locking first (see
`ASRService._inference_lock` for the pattern already used to guard the
Whisper model itself).
