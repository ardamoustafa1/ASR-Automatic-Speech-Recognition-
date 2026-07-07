# Diarization Accuracy Measurement (DER) Workflow

This documents how to turn "we improved diarization" into a verifiable number,
using [Diarization Error Rate](https://pyannote.github.io/pyannote-metrics/)
(DER) - the same metric pyannote reports for its own published benchmarks.

DER = (false alarm + missed detection + speaker confusion) / total reference
speech time. Lower is better; 0.0 is perfect; pyannote-3.1's published DER on
clean benchmark corpora is roughly 10-15%. Telephony audio with real crosstalk
and channel bleed typically scores worse - only a number measured on your own
call recordings tells you what matters for this deployment.

## Two tracks: synthetic smoke test vs. real production DER

**These are not interchangeable and must not be reported as the same thing.**

| | Synthetic smoke test | Real production DER |
|---|---|---|
| Purpose | Prove the DER harness + pipeline plumbing work end-to-end | Measure actual deployment accuracy |
| Source | `scripts/build_synthetic_diarization_benchmark.py` | `scripts/annotate_diarization.py` + real calls |
| Ground truth | Exact (we control every clip boundary) | Human-annotated by a reviewer listening to the call |
| Crosstalk/overlap | None (clips are concatenated with silence gaps) | Present, if it occurs in the recording |
| What a low DER proves | The pipeline correctly separates two genuinely different voices with no noise/overlap | The pipeline works on real, messy telephony audio |

### Synthetic smoke test

```bash
python3 scripts/build_synthetic_diarization_benchmark.py --num-calls 5 --clips-per-call 4
python3 scripts/evaluate_diarization.py \
    --audio-dir benchmarks/diarization_audio \
    --rttm-dir benchmarks/diarization_rttm
```

Known gotcha this script already handles: all 15 source clips in
`benchmarks/audio/` (the WER benchmark's single-speaker narrations) are the
*same narrator* - verified via ECAPA-TDNN cosine similarity (~0.90-0.91
between any two clips, the same range as a genuine same-speaker match).
Alternating them unmodified as "agent"/"customer" would build a benchmark
with only one real voice in it; a diarization system correctly reporting "1
speaker" on that scores ~50% DER against a 2-speaker ground truth - a
benchmark defect, not a diarization failure (this was caught and fixed during
development: an earlier version of this benchmark reported 49% DER across
every synthetic call, which is exactly chance-level for collapsing 2
equal-duration speakers into 1). The script now pitch/tempo-shifts the
"customer" role by -6 semitones via resampling, which drops cross-role ECAPA
cosine similarity to ~0.15-0.30 - genuinely separable. With that fix, the
harness correctly detects 2 speakers and reports well under 1% DER (measured:
0.00%-1.47% across 5 synthetic calls) - as expected for a clean, non-
overlapping, genuinely 2-voice case.

### Real production DER (the number that actually matters)

1. Pick ~20-30 real, representative call recordings (mix of short/long,
   quiet/noisy, two-party and any conference/transfer calls if those occur).
   A handful of files gives a directional signal, not a stable metric -
   more files narrow the confidence interval.
2. For each call, listen to it and note every speaker turn:
   ```bash
   # <call_id>.csv
   start_sec,end_sec,speaker
   0.0,3.2,agent
   3.4,7.9,customer
   8.1,8.6,agent
   ```
3. Convert to RTTM:
   ```bash
   python3 scripts/annotate_diarization.py --csv <call_id>.csv \
       --output benchmarks/diarization_rttm_real/<call_id>.rttm --uri <call_id>
   ```
4. Place the matching audio in a directory (e.g. `benchmarks/diarization_audio_real/<call_id>.wav`).
5. Run the same evaluation script against the real directories:
   ```bash
   python3 scripts/evaluate_diarization.py \
       --audio-dir benchmarks/diarization_audio_real \
       --rttm-dir benchmarks/diarization_rttm_real \
       --output-json data/der_report_production.json \
       --max-der 0.25
   ```
6. Track this number over time (model version bumps, config changes, prompt
   changes) the same way `scripts/evaluate_wer.py` tracks WER - a regression
   gate, not a one-time report.

## Interpreting results

- `hypothesis_speakers` vs `reference_speakers` mismatch in the per-file
  breakdown is the fastest signal for under/over-clustering
  (`diarization_min_speakers`/`diarization_max_speakers` in
  `asr_pro/config.py`).
- A DER that's fine on the synthetic benchmark but bad on real calls almost
  always means the real calls have crosstalk, channel bleed, or noise the
  synthetic benchmark doesn't - see `docs/DIARIZATION_LIMITATIONS.md`.
- Compare `diarization_method` (in `conversation_service`'s output metadata)
  against DER per file: if degraded methods (`stereo_energy`/
  `text_heuristic`) are showing up in production traffic at all, that is a
  configuration/deployment problem (missing HF token) to fix before DER
  measurement is even meaningful.

## Next lever once you have a real DER baseline: fine-tuning

pyannote-3.1's published DER reflects training on general-purpose corpora
(VoxConverse, DIHARD, AMI), not Turkish call-center telephony specifically.
Fine-tuning its segmentation model on your own annotated calls is the single
highest-leverage accuracy lever available - bigger than any inference-time
tuning. Once you have ~20-30+ annotated real calls (the same RTTM data used
for DER measurement above):

```bash
python3 scripts/prepare_diarization_finetune_data.py \
    --audio-dir benchmarks/diarization_audio_real \
    --rttm-dir benchmarks/diarization_rttm_real \
    --output-dir data/diarization_finetune

python3 scripts/finetune_diarization.py \
    --database-yml data/diarization_finetune/database.yml \
    --protocol ASRPro.SpeakerDiarization.Custom \
    --output-dir data/diarization_finetune/checkpoints \
    --max-epochs 20
```

This fine-tunes `pyannote/segmentation-3.0` (the same base model
`pyannote/speaker-diarization-3.1` uses internally) via
`pyannote.audio`'s `SpeakerDiarization` task + a `lightning.pytorch.Trainer`
- standard supervised fine-tuning, not a from-scratch model. The embedding
model, clustering algorithm, and all other pretrained-pipeline hyperparameters
are left untouched; only the segmentation sub-model is replaced.

Point the service at the resulting checkpoint:

```bash
export ASR_DIARIZATION_FINETUNED_SEGMENTATION_PATH=/path/to/best.ckpt
```

`DiarizationService._maybe_swap_finetuned_segmentation()`
(`asr_pro/services/diarization_service.py`) loads it in place of the stock
segmentation model on next pipeline load, reusing the exact `Inference`
construction (duration/step/batch_size) pyannote's own pipeline
`__init__` uses, so nothing else in the pipeline needs to change.

**Verified end-to-end**: this whole pipeline (protocol prep -> fine-tuning
loop -> checkpoint -> production swap) was smoke-tested against the synthetic
benchmark above - a real gradient-descent training run for 1 epoch completed
and produced a loadable checkpoint that `DiarizationService` swapped in and
ran inference with successfully. That 1-epoch/3-file run is deliberately
undertrained (its own validation DER got worse than the stock model, exactly
because 3 files for 1 epoch is not real training data) - it only proves the
*mechanism* works, not that fine-tuning ran with the necessary volume of real
data. Always compare a fine-tuned checkpoint's DER against the stock
pipeline's DER on the same held-out set (`--max-der` flag on
`scripts/evaluate_diarization.py`) before deploying it - an undertrained or
overfit fine-tune can be worse than the stock checkpoint.
