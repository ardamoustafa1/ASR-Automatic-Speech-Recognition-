# Whisper Fine-Tuning Workflow (In-Domain WER Reduction)

After inference-time tuning (VAD gating, second-pass rescue, domain prompts,
phonetic correction), the single biggest remaining accuracy lever is
fine-tuning Whisper on **your own** 8kHz Turkish call-center audio. Stock
large-v3 was trained mostly on broadband internet audio; telephony-band
speech with campaign jargon is exactly the distribution shift fine-tuning
fixes.

## Requirements

- Recordings from the target deployment (e.g. the 170 stereo calls in `sesler/`).
- A CUDA GPU for training (24GB for large-v3 with LoRA; 16GB fits `large-v3-turbo`).
  Data preparation runs fine on a Mac.
- Training-only Python deps (not in `requirements.txt`):
  `pip install transformers datasets peft accelerate soundfile evaluate jiwer`

## Step 1 — Build the dataset (semi-supervised)

```bash
python3 scripts/prepare_whisper_finetune_data.py \
    --audio-dir sesler \
    --output-dir data/whisper_finetune
```

Each call is transcribed with the full production pipeline and cut into
per-segment clips (correct stereo channel per speaker). Segments are split by
decoder confidence:

| Output | Content | Use |
|---|---|---|
| `train_auto.jsonl` | avg_logprob ≥ −0.35 — model near-certain | pseudo-labels, train as-is |
| `review_queue.jsonl` | low-confidence segments | **human correction required** |

The script is resume-safe; re-running skips processed files.

## Step 2 — Human review (where the WER gain lives)

Open `review_queue.jsonl`, fix each row's `"text"` to what was actually said,
set `"review": "corrected"`. Rows you can't understand: leave as `pending`
(they are excluded). Then merge:

```bash
python3 scripts/prepare_whisper_finetune_data.py \
    --merge-reviewed data/whisper_finetune/review_queue.jsonl
```

Realistic effort: the low-confidence queue is typically 10–20% of segments;
an hour of reviewing yields several hundred gold segments. Even 1–2 hours of
corrected audio measurably moves in-domain WER; pseudo-labels alone mostly
reinforce what the model already knows.

## Step 3 — Train (LoRA)

```bash
python3 scripts/finetune_whisper_lora.py \
    --manifests data/whisper_finetune/train_reviewed.jsonl data/whisper_finetune/train_auto.jsonl \
    --data-root data/whisper_finetune \
    --model openai/whisper-large-v3 \
    --output-dir data/whisper_finetune/lora_out \
    --epochs 3
```

- Eval WER is computed against **human-corrected rows only** (10% holdout).
- Pseudo-labeled rows are capped at 3× the reviewed rows so they can't drown
  out the human corrections (`--max-auto-ratio`).

## Step 4 — Deploy the fine-tuned model

The training script writes `lora_out/merged/` (adapter merged into base weights)
and prints both conversion commands:

**Production (GPU, faster-whisper):**

```bash
pip install ctranslate2 transformers
ct2-transformers-converter --model data/whisper_finetune/lora_out/merged \
    --output_dir data/whisper_finetune/lora_out/ct2 --quantization float16
```

Set in `.env`: `ASR_ASR_MODEL_SIZE=/absolute/path/to/lora_out/ct2` — faster-whisper
accepts a local CTranslate2 directory anywhere a model name is accepted.

**Dev (Apple Silicon, MLX):**

```bash
python -m mlx_whisper.convert \
    --torch-name-or-path data/whisper_finetune/lora_out/merged \
    --mlx-path data/whisper_finetune/lora_out/mlx
```

## Measuring the gain

Keep a fixed held-out set of reviewed calls that never enter training, and run
`scripts/evaluate_wer.py` against it before/after. Report WER on that set in
sales material rather than public-benchmark numbers - buyers will test on
their own recordings anyway.

## KVKK note

Fine-tune datasets contain raw customer speech. `prepare_whisper_finetune_data.py`
output inherits whatever was said on the call - treat `data/whisper_finetune/`
with the same access controls as the recordings themselves, and run training
on infrastructure covered by the client's data-processing agreement.
