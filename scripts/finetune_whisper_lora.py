#!/usr/bin/env python3
"""
LoRA fine-tune Whisper on your own call-center segments (prepared by
scripts/prepare_whisper_finetune_data.py).

Fine-tuning on in-domain 8kHz Turkish telephony audio is the single biggest
remaining WER lever after inference-time tuning. LoRA (low-rank adapters)
makes this practical: only ~1% of weights train, so large-v3 fits on a single
24GB GPU (or even 16GB with --model large-v3-turbo).

Extra dependencies (NOT in requirements.txt - training-only):
    pip install transformers datasets peft accelerate soundfile evaluate jiwer

Usage (on a CUDA machine; MPS works but is slow):
  python3 scripts/finetune_whisper_lora.py \\
      --manifests data/whisper_finetune/train_reviewed.jsonl data/whisper_finetune/train_auto.jsonl \\
      --data-root data/whisper_finetune \\
      --model openai/whisper-large-v3 \\
      --output-dir data/whisper_finetune/lora_out \\
      --epochs 3

Quality guidance:
  - Prefer human-corrected rows (train_reviewed.jsonl). Pseudo-labels
    (train_auto.jsonl) are safe but mostly reinforce what the model already
    knows; the corrected low-confidence rows are where the WER gain lives.
  - Hold out ~10% of REVIEWED rows for eval (--eval-holdout 0.1) so the
    reported WER is measured against human ground truth only.

Deploying the result:
  1. Merge the adapter:  (this script does it automatically at the end)
  2. faster-whisper (production GPU):
       ct2-transformers-converter --model <output>/merged \\
           --output_dir <output>/ct2 --quantization float16
     then set ASR_ASR_MODEL_SIZE=<absolute path to <output>/ct2>
  3. Apple Silicon / MLX (dev machines):
       python -m mlx_whisper.convert --torch-name-or-path <output>/merged \\
           --mlx-path <output>/mlx
"""

import argparse
import json
import random
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _require_training_deps():
    missing = []
    for mod in ("transformers", "datasets", "peft", "soundfile", "evaluate"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        sys.exit(
            "Missing training dependencies: "
            + ", ".join(missing)
            + "\nInstall with: pip install transformers datasets peft accelerate soundfile evaluate jiwer"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Whisper on call-center data.")
    parser.add_argument(
        "--manifests",
        nargs="+",
        required=True,
        help="JSONL manifests from prepare_whisper_finetune_data.py (reviewed first).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Directory the manifest 'audio' paths are relative to.",
    )
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--output-dir", type=str, default="data/whisper_finetune/lora_out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--eval-holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-auto-ratio",
        type=float,
        default=3.0,
        help="Cap pseudo-labeled rows at this multiple of reviewed rows so auto "
        "labels never drown out human corrections (ignored if no reviewed rows).",
    )
    args = parser.parse_args()

    _require_training_deps()

    import torch
    from datasets import Audio, Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    random.seed(args.seed)
    data_root = Path(args.data_root)

    reviewed_rows, auto_rows = [], []
    for manifest in args.manifests:
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("text", "").strip():
                continue
            row["audio"] = str(data_root / row["audio"])
            (reviewed_rows if row.get("review") == "corrected" else auto_rows).append(row)

    if reviewed_rows and args.max_auto_ratio > 0:
        cap = int(len(reviewed_rows) * args.max_auto_ratio)
        if len(auto_rows) > cap:
            random.shuffle(auto_rows)
            auto_rows = auto_rows[:cap]

    # Eval set: human-verified rows only - pseudo-label eval would just measure
    # agreement with the base model, not real WER.
    random.shuffle(reviewed_rows)
    n_eval = int(len(reviewed_rows) * args.eval_holdout)
    eval_rows = reviewed_rows[:n_eval]
    train_rows = reviewed_rows[n_eval:] + auto_rows
    random.shuffle(train_rows)
    if not train_rows:
        sys.exit("No training rows found in the given manifests.")
    print(
        f"Training rows: {len(train_rows)} ({len(reviewed_rows) - n_eval} reviewed + "
        f"{len(auto_rows)} pseudo-labeled) | Eval rows (human-verified): {len(eval_rows)}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    if device != "cuda":
        print(f"WARNING: training on {device} - expect this to be slow; a CUDA GPU is recommended.")

    processor = WhisperProcessor.from_pretrained(args.model, language="turkish", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def to_dataset(rows):
        ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))

        def prepare(batch):
            audio = batch["audio"]
            batch["input_features"] = processor.feature_extractor(
                audio["array"], sampling_rate=16000
            ).input_features[0]
            batch["labels"] = processor.tokenizer(batch["text"]).input_ids
            return batch

        return ds.map(prepare, remove_columns=ds.column_names, num_proc=1)

    train_ds = to_dataset(train_rows)
    eval_ds = to_dataset(eval_rows) if eval_rows else None

    import numpy as np

    def collate(features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

    metric = None
    if eval_ds is not None:
        import evaluate

        metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

    out_dir = Path(args.output_dir)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        fp16=device == "cuda",
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        predict_with_generate=eval_ds is not None,
        generation_max_length=225,
        logging_steps=25,
        report_to=[],
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="wer" if eval_ds is not None else None,
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        processing_class=processor.feature_extractor,
    )
    _ = np  # keep numpy import for downstream hooks

    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to {adapter_dir}")

    # Merge adapter into base weights for deployment converters (CT2 / MLX).
    merged = trainer.model.merge_and_unload()
    merged_dir = out_dir / "merged"
    merged.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    print(
        f"Merged model saved to {merged_dir}\n\n"
        "Deploy:\n"
        f"  faster-whisper: ct2-transformers-converter --model {merged_dir} "
        f"--output_dir {out_dir / 'ct2'} --quantization float16\n"
        f"                  then ASR_ASR_MODEL_SIZE={out_dir / 'ct2'}\n"
        f"  MLX (Mac):      python -m mlx_whisper.convert --torch-name-or-path {merged_dir} "
        f"--mlx-path {out_dir / 'mlx'}"
    )


if __name__ == "__main__":
    main()
