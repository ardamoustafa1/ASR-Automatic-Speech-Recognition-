#!/usr/bin/env python3
"""
Fine-tune pyannote's segmentation model on your own annotated call recordings.

pyannote-3.1's published DER (~10-15% on clean benchmark corpora) reflects
training on general-purpose datasets (VoxConverse, DIHARD, AMI, etc.), not
Turkish call-center telephony audio specifically. Fine-tuning the
segmentation model on real annotated calls from the target deployment is the
single highest-leverage accuracy lever available - bigger than any inference-
time tuning (VAD thresholds, speaker-count bounds, audio conditioning).

Prerequisites:
  1. Real annotated calls (scripts/annotate_diarization.py) - realistically
     at least a few hours of annotated audio; a handful of calls will run but
     won't produce a model that generalizes.
  2. A prepared protocol (scripts/prepare_diarization_finetune_data.py).

Usage:
  python3 scripts/finetune_diarization.py \\
      --database-yml data/diarization_finetune/database.yml \\
      --protocol ASRPro.SpeakerDiarization.Custom \\
      --output-dir data/diarization_finetune/checkpoints \\
      --max-epochs 20

After training, point `ASR_DIARIZATION_FINETUNED_SEGMENTATION_PATH` at the
best checkpoint (printed at the end of this script) - DiarizationService
will load it in place of the stock pyannote/segmentation-3.0 segmentation
model on next startup (see asr_pro/services/diarization_service.py,
_load_finetuned_segmentation_model()).
"""

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune pyannote segmentation on annotated call recordings."
    )
    parser.add_argument("--database-yml", type=str, required=True)
    parser.add_argument("--protocol", type=str, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="pyannote/segmentation-3.0",
        help="Pretrained segmentation checkpoint to start fine-tuning from - this is the "
        "same base model pyannote/speaker-diarization-3.1 itself uses internally.",
    )
    parser.add_argument("--output-dir", type=str, default="data/diarization_finetune/checkpoints")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-speakers-per-frame",
        type=int,
        default=2,
        help="2 is correct for standard two-party contact-center calls; raise for "
        "conference/multi-party calls if your annotated data includes them.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Low LR appropriate for fine-tuning, not from-scratch training.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop if validation DER hasn't improved for this many epochs.",
    )
    args = parser.parse_args()

    database_yml = WORKSPACE_ROOT / args.database_yml
    if not database_yml.exists():
        print(
            f"Database config not found: {database_yml}. Run scripts/prepare_diarization_finetune_data.py first."
        )
        sys.exit(1)

    # pyannote.audio's Model subclasses lightning.pytorch.LightningModule
    # specifically - the separately-installed `pytorch_lightning` package
    # ships a *different* LightningModule class, and passing a pyannote Model
    # to a `pytorch_lightning.Trainer` fails an isinstance check
    # ("model must be a LightningModule ... got PyanNet"). Import from
    # `lightning.pytorch`, not `pytorch_lightning`, to match.
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pyannote.audio import Model
    from pyannote.audio.tasks import SpeakerDiarization
    from pyannote.database import get_protocol, registry

    registry.load_database(str(database_yml))
    protocol = get_protocol(args.protocol)

    print(f"Loading base segmentation checkpoint: {args.base_checkpoint}")
    model = Model.from_pretrained(args.base_checkpoint)

    task = SpeakerDiarization(
        protocol,
        duration=model.specifications.duration,
        max_speakers_per_frame=args.max_speakers_per_frame,
        batch_size=args.batch_size,
    )
    model.task = task

    # Fine-tuning (not from-scratch training): a low, fixed learning rate on
    # all parameters. pyannote's Model.configure_optimizers default already
    # reads model.hparams if set; we override it explicitly here so the
    # learning rate is visible and controllable from the CLI instead of
    # buried in defaults.
    def configure_optimizers():
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.configure_optimizers = configure_optimizers

    output_dir = WORKSPACE_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # task.val_monitor is (metric_name, "min"|"max") - the task's own validation
    # DiarizationErrorRate metric. Monitoring it directly picks the best-DER
    # epoch instead of just the most recent one.
    metric_name, metric_mode = task.val_monitor
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="{epoch}-{step}",
        save_top_k=3,
        monitor=metric_name,
        mode=metric_mode,
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor=metric_name, mode=metric_mode, patience=args.early_stopping_patience),
    ]

    accelerator = (
        "gpu"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=1,
        default_root_dir=str(output_dir),
        log_every_n_steps=1,
    )

    print(f"Starting fine-tuning: {args.max_epochs} max epochs, accelerator={accelerator}")
    trainer.fit(model)

    best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Best checkpoint: {best_path}")
    print(
        "\nTo use it in production, set:\n"
        f"  ASR_DIARIZATION_FINETUNED_SEGMENTATION_PATH={best_path}\n"
        "then restart the service. Verify with scripts/evaluate_diarization.py "
        "against a held-out set BEFORE relying on it - a fine-tuned model that "
        "overfit a small annotated set can be worse than the stock checkpoint."
    )


if __name__ == "__main__":
    main()
