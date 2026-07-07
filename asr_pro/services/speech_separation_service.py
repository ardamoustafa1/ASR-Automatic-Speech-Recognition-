"""Speech source separation for overlapping speech (crosstalk) windows.

DiarizationService already detects *when* two speakers talk over each other
(real acoustic overlap regions from pyannote, see
DiarizationResult.overlap_regions) but does not un-mix the audio - Whisper
transcribes the overlapping window as whichever single stream of words it
happens to lock onto, silently dropping or garbling the other party's speech.

This module separates a short overlapping window into per-source waveforms
using speechbrain's SepFormer (trained on WHAMR!, which includes noise and
reverberation - closer to real telephony conditions than a clean-speech
separation model), so each party's words can be re-transcribed independently.

Verified (see docs/DIARIZATION_DER_WORKFLOW.md) on a real 2-voice mixture:
separated streams matched their true source with 0.73/0.53 cosine similarity
(ECAPA-TDNN) vs 0.27/0.15 cross-matches - correct separation direction with
real, if imperfect, source isolation, consistent with SepFormer's published
quality on noisy/reverberant (WHAMR) conditions rather than clean-speech
benchmarks.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from loguru import logger

from asr_pro.config import _is_testing

SEPFORMER_SAMPLE_RATE = 16000


class _SepformerSeparator:
    """Thread-safe lazy singleton wrapping speechbrain's SepFormer (WHAMR!) model."""

    _instance: Any = None
    _lock = threading.Lock()
    _load_attempted = False

    @classmethod
    def get(cls) -> Any:
        if cls._instance is not None or cls._load_attempted:
            return cls._instance
        with cls._lock:
            if cls._instance is not None or cls._load_attempted:
                return cls._instance
            cls._load_attempted = True
            if _is_testing:
                logger.debug("SpeechSeparation: testing mode, skipping SepFormer model load.")
                return None
            try:
                import torch
                from speechbrain.inference.separation import SepformerSeparation

                # speechbrain's custom ops are not reliably supported on MPS
                # (same constraint as BiometricService's ECAPA-TDNN); CUDA if
                # available, otherwise CPU. Separation only runs on short
                # (sub-few-second) crosstalk windows, not full calls, so CPU
                # latency is acceptable.
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._instance = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-whamr16k",
                    savedir="data/models/sepformer-whamr16k",
                    run_opts={"device": device},
                )
                logger.info(f"SpeechSeparation: SepFormer (WHAMR16k) loaded on {device}.")
            except Exception as exc:
                logger.warning(
                    f"SpeechSeparation: could not load SepFormer ({exc}). Crosstalk windows will "
                    "be transcribed as a single mixed stream - overlapping speech accuracy is "
                    "unmitigated in this mode."
                )
                cls._instance = None
        return cls._instance


def separate_two_speakers(
    audio: np.ndarray, sample_rate: int = SEPFORMER_SAMPLE_RATE
) -> list[np.ndarray] | None:
    """Separate a mixed audio window into two estimated per-speaker waveforms.

    Returns None if the separation model is unavailable or inference fails -
    callers must treat this as "no separation possible", not an error to
    propagate, since the caller's fallback (transcribing the raw mixture) is
    always a valid degraded mode.
    """
    if audio.size == 0:
        return None
    if sample_rate != SEPFORMER_SAMPLE_RATE:
        raise ValueError(
            f"separate_two_speakers requires {SEPFORMER_SAMPLE_RATE}Hz audio, got {sample_rate}Hz"
        )

    model = _SepformerSeparator.get()
    if model is None:
        return None

    try:
        import torch

        waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, time)
        with torch.no_grad():
            estimated = model.separate_batch(waveform)  # (1, time, n_src)
        n_sources = estimated.shape[-1]
        return [
            estimated[0, :, i].detach().cpu().numpy().astype(np.float32) for i in range(n_sources)
        ]
    except Exception as exc:
        logger.warning(f"SpeechSeparation: inference failed ({exc}). No separation applied.")
        return None
