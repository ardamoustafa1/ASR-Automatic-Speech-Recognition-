"""Thread-safe Singleton ASR Service using Faster-Whisper."""

import os
import platform
import threading
import warnings
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment]

warnings.simplefilter("ignore")


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


class ASRService:
    """Thread-safe Singleton for Faster-Whisper model management."""

    _instance: Optional["ASRService"] = None
    _lock = threading.Lock()
    _model = None
    _model_size = "turbo"

    @classmethod
    def get_instance(cls) -> "ASRService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._device = self._choose_device()
        self._compute_type = self._choose_compute_type()

    def _choose_device(self) -> str:
        if platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
            # Apple Silicon: faster-whisper falls back to CPU (use mlx-whisper for native MPS)
            return "cpu"
        return "cuda" if self._check_cuda() else "cpu"

    def _check_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def _choose_compute_type(self) -> str:
        if self._device == "cuda":
            return "float16"
        if self._device == "mps":
            return "float16"
        return "int8"

    def load_model(self, model_size: str = "turbo") -> "WhisperModel | None":
        if self._model is not None and self._model_size == model_size:
            return self._model
        if getattr(self, "_is_mlx", False) and self._model_size == model_size:
            return None

        # Apple Silicon MLX Check
        if platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
            try:
                import mlx_whisper  # noqa: F401

                self._device = "mps"
                self._compute_type = "float16"
                self._model_size = model_size
                self._is_mlx = True
                logger.info(
                    f"Initialized ASR model '{model_size}' using hardware-accelerated Apple MLX engine."
                )
                return None  # MLX handles loading during transcribe
            except ImportError:
                logger.warning(
                    "mlx-whisper module not detected. Falling back to CPU int8 execution. For optimal Apple Silicon performance, install mlx-whisper."
                )
                self._is_mlx = False
        else:
            self._is_mlx = False

        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed. Run: pip install faster-whisper")

        cpu_threads = max(4, os.cpu_count() or 4)
        num_workers = 1 if self._device == "cuda" else min(4, max(1, cpu_threads // 2))

        logger.info(f"Loading ASR model '{model_size}' on {self._device} ({self._compute_type})")
        self._model = WhisperModel(
            model_size,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        self._model_size = model_size
        logger.info(f"ASR model '{model_size}' loaded successfully.")
        return self._model

    def transcribe(
        self, audio_path: str, language: str = "tr"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribes an audio file using Faster-Whisper.

        Returns:
            Tuple of (list of segments, total duration in seconds).
        """
        if self._model is None:
            self.load_model()

        if self._is_mlx:
            import mlx_whisper

            repo = f"mlx-community/whisper-{self._model_size}"
            logger.debug(f"Transcribing via MLX ({repo})...")
            res = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=repo,
                language=language,
            )
            segments_gen = []
            duration = 0.0
            for s in res.get("segments", []):
                segments_gen.append(
                    TranscriptionSegment(start=s["start"], end=s["end"], text=s["text"].strip())
                )
                duration = max(duration, s["end"])
            return segments_gen, duration

        segments_gen_fw, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        segments = [
            TranscriptionSegment(start=s.start, end=s.end, text=s.text.strip())
            for s in segments_gen_fw
        ]

        logger.debug(f"Transcribed {len(segments)} segments, duration={info.duration:.1f}s")
        return segments, info.duration
