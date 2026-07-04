# Speech recognition service integrating Faster-Whisper and MLX acceleration.
"""Thread-safe Singleton ASR Service using Faster-Whisper."""

import os
import platform
import re
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Optional

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
    speaker: Optional[str] = None


def lock_language(lang: Any) -> str:
    """Locks ASR language to Turkish ('tr') when auto-detection or empty language is passed.

    Prevents Whisper from switching languages mid-audio when encountering English
    brand names (e.g., YouTube, WhatsApp, Netflix).
    """
    if not lang or str(lang).strip().lower() in ("auto", "otomatik", "auto-detect", "detect", "default", "none", "null", ""):
        return "tr"
    return str(lang).strip().lower()


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

    @staticmethod
    def _is_stereo_file(file_path: Any) -> bool:
        if not file_path or not isinstance(file_path, str) or not os.path.exists(file_path):
            return False
        try:
            import av

            with av.open(file_path, mode="r", metadata_errors="ignore") as container:
                if container.streams.audio:
                    return getattr(container.streams.audio[0].codec_context, "channels", 1) >= 2
        except Exception:
            pass
        try:
            import wave

            with wave.open(file_path, "rb") as wf:
                return wf.getnchannels() >= 2
        except Exception:
            pass
        return False

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Sanitize Whisper hallucinations and phrase repetition loops."""
        if not text or not text.strip():
            return text
        text = re.sub(r'(\s*\.\s*){3,}', '... ', text)
        text = re.sub(r'\.{4,}', '... ', text)
        text = re.sub(r'(?:\.\s*){3,}$', '.', text)
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(parts) > 1:
            cleaned_parts = []
            for p in parts:
                p_clean = p.strip()
                if not p_clean:
                    continue
                p_norm = re.sub(r'[^\w\s]', '', p_clean.lower()).strip()
                if cleaned_parts:
                    prev_norm = re.sub(r'[^\w\s]', '', cleaned_parts[-1].lower()).strip()
                    if p_norm and prev_norm == p_norm:
                        continue
                cleaned_parts.append(p_clean)
            text = " ".join(cleaned_parts)
        for n in range(5, 0, -1):
            min_repeat = 2 if n >= 3 else 3
            pattern = r'\b((?:\w+)(?:[,.!?]?\s+\w+){' + str(n - 1) + r'})(?:[,.!?]?\s+(?:\1)){' + str(min_repeat - 1) + r',}\b'
            text = re.sub(pattern, r'\1', text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _deduplicate_segments(segments: list[Any]) -> list[Any]:
        """Apple Silicon Top-Tier Cross-Segment Deduplicator for backend services.
        Eliminates consecutive hallucinated segment loops across timestamps.
        """
        if not segments:
            return []
        cleaned = []
        recent_by_speaker: dict[str, list[tuple[float, float, str]]] = {}
        for seg in segments:
            spk = getattr(seg, "speaker", "default") or "default"
            text = getattr(seg, "text", "").strip()
            if not text:
                continue
            norm = re.sub(r'[^\w\s]', '', text.lower()).strip()
            if not norm:
                continue
            words = norm.split()
            start = float(getattr(seg, "start", 0))
            end = float(getattr(seg, "end", 0))
            
            is_dup = False
            if spk in recent_by_speaker:
                for prev_start, prev_end, prev_norm in recent_by_speaker[spk][-4:]:
                    gap = start - prev_end
                    if norm == prev_norm and gap < 25.0:
                        is_dup = True
                        break
                    if len(words) <= 4 and (norm in prev_norm or prev_norm in norm) and gap < 15.0:
                        is_dup = True
                        break
            if is_dup:
                logger.debug(f"ASR Deduplication: Dropping repeated hallucination [{spk}] '{text}' at {start:.1f}s")
                continue
            if spk not in recent_by_speaker:
                recent_by_speaker[spk] = []
            recent_by_speaker[spk].append((start, end, norm))
            cleaned.append(seg)
        return cleaned

    @staticmethod
    def _split_into_sentences(segments: list[Any]) -> list[Any]:
        """Apple Silicon Top-Tier Sentence & Timestamp Splitter.
        Breaks long multi-sentence Whisper chunks into individual sentences with proportional
        timestamps so stereo & mono dialogues are chronologically ordered.
        """
        if not segments:
            return []
        refined = []
        for seg in segments:
            is_dict = isinstance(seg, dict)
            start = float(seg["start"] if is_dict else getattr(seg, "start", 0.0))
            end = float(seg["end"] if is_dict else getattr(seg, "end", 0.0))
            text = str(seg["text"] if is_dict else getattr(seg, "text", "")).strip()
            speaker = seg["speaker"] if (is_dict and "speaker" in seg) else getattr(seg, "speaker", None)
            
            if not text:
                continue
                
            sentences = [s.strip() for s in re.split(r'(?<=[.?!;])\s+', text) if s.strip()]
            
            if len(sentences) <= 1 or (end - start) <= 2.5:
                refined.append(seg)
                continue
                
            total_chars = sum(max(len(s), 1) for s in sentences)
            duration = max(end - start, 0.1)
            
            cur_start = start
            for idx, sent in enumerate(sentences):
                sent_len = max(len(sent), 1)
                if idx == len(sentences) - 1:
                    sent_end = end
                else:
                    sent_dur = duration * (sent_len / total_chars)
                    sent_end = round(cur_start + sent_dur, 2)
                    
                if is_dict:
                    new_seg = dict(seg)
                    new_seg.update({"start": round(cur_start, 2), "end": sent_end, "text": sent})
                elif hasattr(seg, "_replace"):
                    replace_kwargs = {"start": round(cur_start, 2), "end": sent_end, "text": sent}
                    if hasattr(seg, "_fields") and "words" in seg._fields:
                        replace_kwargs["words"] = getattr(seg, "words", None) or []
                    try:
                        new_seg = seg._replace(**replace_kwargs)
                    except Exception:
                        replace_kwargs["words"] = None
                        new_seg = seg._replace(**replace_kwargs)
                elif isinstance(seg, TranscriptionSegment):
                    new_seg = TranscriptionSegment(
                        start=round(cur_start, 2),
                        end=sent_end,
                        text=sent,
                        speaker=speaker
                    )
                else:
                    import copy
                    new_seg = copy.copy(seg)
                    if hasattr(new_seg, "start"):
                        new_seg.start = round(cur_start, 2)
                    if hasattr(new_seg, "end"):
                        new_seg.end = sent_end
                    if hasattr(new_seg, "text"):
                        new_seg.text = sent
                refined.append(new_seg)
                cur_start = sent_end
        return refined

    def _transcribe_single_channel(
        self, audio_input: Any, language: str = "tr"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribe a mono file path or 1D audio numpy array."""
        language = lock_language(language)
        if self._is_mlx:
            import mlx_whisper

            repo = f"mlx-community/whisper-{self._model_size}"
            logger.debug(f"Transcribing via MLX ({repo})...")
            res = mlx_whisper.transcribe(
                audio_input,
                path_or_hf_repo=repo,
                language=language,
                condition_on_previous_text=False,
                compression_ratio_threshold=1.8,
                no_speech_threshold=0.45,
            )
            segments_gen = []
            duration = 0.0
            for s in res.get("segments", []):
                cleaned_text = self._sanitize_text(s["text"].strip())
                if cleaned_text:
                    segments_gen.append(
                        TranscriptionSegment(start=s["start"], end=s["end"], text=cleaned_text)
                    )
                duration = max(duration, s["end"])
            if duration == 0.0 and (
                isinstance(audio_input, (list, tuple))
                or (hasattr(audio_input, "shape") and len(audio_input) > 0)
            ):
                duration = len(audio_input) / 16000.0
            return self._split_into_sentences(self._deduplicate_segments(segments_gen)), duration

        segments_gen_fw, info = self._model.transcribe(
            audio_input,
            language=language,
            beam_size=5,
            condition_on_previous_text=False,
            repetition_penalty=1.20,
            no_repeat_ngram_size=3,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
        )

        segments = []
        for s in segments_gen_fw:
            cleaned_text = self._sanitize_text(s.text.strip())
            if cleaned_text:
                segments.append(
                    TranscriptionSegment(start=s.start, end=s.end, text=cleaned_text)
                )
        return self._split_into_sentences(self._deduplicate_segments(segments)), info.duration

    def transcribe(
        self, audio_path: Any, language: str = "tr"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribes an audio file or numpy array using Faster-Whisper / MLX.

        If a stereo file is detected, Left and Right channels are transcribed separately
        to guarantee 100% clean speaker separation without overlapping dialog in segments.
        """
        language = lock_language(language)
        if self._model is None:
            self.load_model()

        if isinstance(audio_path, str) and self._is_stereo_file(audio_path):
            logger.info(
                f"ASR: Stereo audio detected for {audio_path}. Transcribing Left and Right channels independently!"
            )
            try:
                from faster_whisper import decode_audio

                left_ch, right_ch = decode_audio(audio_path, sampling_rate=16000, split_stereo=True)

                # Transcribe Left Channel (SPEAKER_00)
                segs_l, dur_l = self._transcribe_single_channel(left_ch, language=language)
                for s in segs_l:
                    s.speaker = "SPEAKER_00"

                # Transcribe Right Channel (SPEAKER_01)
                segs_r, dur_r = self._transcribe_single_channel(right_ch, language=language)
                for s in segs_r:
                    s.speaker = "SPEAKER_01"

                combined = sorted(segs_l + segs_r, key=lambda x: x.start)
                combined = self._deduplicate_segments(combined)
                total_dur = max(dur_l, dur_r, 0.0)
                logger.info(
                    f"Stereo independent transcription complete: {len(segs_l)} left segs, {len(segs_r)} right segs, total={len(combined)}."
                )
                return combined, total_dur
            except Exception as exc:
                logger.warning(
                    f"Stereo independent transcription failed ({exc}), falling back to mono..."
                )

        return self._transcribe_single_channel(audio_path, language=language)
