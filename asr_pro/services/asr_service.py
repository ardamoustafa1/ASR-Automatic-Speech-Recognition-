# Speech recognition service integrating Faster-Whisper and MLX acceleration.
"""Thread-safe Singleton ASR Service using Faster-Whisper."""

import os
import platform
import re
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from loguru import logger

from asr_pro.config import settings
from asr_pro.observability.metrics import asr_transcribe_duration_seconds, time_block

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


from asr_pro.utils.lang_utils import lock_language


class ASRService:
    """Thread-safe Singleton for Faster-Whisper model management."""

    _instance: Optional["ASRService"] = None
    _lock = threading.Lock()
    _model = None
    _model_size = "large-v3"

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
        # Serializes calls into the underlying model: CTranslate2/mlx are not
        # guaranteed safe for concurrent inference from multiple threads on a
        # single model instance (relevant once live streaming sessions and
        # batch uploads can call transcribe() at the same time).
        self._inference_lock = threading.Lock()

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

    def load_model(self, model_size: str = "large-v3") -> "WhisperModel | None":
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
        text = re.sub(r"(\s*\.\s*){3,}", "... ", text)
        text = re.sub(r"\.{4,}", "... ", text)
        text = re.sub(r"(?:\.\s*){3,}$", ".", text)
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(parts) > 1:
            cleaned_parts = []
            for p in parts:
                p_clean = p.strip()
                if not p_clean:
                    continue
                p_norm = re.sub(r"[^\w\s]", "", p_clean.lower()).strip()
                if cleaned_parts:
                    prev_norm = re.sub(r"[^\w\s]", "", cleaned_parts[-1].lower()).strip()
                    if p_norm and prev_norm == p_norm:
                        continue
                cleaned_parts.append(p_clean)
            text = " ".join(cleaned_parts)
        for n in range(5, 0, -1):
            min_repeat = 2 if n >= 3 else 3
            pattern = (
                r"\b((?:\w+)(?:[,.!?]?\s+\w+){"
                + str(n - 1)
                + r"})(?:[,.!?]?\s+(?:\1)){"
                + str(min_repeat - 1)
                + r",}\b"
            )
            text = re.sub(pattern, r"\1", text, flags=re.IGNORECASE)
        from asr_pro.services.domain_adaptation import DomainAdaptationService
        text = DomainAdaptationService.correct_telecom_terms(text)
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
            norm = re.sub(r"[^\w\s]", "", text.lower()).strip()
            if not norm:
                continue
            words = norm.split()
            start = float(getattr(seg, "start", 0))
            end = float(getattr(seg, "end", 0))

            is_dup = False
            if spk in recent_by_speaker:
                for _prev_start, prev_end, prev_norm in recent_by_speaker[spk][-4:]:
                    gap = start - prev_end
                    if norm == prev_norm and gap < 25.0:
                        is_dup = True
                        break
                    if len(words) <= 4 and (norm in prev_norm or prev_norm in norm) and gap < 15.0:
                        is_dup = True
                        break
            if is_dup:
                logger.debug(
                    f"ASR Deduplication: Dropping repeated hallucination [{spk}] '{text}' at {start:.1f}s"
                )
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
            speaker = (
                seg["speaker"] if (is_dict and "speaker" in seg) else getattr(seg, "speaker", None)
            )

            if not text:
                continue

            sentences = [s.strip() for s in re.split(r"(?<=[.?!;])\s+", text) if s.strip()]

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
                        start=round(cur_start, 2), end=sent_end, text=sent, speaker=speaker
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

    @staticmethod
    def _safe_float_attr(obj, attr, default=0.0):
        val = getattr(obj, attr, default) if not isinstance(obj, dict) else obj.get(attr, default)
        if val is None or "mock" in str(type(val)).lower():
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _transcribe_single_channel(
        self, audio_input: Any, language: str = "tr"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribe a mono file path or 1D audio numpy array."""
        language = lock_language(language)
        from asr_pro.services.audio_conditioning import condition_telephony_audio
        from asr_pro.services.domain_adaptation import DomainAdaptationService
        audio_input = condition_telephony_audio(audio_input)
        domain_prompt = f"{settings.asr_initial_prompt or ''} {DomainAdaptationService.get_initial_prompt()}".strip()
        if self._is_mlx:
            import mlx_whisper

            # mlx-community's repo naming isn't a uniform "whisper-{size}"
            # pattern (e.g. large-v3 is hosted as "large-v3-mlx", not
            # "large-v3") - translate known exceptions, otherwise mirror
            # size unchanged (works for "turbo", "small", etc).
            mlx_repo_overrides = {"large-v3": "large-v3-mlx"}
            repo_size = mlx_repo_overrides.get(self._model_size, self._model_size)
            repo = f"mlx-community/whisper-{repo_size}"
            logger.debug(f"Transcribing via MLX ({repo})...")
            with self._inference_lock:
                res = mlx_whisper.transcribe(
                    audio_input,
                    path_or_hf_repo=repo,
                    language=language,
                    condition_on_previous_text=False,
                    initial_prompt=domain_prompt,
                    compression_ratio_threshold=2.0,
                    no_speech_threshold=settings.asr_no_speech_threshold,
                    word_timestamps=True,
                )
            segments_gen = []
            duration = 0.0
            for s in res.get("segments", []):
                no_speech = self._safe_float_attr(s, "no_speech_prob", 0.0)
                logprob = self._safe_float_attr(s, "avg_logprob", 0.0)
                comp_ratio = self._safe_float_attr(s, "compression_ratio", 1.0)
                if no_speech > 0.6 or (logprob < -0.8 and comp_ratio > 1.8):
                    continue
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

        temperature_schedule = tuple(
            float(t.strip()) for t in settings.asr_temperature.split(",") if t.strip()
        )

        with self._inference_lock:
            segments_gen_fw, info = self._model.transcribe(
                audio_input,
                language=language,
                beam_size=settings.asr_beam_size,
                initial_prompt=domain_prompt,
                word_timestamps=settings.asr_word_timestamps,
                condition_on_previous_text=False,
                repetition_penalty=1.20,
                no_repeat_ngram_size=3,
                compression_ratio_threshold=2.0,
                log_prob_threshold=settings.asr_log_prob_threshold,
                no_speech_threshold=settings.asr_no_speech_threshold,
                temperature=temperature_schedule,
                vad_filter=True,
                vad_parameters={
                    "threshold": settings.asr_vad_threshold,
                    "min_speech_duration_ms": settings.asr_vad_min_speech_ms,
                    "min_silence_duration_ms": settings.asr_vad_min_silence_ms,
                    "speech_pad_ms": settings.asr_vad_speech_pad_ms,
                },
            )
            # Materialize the generator while still holding the lock: faster-whisper
            # decodes lazily as the generator is consumed, so a concurrent caller
            # could otherwise interleave inference with this one.
            segments_gen_fw = list(segments_gen_fw)

        segments = []
        for s in segments_gen_fw:
            no_speech = self._safe_float_attr(s, "no_speech_prob", 0.0)
            logprob = self._safe_float_attr(s, "avg_logprob", 0.0)
            comp_ratio = self._safe_float_attr(s, "compression_ratio", 1.0)
            if no_speech > 0.6 or (logprob < -0.8 and comp_ratio > 1.8):
                continue
            cleaned_text = self._sanitize_text(s.text.strip())
            if cleaned_text:
                segments.append(TranscriptionSegment(start=s.start, end=s.end, text=cleaned_text))
        return self._split_into_sentences(self._deduplicate_segments(segments)), info.duration

    def _transcribe_stereo(
        self, audio_path: str, language: str = "tr"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribe stereo audio by isolating left/right channels with post-transcription RMS bleed filtering.

        Left channel is assigned to SPEAKER_00 and right channel to SPEAKER_01.
        An acoustic RMS energy comparison filters out bleed-through after transcription so each party's words
        are only retained on their actual physical channel.
        """
        from faster_whisper import decode_audio

        logger.info(
            f"ASR: Stereo audio detected for {audio_path}. Isolating channels with post-transcription RMS bleed filtering."
        )
        left_ch, right_ch = decode_audio(audio_path, sampling_rate=16000, split_stereo=True)
        from asr_pro.services.audio_conditioning import condition_telephony_audio, is_ivr_segment
        left_ch = condition_telephony_audio(left_ch)
        right_ch = condition_telephony_audio(right_ch)

        with time_block(asr_transcribe_duration_seconds, mode="batch"):
            segs_left, dur_left = self._transcribe_single_channel(left_ch, language=language)
            segs_right, dur_right = self._transcribe_single_channel(right_ch, language=language)

        def _get_rms(channel_data: np.ndarray, start_sec: float, end_sec: float) -> float:
            start_idx = int(start_sec * 16000)
            end_idx = int(end_sec * 16000)
            slice_data = channel_data[start_idx:end_idx]
            if len(slice_data) == 0:
                return 0.0
            return float(np.sqrt(np.mean(slice_data**2)))

        clean_left = []
        for s in segs_left:
            rms_l = _get_rms(left_ch, s.start, s.end)
            rms_r = _get_rms(right_ch, s.start, s.end)
            if rms_r > rms_l * 4.0 and rms_l < 0.015:
                continue
            s.speaker = "[🤖 IVR / Sistem Mesajı]" if is_ivr_segment(s.text, s.start) else "SPEAKER_00"
            clean_left.append(s)

        clean_right = []
        for s in segs_right:
            rms_r = _get_rms(right_ch, s.start, s.end)
            rms_l = _get_rms(left_ch, s.start, s.end)
            if rms_l > rms_r * 4.0 and rms_r < 0.015:
                continue
            s.speaker = "[🤖 IVR / Sistem Mesajı]" if is_ivr_segment(s.text, s.start) else "SPEAKER_01"
            clean_right.append(s)

        merged_segments = sorted(clean_left + clean_right, key=lambda x: (x.start, x.end))
        duration = max(dur_left, dur_right)
        if duration == 0.0 and len(merged_segments) > 0:
            duration = max(s.end for s in merged_segments)

        return self._split_into_sentences(self._deduplicate_segments(merged_segments)), duration

    def transcribe(
        self, audio_path: Any, language: str = "tr"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribes an audio file or numpy array using Faster-Whisper / MLX.

        Stereo files are transcribed by isolating left/right channels with DSP crosstalk suppression.
        """
        language = lock_language(language)
        if self._model is None and not getattr(self, "_is_mlx", False):
            self.load_model()

        if isinstance(audio_path, str) and self._is_stereo_file(audio_path):
            try:
                return self._transcribe_stereo(audio_path, language=language)
            except Exception as exc:
                logger.warning(
                    f"Stereo channel isolation transcription failed ({exc}), falling back to mono downmix..."
                )
                try:
                    from faster_whisper import decode_audio

                    mono_audio = decode_audio(audio_path, sampling_rate=16000)
                    with time_block(asr_transcribe_duration_seconds, mode="batch"):
                        return self._transcribe_single_channel(mono_audio, language=language)
                except Exception as exc2:
                    logger.warning(f"Mono downmix fallback also failed ({exc2}), falling back to raw file...")

        with time_block(asr_transcribe_duration_seconds, mode="batch"):
            return self._transcribe_single_channel(audio_path, language=language)

    def transcribe_array(
        self, pcm: np.ndarray, language: str = "tr"
    ) -> tuple[list[TranscriptionSegment], float]:
        """Transcribe an in-memory mono float32 PCM array (16kHz) for live streaming sessions.

        Bypasses stereo-file detection and disk I/O entirely — the caller (a
        StreamingASRSession) is responsible for supplying a bounded, already-decoded window.
        """
        language = lock_language(language)
        if self._model is None and not getattr(self, "_is_mlx", False):
            self.load_model()
        if pcm.size == 0:
            return [], 0.0
        with time_block(asr_transcribe_duration_seconds, mode="streaming"):
            return self._transcribe_single_channel(pcm, language=language)
