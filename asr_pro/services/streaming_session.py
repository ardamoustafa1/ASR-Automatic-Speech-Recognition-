# Per-connection live ASR session: bounded pending-audio window + VAD endpointing,
# turning batch-only Whisper inference into an incremental partial/final protocol.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from asr_pro.config import (
    STREAMING_MAX_PENDING_SEC,
    STREAMING_MIN_PARTIAL_INTERVAL_SEC,
    STREAMING_SILENCE_COMMIT_SEC,
)
from asr_pro.services.asr_service import ASRService
from asr_pro.services.audio_stream_decoder import AudioStreamDecoder
from asr_pro.services.vad_service import VADService

SAMPLE_RATE = 16000
# Grace period before we allow "no speech detected" to drop a still-growing
# buffer — avoids discarding real speech that hasn't crossed Silero's own
# min_speech_duration_ms gate yet.
_SILENCE_DROP_GRACE_SEC = 1.5
# RMS energy fallback threshold when the neural VAD model isn't loaded.
_ENERGY_SILENCE_THRESHOLD = 0.008


@dataclass
class StreamingASRSession:
    language: str = "tr"
    decoder: AudioStreamDecoder = field(default_factory=AudioStreamDecoder)
    pcm_buffer: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    committed_offset_sec: float = 0.0
    committed_text: str = ""
    committed_segments: list = field(default_factory=list)
    _sec_since_last_transcribe: float = 0.0
    _started: bool = False

    async def start(self) -> None:
        await self.decoder.start()
        self._started = True

    async def close(self) -> None:
        if self._started:
            await self.decoder.close()

    def _buffer_duration_sec(self) -> float:
        return float(len(self.pcm_buffer)) / SAMPLE_RATE

    def _is_energy_silent(self, pcm: np.ndarray) -> bool:
        if pcm.size == 0:
            return True
        return float(np.sqrt(np.mean(np.square(pcm)))) < _ENERGY_SILENCE_THRESHOLD

    async def push_audio(self, chunk: bytes) -> dict[str, Any] | None:
        """Feed a raw WebM/Opus chunk. Returns a partial/final message dict, or None."""
        await self.decoder.write(chunk)
        new_pcm = await self.decoder.read_available()
        if new_pcm.size:
            self.pcm_buffer = np.concatenate([self.pcm_buffer, new_pcm])
            self._sec_since_last_transcribe += new_pcm.size / SAMPLE_RATE

        if self.pcm_buffer.size == 0:
            return None
        if self._sec_since_last_transcribe < STREAMING_MIN_PARTIAL_INTERVAL_SEC:
            return None

        vad = VADService.get_instance()
        vad_available = bool(vad.loaded)
        speech_ts = vad.filter_speech_timestamps(self.pcm_buffer, sampling_rate=SAMPLE_RATE) if vad_available else []

        buffer_duration = self._buffer_duration_sec()
        forced = buffer_duration >= STREAMING_MAX_PENDING_SEC

        if vad_available and not speech_ts:
            if buffer_duration > _SILENCE_DROP_GRACE_SEC or forced:
                logger.debug("StreamingASRSession: confirmed silence, dropping pending buffer.")
                self.pcm_buffer = np.empty(0, dtype=np.float32)
                self._sec_since_last_transcribe = 0.0
            return None
        if not vad_available and self._is_energy_silent(self.pcm_buffer):
            if buffer_duration > _SILENCE_DROP_GRACE_SEC or forced:
                self.pcm_buffer = np.empty(0, dtype=np.float32)
                self._sec_since_last_transcribe = 0.0
            return None

        self._sec_since_last_transcribe = 0.0
        segments, _duration = ASRService.get_instance().transcribe_array(
            self.pcm_buffer, language=self.language
        )
        if not segments:
            return None

        trailing_silence = 0.0
        last_speech_end_sample = len(self.pcm_buffer)
        if vad_available and speech_ts:
            last_speech_end_sample = int(speech_ts[-1]["end"])
            trailing_silence = buffer_duration - (last_speech_end_sample / SAMPLE_RATE)

        is_final = forced or trailing_silence >= STREAMING_SILENCE_COMMIT_SEC

        out_segments = [
            {
                "start": round(self.committed_offset_sec + s.start, 2),
                "end": round(self.committed_offset_sec + s.end, 2),
                "text": s.text,
                "speaker": s.speaker,
            }
            for s in segments
        ]
        text = " ".join(s["text"] for s in out_segments if s["text"]).strip()
        if not text:
            return None

        if not is_final:
            return {"type": "partial", "text": text, "segments": out_segments}

        commit_sample = last_speech_end_sample if last_speech_end_sample > 0 else len(self.pcm_buffer)
        commit_sample = min(commit_sample, len(self.pcm_buffer))
        self.pcm_buffer = self.pcm_buffer[commit_sample:]
        self.committed_offset_sec += commit_sample / SAMPLE_RATE
        self.committed_text = (self.committed_text + " " + text).strip()
        self.committed_segments.extend(out_segments)

        return {
            "type": "final",
            "text": text,
            "segments": out_segments,
            "transcript_so_far": self.committed_text,
        }

    async def flush_final(self) -> dict[str, Any] | None:
        """Force-commit whatever remains in the buffer, e.g. on disconnect."""
        if self.pcm_buffer.size == 0:
            return None
        segments, _duration = ASRService.get_instance().transcribe_array(
            self.pcm_buffer, language=self.language
        )
        out_segments = [
            {
                "start": round(self.committed_offset_sec + s.start, 2),
                "end": round(self.committed_offset_sec + s.end, 2),
                "text": s.text,
                "speaker": s.speaker,
            }
            for s in segments
        ]
        text = " ".join(s["text"] for s in out_segments if s["text"]).strip()
        self.committed_offset_sec += self._buffer_duration_sec()
        self.pcm_buffer = np.empty(0, dtype=np.float32)
        if not text:
            return None
        self.committed_text = (self.committed_text + " " + text).strip()
        self.committed_segments.extend(out_segments)
        return {
            "type": "final",
            "text": text,
            "segments": out_segments,
            "transcript_so_far": self.committed_text,
        }
