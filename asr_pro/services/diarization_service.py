from __future__ import annotations

"""Thread-safe Singleton Speaker Diarization & Role Assignment Service using pyannote.audio.

Architected for Apple Silicon MPS acceleration, automatic speaker turn alignment,
and intelligent contact center Agent vs. Customer role identification.
"""

import os
import platform
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from asr_pro.config import _is_testing, settings
from asr_pro.core.keyword_engine import SegmentInput

try:
    from pyannote.audio import Pipeline
except ImportError:
    Pipeline = None  # type: ignore[assignment]

warnings.simplefilter("ignore")


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class DiarizationService:
    """Thread-safe Singleton for speaker diarization and role assignment."""

    _instance: DiarizationService | None = None
    _lock = threading.Lock()
    _pipeline = None

    @classmethod
    def get_instance(cls) -> DiarizationService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._device_str = self._choose_device()
        self._use_mps = self._device_str == "mps"

    def _choose_device(self) -> str:
        if _is_testing:
            return "cpu"
        try:
            import torch

            if platform.system() == "Darwin" and platform.machine() in ["arm64", "aarch64"]:
                if torch.backends.mps.is_available():
                    logger.info(
                        "Diarization: Apple Silicon MPS detected. Enabling Metal acceleration."
                    )
                    return "mps"
            if torch.cuda.is_available():
                logger.info("Diarization: CUDA GPU detected. Enabling GPU acceleration.")
                return "cuda"
        except Exception:
            pass
        logger.info("Diarization: Running on CPU.")
        return "cpu"

    def load_pipeline(self) -> Any:
        """Lazy load the pyannote speaker diarization pipeline."""
        if self._pipeline is not None:
            return self._pipeline
        if _is_testing:
            logger.debug("Diarization: In testing mode, skipping pyannote model load.")
            return None
        if Pipeline is None:
            logger.warning(
                "pyannote.audio not installed. Diarization will use heuristic fallbacks."
            )
            return None

        token = settings.hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            logger.warning(
                "No Hugging Face token found. Diarization pipeline will fallback to heuristic mode."
            )
            return None

        logger.info("Loading pyannote/speaker-diarization-3.1 pipeline into memory...")
        try:
            import torch

            # Load pipeline with authentication
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
            if self._pipeline is not None and self._device_str in ("cuda", "mps"):
                try:
                    self._pipeline.to(torch.device(self._device_str))
                    logger.info(f"pyannote pipeline moved to device: {self._device_str}")
                except Exception as exc:
                    logger.warning(
                        f"Could not move pyannote pipeline to {self._device_str}: {exc}. Using default device."
                    )
            logger.info("Speaker diarization pipeline loaded successfully.")
            return self._pipeline
        except Exception as exc:
            logger.warning(
                f"Failed to load pyannote diarization pipeline: {exc}. Using heuristics."
            )
            self._pipeline = None
            return None

    def diarize(self, audio_path: str) -> list[SpeakerTurn]:
        """Perform speaker diarization on an audio file and return speaker turns."""
        if not audio_path or not os.path.exists(audio_path):
            return []

        pipeline = self.load_pipeline()
        if pipeline is None:
            return []

        logger.info(f"Running acoustic speaker diarization on: {audio_path}")
        try:
            diarization = pipeline(audio_path)
            turns: list[SpeakerTurn] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append(
                    SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
                )
            logger.info(
                f"Diarization complete. Found {len(turns)} speech turns across {len({t.speaker for t in turns})} speakers."
            )
            return turns
        except Exception as exc:
            logger.error(f"Error during pyannote diarization: {exc}")
            return []

    def assign_speakers_to_segments(
        self,
        segments_data: Sequence[Any],
        audio_path: str | None = None,
    ) -> tuple[list[SegmentInput], str | None, str | None]:
        """Align acoustic speaker turns with Whisper text segments and identify Agent vs Customer.

        Returns:
            Tuple of (aligned_segments, agent_speaker_id, customer_speaker_id)
        """
        # Convert any input format to SegmentInput
        segments: list[SegmentInput] = []
        for idx, seg in enumerate(segments_data):
            if isinstance(seg, SegmentInput):
                segments.append(seg)
            else:
                segments.append(
                    SegmentInput(
                        start=float(getattr(seg, "start", 0)),
                        end=float(getattr(seg, "end", 0)),
                        text=str(getattr(seg, "text", "")),
                        speaker=getattr(seg, "speaker", None),
                        segment_index=idx,
                    )
                )

        if not segments:
            return [], None, None

        turns = self.diarize(audio_path) if audio_path else []

        aligned_segments: list[SegmentInput] = []
        speakers_present = set()

        if turns:
            # Acoustic alignment: find overlapping speaker turn for each segment
            for seg in segments:
                best_speaker = seg.speaker
                max_overlap = 0.0
                for turn in turns:
                    overlap_start = max(seg.start, turn.start)
                    overlap_end = min(seg.end, turn.end)
                    overlap_duration = max(0.0, overlap_end - overlap_start)
                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        best_speaker = turn.speaker

                if not best_speaker:
                    best_speaker = turns[0].speaker if turns else "SPEAKER_00"

                speakers_present.add(best_speaker)
                aligned_segments.append(
                    SegmentInput(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        speaker=best_speaker,
                        segment_index=seg.segment_index,
                    )
                )
        else:
            # Heuristic alignment (fallback when no audio file or model skipped in test mode)
            current_speaker = "SPEAKER_00"
            for idx, seg in enumerate(segments):
                spk = seg.speaker
                if not spk:
                    # Alternating turn detection based on pause duration (> 1.2s silence implies turn change)
                    if idx > 0 and (seg.start - segments[idx - 1].end) > 1.2:
                        current_speaker = (
                            "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
                        )
                    spk = current_speaker
                speakers_present.add(spk)
                aligned_segments.append(
                    SegmentInput(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        speaker=spk,
                        segment_index=seg.segment_index,
                    )
                )

        agent_id, customer_id = self._identify_roles(aligned_segments, sorted(speakers_present))
        return aligned_segments, agent_id, customer_id

    def _identify_roles(
        self, segments: list[SegmentInput], speakers: list[str]
    ) -> tuple[str | None, str | None]:
        """Intelligently determine which speaker is the Agent and which is the Customer."""
        if not speakers:
            return None, None
        if len(speakers) == 1:
            return speakers[0], None

        agent_keywords = [
            "hoş geldiniz",
            "nasıl yardımcı",
            "yardımcı olacağım",
            "müşteri hizmetleri",
            "benim adım",
            "ben arda",
            "iyi günler dilerim",
            "kayıt altına alınmaktadır",
            "kontrol ediyorum",
            "anlayışınız için",
        ]

        speaker_scores: dict[str, int] = dict.fromkeys(speakers, 0)

        # First segment speaker gets a strong greeting bonus
        if segments and segments[0].speaker in speaker_scores:
            speaker_scores[segments[0].speaker] += 3

        for seg in segments:
            if not seg.speaker or not seg.text:
                continue
            text_lower = seg.text.lower()
            for kw in agent_keywords:
                if kw in text_lower:
                    speaker_scores[seg.speaker] += 5

        # Agent is the speaker with highest greeting/support score
        sorted_speakers = sorted(speakers, key=lambda s: speaker_scores.get(s, 0), reverse=True)
        agent_id = sorted_speakers[0]
        customer_id = sorted_speakers[1] if len(sorted_speakers) > 1 else None

        logger.debug(f"Role identification -> Agent: {agent_id}, Customer: {customer_id}")
        return agent_id, customer_id
