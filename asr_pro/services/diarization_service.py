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

            try:
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=token,
                )
            except TypeError:
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
            # Smart heuristic alignment when no pyannote model is available.
            # Strategy: cluster segments into "conversation turns" by detecting pauses,
            # then alternate speakers per turn (not per segment).
            MIN_PAUSE_FOR_TURN = 0.6  # seconds between turns
            turn_groups: list[list[int]] = []
            current_group: list[int] = [0]

            for idx in range(1, len(segments)):
                pause = segments[idx].start - segments[idx - 1].end
                if pause > MIN_PAUSE_FOR_TURN:
                    turn_groups.append(current_group)
                    current_group = [idx]
                else:
                    current_group.append(idx)
            if current_group:
                turn_groups.append(current_group)

            # Assign alternating speakers per turn group
            speaker_seq = ["SPEAKER_00", "SPEAKER_01"]
            for turn_idx, group in enumerate(turn_groups):
                spk = speaker_seq[turn_idx % 2]
                for seg_idx in group:
                    seg = segments[seg_idx]
                    seg_spk = seg.speaker or spk
                    speakers_present.add(seg_spk)
                    aligned_segments.append(
                        SegmentInput(
                            start=seg.start,
                            end=seg.end,
                            text=seg.text,
                            speaker=seg_spk,
                            segment_index=seg.segment_index,
                        )
                    )

        agent_id, customer_id = self._identify_roles(aligned_segments, sorted(speakers_present))
        return aligned_segments, agent_id, customer_id

    def _identify_roles(
        self, segments: list[SegmentInput], speakers: list[str]
    ) -> tuple[str | None, str | None]:
        """Intelligently determine which speaker is the Agent and which is the Customer.

        Uses multiple heuristics:
        - Agent keywords (greeting/support phrases in Turkish)
        - Customer keywords (complaint/question phrases in Turkish)
        - Longer sentences → more likely agent (agents explain, customers complain)
        - More question marks → more likely customer
        - First speaker bonus (agent usually greets first)
        """
        if not speakers:
            return None, None
        if len(speakers) == 1:
            return speakers[0], None

        # Turkish contact center agent phrases
        agent_keywords = [
            "hoş geldiniz",
            "nasıl yardımcı",
            "yardımcı olabilirim",
            "yardımcı olacağım",
            "müşteri hizmetleri",
            "benim adım",
            "iyi günler",
            "iyi akşamlar",
            "kayıt altına alınmaktadır",
            "kontrol ediyorum",
            "kontrol edelim",
            "anlayışınız için",
            "teşekkür ederiz",
            "teşekkür ederim",
            "sistemimize bakıyorum",
            "hesabınıza bakıyorum",
            "baktım",
            "yapabilirim",
            "yaptım",
            "işleminiz",
            "tarafınıza",
            "sizi anlıyorum",
            "haklısınız",
            "not aldım",
            "aktarıyorum",
            "yönlendiriyorum",
            "bağlıyorum",
        ]

        # Turkish customer complaint/question phrases
        customer_keywords = [
            "şikayet",
            "sorunum var",
            "olmuyor",
            "çalışmıyor",
            "bozuk",
            "neden",
            "niye",
            "nasıl",
            "ne zaman",
            "ne oldu",
            "hata",
            "ücret",
            "para",
            "fatura",
            "ödeme",
            "çekti",
            "kesildi",
            "iade",
            "iptal",
            "gelmedi",
            "gönderilmedi",
            "gecikti",
            "memnun değilim",
            "kötü",
            "berbat",
            "rezalet",
            "bir türlü",
            "hâlâ",
            "yine",
        ]

        speaker_scores: dict[str, int] = dict.fromkeys(speakers, 0)
        speaker_word_counts: dict[str, int] = dict.fromkeys(speakers, 0)
        speaker_question_counts: dict[str, int] = dict.fromkeys(speakers, 0)
        speaker_seg_counts: dict[str, int] = dict.fromkeys(speakers, 0)

        # First segment speaker gets a greeting bonus (agents usually speak first)
        if segments and segments[0].speaker in speaker_scores:
            speaker_scores[segments[0].speaker] += 4

        for seg in segments:
            if not seg.speaker or not seg.text:
                continue
            spk = seg.speaker
            text_lower = seg.text.lower().strip()
            words = text_lower.split()
            speaker_word_counts[spk] = speaker_word_counts.get(spk, 0) + len(words)
            speaker_seg_counts[spk] = speaker_seg_counts.get(spk, 0) + 1

            # Count question marks (customers ask more questions)
            speaker_question_counts[spk] = speaker_question_counts.get(spk, 0) + text_lower.count(
                "?"
            )

            for kw in agent_keywords:
                if kw in text_lower:
                    speaker_scores[spk] = speaker_scores.get(spk, 0) + 5

            for kw in customer_keywords:
                if kw in text_lower:
                    # Customer keywords REDUCE the agent score for this speaker
                    speaker_scores[spk] = speaker_scores.get(spk, 0) - 3

        # Agents typically talk MORE words per segment (they explain, clarify)
        # Give bonus to the speaker with longer average sentences
        for spk in speakers:
            seg_count = speaker_seg_counts.get(spk, 1) or 1
            avg_words = speaker_word_counts.get(spk, 0) / seg_count
            if avg_words > 12:  # long explanatory sentences → likely agent
                speaker_scores[spk] = speaker_scores.get(spk, 0) + 3
            # More questions → likely customer
            q_count = speaker_question_counts.get(spk, 0)
            speaker_scores[spk] = speaker_scores.get(spk, 0) - (q_count * 2)

        logger.debug(f"Speaker scores for role identification: {speaker_scores}")

        # Agent is the speaker with highest agent score
        sorted_speakers = sorted(speakers, key=lambda s: speaker_scores.get(s, 0), reverse=True)
        agent_id = sorted_speakers[0]
        customer_id = sorted_speakers[1] if len(sorted_speakers) > 1 else None

        logger.debug(f"Role identification -> Agent: {agent_id}, Customer: {customer_id}")
        return agent_id, customer_id
