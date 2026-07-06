"""Biometric Voiceprint Service for acoustic speaker identification and deterministic role assignment.

Extracts robust 128-dimensional spectral envelope signatures (formant energy profile) from telephony
audio and computes Cosine Similarity against enrolled contact center agent voiceprints.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from asr_pro.db.models import AgentVoiceprint
from asr_pro.db.session import SessionLocal

logger = logging.getLogger("asr_pro.services.biometric_service")


class BiometricService:
    """Service for managing agent voiceprints and matching speaker acoustics."""

    def __init__(self, db_session: Session | None = None) -> None:
        self._db = db_session

    def _get_session(self) -> Session:
        if self._db is not None:
            return self._db
        return SessionLocal()

    @staticmethod
    def extract_voiceprint(audio: np.ndarray | str, sample_rate: int = 16000) -> list[float]:
        """Extract a 128-dimensional L2-normalized acoustic spectral embedding vector (voiceprint).

        Computes the Log-Mel/FFT energy spectrum distribution representing vocal tract formants.
        """
        if isinstance(audio, str):
            from faster_whisper import decode_audio

            audio = decode_audio(audio, sampling_rate=sample_rate)

        if not isinstance(audio, np.ndarray) or len(audio) == 0:
            return [0.0] * 128

        # Ensure 1D float32
        pcm = audio.flatten().astype(np.float32)
        if np.max(np.abs(pcm)) > 0:
            pcm = pcm / np.max(np.abs(pcm))

        # Compute Power Spectrum via FFT across sliding frames
        frame_len = int(sample_rate * 0.025)  # 25ms frame
        hop_len = int(sample_rate * 0.010)  # 10ms hop

        if len(pcm) < frame_len:
            padded = np.zeros(frame_len, dtype=np.float32)
            padded[: len(pcm)] = pcm
            pcm = padded

        n_frames = max(1, (len(pcm) - frame_len) // hop_len + 1)
        # Limit to first 300 frames (~3 seconds of speech) for efficiency and stability
        n_frames = min(n_frames, 300)

        # Compute 128-bin logarithmic frequency energy distribution
        n_fft = 256  # yields 128 positive frequency bins
        window = np.hanning(n_fft)

        spectral_acc = np.zeros(128, dtype=np.float32)
        for i in range(n_frames):
            start_idx = i * hop_len
            slice_data = pcm[start_idx : start_idx + n_fft]
            if len(slice_data) < n_fft:
                padded_slice = np.zeros(n_fft, dtype=np.float32)
                padded_slice[: len(slice_data)] = slice_data
                slice_data = padded_slice

            fft_vals = np.fft.rfft(slice_data * window, n=n_fft)
            power_spec = np.abs(fft_vals)[:128] ** 2
            spectral_acc += power_spec

        # Apply log compression and L2 normalization
        log_spec = np.log1p(spectral_acc)
        norm = np.linalg.norm(log_spec)
        if norm > 1e-6:
            embedding = log_spec / norm
        else:
            embedding = np.zeros(128, dtype=np.float32)

        return [float(x) for x in embedding]

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        if len(vec_a) != len(vec_b) or len(vec_a) == 0:
            return 0.0
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def enroll_agent(
        self, agent_code: str, agent_name: str, audio: np.ndarray | str
    ) -> dict[str, Any]:
        """Enroll or update an agent's voiceprint in the database."""
        embedding = self.extract_voiceprint(audio)

        session = self._get_session()
        close_after = self._db is None
        try:
            stmt = select(AgentVoiceprint).where(AgentVoiceprint.agent_code == agent_code)
            record = session.scalar(stmt)
            if record:
                record.agent_name = agent_name
                record.embedding_json = embedding
                record.is_active = True
                logger.info(
                    f"BiometricService: Updated voiceprint for {agent_name} ({agent_code})."
                )
            else:
                record = AgentVoiceprint(
                    agent_code=agent_code,
                    agent_name=agent_name,
                    embedding_json=embedding,
                    is_active=True,
                )
                session.add(record)
                logger.info(
                    f"BiometricService: Enrolled new voiceprint for {agent_name} ({agent_code})."
                )

            session.commit()
            return {
                "status": "success",
                "agent_code": agent_code,
                "agent_name": agent_name,
                "embedding_dimensions": len(embedding),
            }
        except Exception as exc:
            session.rollback()
            logger.error(f"BiometricService enroll error: {exc}")
            raise
        finally:
            if close_after:
                session.close()

    def match_speaker(
        self, audio: np.ndarray | str, threshold: float = 0.85
    ) -> tuple[str | None, float, str | None]:
        """Match audio against all enrolled active agent voiceprints.

        Returns:
            tuple of (matched_agent_code, similarity_score, matched_agent_name).
            If no match exceeds threshold, returns (None, best_score, None).
        """
        query_embedding = self.extract_voiceprint(audio)

        session = self._get_session()
        close_after = self._db is None
        try:
            stmt = select(AgentVoiceprint).where(AgentVoiceprint.is_active == True)  # noqa: E712
            records = session.scalars(stmt).all()

            best_score = 0.0
            best_code = None
            best_name = None

            for rec in records:
                if not rec.embedding_json:
                    continue
                score = self.cosine_similarity(query_embedding, rec.embedding_json)
                if score > best_score:
                    best_score = score
                    best_code = rec.agent_code
                    best_name = rec.agent_name

            if best_score >= threshold and best_code is not None:
                logger.info(
                    f"BiometricService: Matched speaker as {best_name} ({best_code}) with score {best_score:.3f}"
                )
                return best_code, best_score, best_name

            return None, best_score, None
        finally:
            if close_after:
                session.close()
