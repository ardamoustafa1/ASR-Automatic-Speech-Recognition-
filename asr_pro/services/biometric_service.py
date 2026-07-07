"""Biometric Voiceprint Service for acoustic speaker identification and deterministic role assignment.

Extracts a speaker embedding from telephony audio using speechbrain's ECAPA-TDNN
(spkrec-ecapa-voxceleb, 192-dim, trained on VoxCeleb for speaker verification)
and computes cosine similarity against enrolled contact center agent voiceprints.

Falls back to a much weaker hand-rolled 128-dim raw FFT power-spectrum embedding
only if speechbrain/the pretrained model is unavailable (e.g. offline first boot
before the model is cached) - that fallback has no learned discriminative power
and should not be relied on for production agent identification; every
voiceprint records which embedding model produced it (`embedding_model`) so the
two are never compared against each other.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from asr_pro.config import _is_testing, settings
from asr_pro.db.models import AgentVoiceprint
from asr_pro.db.session import SessionLocal

logger = logging.getLogger("asr_pro.services.biometric_service")

ECAPA_MODEL_NAME = "ecapa-tdnn"
LEGACY_MODEL_NAME = "fft-legacy-v1"


class _EcapaEncoder:
    """Thread-safe lazy singleton wrapping speechbrain's ECAPA-TDNN speaker encoder."""

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
                logger.debug("BiometricService: testing mode, skipping ECAPA-TDNN model load.")
                return None
            try:
                import torch
                from speechbrain.inference.speaker import EncoderClassifier

                # speechbrain's custom ops (e.g. its STFT/conv kernels) are not
                # reliably supported on MPS; CUDA is safe, otherwise CPU. This
                # model is small (~20MB) so CPU inference is fast enough for
                # the short enrollment/match clips it processes (<=3s).
                device = "cuda" if torch.cuda.is_available() else "cpu"
                cls._instance = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="data/models/spkrec-ecapa-voxceleb",
                    run_opts={"device": device},
                )
                logger.info(f"BiometricService: ECAPA-TDNN speaker encoder loaded on {device}.")
            except Exception as exc:
                logger.warning(
                    f"BiometricService: could not load ECAPA-TDNN ({exc}). Falling back to "
                    "legacy raw-FFT voiceprints - these have materially weaker speaker "
                    "discrimination and should not be trusted for production agent ID."
                )
                cls._instance = None
        return cls._instance


class BiometricService:
    """Service for managing agent voiceprints and matching speaker acoustics."""

    def __init__(self, db_session: Session | None = None) -> None:
        self._db = db_session

    def _get_session(self) -> Session:
        if self._db is not None:
            return self._db
        return SessionLocal()

    @staticmethod
    def _decode(audio: np.ndarray | str, sample_rate: int) -> np.ndarray:
        if isinstance(audio, str):
            from asr_pro.services.audio_conditioning import condition_telephony_audio

            return condition_telephony_audio(audio, sample_rate=sample_rate)
        if not isinstance(audio, np.ndarray):
            return np.array([], dtype=np.float32)
        pcm = audio.flatten().astype(np.float32)
        if np.max(np.abs(pcm)) > 0:
            pcm = pcm / np.max(np.abs(pcm))
        return pcm

    @staticmethod
    def extract_voiceprint(
        audio: np.ndarray | str, sample_rate: int = 16000
    ) -> tuple[list[float], str]:
        """Extract an L2-normalized speaker embedding vector (voiceprint) and its model tag.

        Returns (embedding, embedding_model) where embedding_model is
        "ecapa-tdnn" (192-dim, learned speaker-discriminative embedding) or
        "fft-legacy-v1" (128-dim raw spectral fallback, weak discrimination).
        """
        pcm = BiometricService._decode(audio, sample_rate)
        if pcm.size == 0:
            return [0.0] * 128, LEGACY_MODEL_NAME

        encoder = _EcapaEncoder.get()
        if encoder is not None:
            try:
                import torch

                waveform = torch.from_numpy(pcm).float().unsqueeze(0)  # (1, time)
                with torch.no_grad():
                    emb = encoder.encode_batch(waveform).squeeze().cpu().numpy()
                emb = emb.astype(np.float32).flatten()
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
                return [float(x) for x in emb], ECAPA_MODEL_NAME
            except Exception as exc:
                logger.warning(
                    f"BiometricService: ECAPA-TDNN inference failed ({exc}), using legacy fallback."
                )

        return BiometricService._extract_legacy_fft_voiceprint(pcm), LEGACY_MODEL_NAME

    @staticmethod
    def _extract_legacy_fft_voiceprint(pcm: np.ndarray) -> list[float]:
        """Legacy 128-dim raw FFT power-spectrum embedding (weak fallback only)."""
        sample_rate = 16000
        frame_len = int(sample_rate * 0.025)  # 25ms frame
        hop_len = int(sample_rate * 0.010)  # 10ms hop

        if len(pcm) < frame_len:
            padded = np.zeros(frame_len, dtype=np.float32)
            padded[: len(pcm)] = pcm
            pcm = padded

        n_frames = max(1, (len(pcm) - frame_len) // hop_len + 1)
        n_frames = min(n_frames, 300)  # cap at ~3 seconds of speech

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

        log_spec = np.log1p(spectral_acc)
        norm = np.linalg.norm(log_spec)
        if norm > 1e-6:
            embedding = log_spec / norm
        else:
            embedding = np.zeros(128, dtype=np.float32)

        return [float(x) for x in embedding]

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two embedding vectors.

        Returns 0.0 for mismatched dimensions (e.g. comparing an ecapa-tdnn
        embedding against a fft-legacy-v1 one) rather than raising, since the
        comparison is meaningless either way.
        """
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
        self, agent_code: str, agent_name: str, audio_path_or_array: np.ndarray | str
    ) -> AgentVoiceprint | None:
        """Enroll or update an agent's voiceprint in the database.

        Returns the persisted AgentVoiceprint record on success, or None if no
        usable voiceprint could be extracted from the audio (e.g. silence).
        """
        embedding, embedding_model = self.extract_voiceprint(audio_path_or_array)
        if not any(embedding):
            logger.error(f"BiometricService: extracted an all-zero voiceprint for {agent_code}.")
            return None

        session = self._get_session()
        close_after = self._db is None
        try:
            stmt = select(AgentVoiceprint).where(AgentVoiceprint.agent_code == agent_code)
            record = session.scalar(stmt)
            if record:
                record.agent_name = agent_name
                record.embedding_json = embedding
                record.embedding_model = embedding_model
                record.is_active = True
                logger.info(
                    f"BiometricService: Updated voiceprint for {agent_name} ({agent_code}) "
                    f"using {embedding_model}."
                )
            else:
                record = AgentVoiceprint(
                    agent_code=agent_code,
                    agent_name=agent_name,
                    embedding_json=embedding,
                    embedding_model=embedding_model,
                    is_active=True,
                )
                session.add(record)
                logger.info(
                    f"BiometricService: Enrolled new voiceprint for {agent_name} ({agent_code}) "
                    f"using {embedding_model}."
                )

            session.commit()
            session.refresh(record)
            return record
        except Exception as exc:
            session.rollback()
            logger.error(f"BiometricService enroll error: {exc}")
            raise
        finally:
            if close_after:
                session.close()

    def list_voiceprints(self) -> list[AgentVoiceprint]:
        """List all enrolled agent voiceprints (most recently created first)."""
        session = self._get_session()
        close_after = self._db is None
        try:
            stmt = select(AgentVoiceprint).order_by(AgentVoiceprint.created_at.desc())
            return list(session.scalars(stmt).all())
        finally:
            if close_after:
                session.close()

    def match_speaker(
        self, audio_path_or_array: np.ndarray | str, threshold: float | None = None
    ) -> tuple[str | None, float, str | None]:
        """Match audio against all enrolled active agent voiceprints.

        Only compares against voiceprints sharing the same embedding_model as
        the freshly extracted query embedding - an ecapa-tdnn query should
        never be scored against a legacy fft embedding, even incidentally.

        Returns:
            tuple of (matched_agent_code, similarity_score, matched_agent_name).
            If no match exceeds threshold, returns (None, best_score, None).
        """
        if threshold is None:
            threshold = settings.biometric_match_threshold
        query_embedding, query_model = self.extract_voiceprint(audio_path_or_array)

        session = self._get_session()
        close_after = self._db is None
        try:
            stmt = select(AgentVoiceprint).where(AgentVoiceprint.is_active == True)  # noqa: E712
            records = session.scalars(stmt).all()

            best_score = 0.0
            best_code = None
            best_name = None
            skipped_mismatched = 0

            for rec in records:
                if not rec.embedding_json:
                    continue
                if getattr(rec, "embedding_model", LEGACY_MODEL_NAME) != query_model:
                    skipped_mismatched += 1
                    continue
                score = self.cosine_similarity(query_embedding, rec.embedding_json)
                if score > best_score:
                    best_score = score
                    best_code = rec.agent_code
                    best_name = rec.agent_name

            if skipped_mismatched:
                logger.debug(
                    f"BiometricService: skipped {skipped_mismatched} voiceprint(s) enrolled "
                    f"under a different embedding model than {query_model!r}."
                )

            if best_score >= threshold and best_code is not None:
                logger.info(
                    f"BiometricService: Matched speaker as {best_name} ({best_code}) with score {best_score:.3f}"
                )
                return best_code, best_score, best_name

            return None, best_score, None
        finally:
            if close_after:
                session.close()
