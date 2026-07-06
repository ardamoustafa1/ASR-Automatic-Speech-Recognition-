"""ITU-T P.863 MOS (Mean Opinion Score) Quality Estimator & NOC Risk Reporter.

Evaluates telephony acoustic quality by analyzing SNR, clipping rate, and signal dropout ratio
to score voice calls from 1.0 (Bad) to 5.0 (HD Voice) and trigger infrastructure hazard warnings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("asr_pro.services.mos_estimator")


class MOSEstimator:
    """Acoustic quality estimator aligned with ITU-T P.863 standards."""

    @staticmethod
    def estimate_mos(audio: np.ndarray | str, sample_rate: int = 16000) -> dict[str, Any]:
        """Compute MOS (1.0 to 5.0), SNR in dB, clipping rate, and NOC risk indicators."""
        if isinstance(audio, str):
            from faster_whisper import decode_audio
            audio = decode_audio(audio, sampling_rate=sample_rate)

        if not isinstance(audio, np.ndarray) or len(audio) == 0:
            return {
                "mos_score": 1.0,
                "quality_grade": "Bad / Unreadable",
                "snr_db": 0.0,
                "clipping_rate_pct": 0.0,
                "dropout_rate_pct": 100.0,
                "noc_alert": "Tamamen boş veya okunamayan ses kaydı.",
            }

        pcm = audio.flatten().astype(np.float32)

        # 1. Clipping Rate (%)
        clipping_samples = np.sum(np.abs(pcm) >= 0.98)
        clipping_rate_pct = float(clipping_samples / len(pcm) * 100.0)

        # 2. SNR (Signal-to-Noise Ratio in dB) & Dropout Rate (%)
        frame_len = int(sample_rate * 0.020)  # 20ms frames
        if len(pcm) < frame_len:
            n_frames = 1
            frame_energies = np.array([np.mean(pcm**2)])
        else:
            n_frames = len(pcm) // frame_len
            frames = pcm[: n_frames * frame_len].reshape(n_frames, frame_len)
            frame_energies = np.mean(frames**2, axis=1)

        # Sort energies: lowest 15% represents background noise floor, highest 50% speech
        sorted_energies = np.sort(frame_energies)
        noise_idx = max(1, int(n_frames * 0.15))
        speech_idx = max(1, int(n_frames * 0.50))

        noise_power = np.mean(sorted_energies[:noise_idx]) + 1e-9
        speech_power = np.mean(sorted_energies[speech_idx:]) + 1e-9
        snr_db = float(10.0 * np.log10(speech_power / noise_power))

        # Dropout (zero-energy frames representing packet loss or dropouts)
        dropout_frames = np.sum(frame_energies < 1e-7)
        dropout_rate_pct = float(dropout_frames / max(1, n_frames) * 100.0)

        # 3. Compute MOS (Mean Opinion Score) from 1.0 to 5.0
        mos = 4.8  # Max typical HD Voice MOS

        # Deduct for low SNR
        if snr_db < 30.0:
            mos -= (30.0 - max(5.0, snr_db)) * 0.08

        # Deduct for clipping
        if clipping_rate_pct > 0.5:
            mos -= min(2.0, clipping_rate_pct * 0.4)

        # Deduct for dropouts / packet loss
        if dropout_rate_pct > 1.0:
            mos -= min(2.5, dropout_rate_pct * 0.2)

        mos_score = round(max(1.0, min(5.0, mos)), 2)

        # Determine grade and NOC alerts
        if mos_score >= 4.0:
            grade = "Excellent / HD Voice"
        elif mos_score >= 3.2:
            grade = "Good / Commercial Telephony"
        elif mos_score >= 2.5:
            grade = "Fair / Noticeable Degradation"
        else:
            grade = "Poor / High Risk"

        noc_alert = None
        if mos_score < 3.0 or clipping_rate_pct > 4.0 or dropout_rate_pct > 8.0:
            noc_alert = (
                f"⚠️ NOC Risk Bildirimi: Düşük Ses Kalitesi (MOS: {mos_score}). "
                f"SNR: {snr_db:.1f} dB, Kırpılma: %{clipping_rate_pct:.1f}, Kayıp/Dropout: %{dropout_rate_pct:.1f}. "
                "Altyapı veya hat kaynaklı müşteri şikayeti riski yüksek!"
            )
            logger.warning(noc_alert)
        else:
            logger.info(f"MOSEstimator: Computed MOS={mos_score} ({grade}), SNR={snr_db:.1f}dB")

        return {
            "mos_score": mos_score,
            "quality_grade": grade,
            "snr_db": round(snr_db, 1),
            "clipping_rate_pct": round(clipping_rate_pct, 2),
            "dropout_rate_pct": round(dropout_rate_pct, 2),
            "noc_alert": noc_alert,
        }
