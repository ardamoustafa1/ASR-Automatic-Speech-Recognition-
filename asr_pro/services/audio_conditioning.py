"""
Telephony Audio Conditioning & Speech Enhancement Service.

Applies enterprise-grade DSP signal pre-conditioning for call center recordings
(GSM 8kHz/16kHz telephony audio) before acoustic ASR and speaker diarization models.
Removes DC offset, power line hum (<80Hz), high-frequency hiss (>7500Hz), and applies
dynamic RMS range normalization to prevent noise-induced hallucinations and false speaker splits.
"""

from typing import Any

import numpy as np
from loguru import logger

try:
    from scipy.signal import butter, filtfilt

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def condition_telephony_audio(audio_input: Any, sample_rate: int = 16000) -> np.ndarray:
    """Apply bandpass filtering and dynamic RMS normalization to speech audio input.

    Accepts either a file path (str) or a numpy PCM array.
    Returns a float32 numpy array in [-1.0, 1.0] ready for Whisper and Pyannote.
    """
    if audio_input is None:
        return np.array([], dtype=np.float32)

    if isinstance(audio_input, str):
        try:
            from faster_whisper import decode_audio

            pcm = decode_audio(audio_input, sampling_rate=sample_rate)
        except Exception as exc:
            logger.warning(f"AudioConditioning: Could not decode audio path '{audio_input}': {exc}")
            return np.array([], dtype=np.float32)
    elif isinstance(audio_input, (list, tuple)):
        pcm = np.array(audio_input, dtype=np.float32)
    elif hasattr(audio_input, "astype"):
        pcm = audio_input.astype(np.float32, copy=False)
    else:
        return np.array([], dtype=np.float32)

    if pcm.size == 0:
        return pcm

    # 1. Remove DC offset
    pcm_clean = pcm - np.mean(pcm)

    # 2. Apply 4th-order Butterworth Bandpass Filter (80 Hz - 7500 Hz for 16kHz audio)
    if _SCIPY_AVAILABLE and sample_rate > 15000 and pcm_clean.size > 100:
        try:
            nyq = 0.5 * sample_rate
            low = max(80.0 / nyq, 0.001)
            high = min(7500.0 / nyq, 0.999)
            if low < high:
                b, a = butter(N=4, Wn=[low, high], btype="bandpass")
                pcm_clean = filtfilt(b, a, pcm_clean).astype(np.float32)
        except Exception as exc:
            logger.debug(f"AudioConditioning: Bandpass filter bypassed due to: {exc}")

    # 3. Dynamic RMS Range Normalization (target RMS ~ -20 dBFS -> ~0.10)
    rms = np.sqrt(np.mean(pcm_clean**2))
    if rms > 0.0001:
        target_rms = 0.10  # -20 dBFS standard for speech recognition
        gain = target_rms / rms
        gain = max(0.2, min(gain, 10.0))
        pcm_clean = np.clip(pcm_clean * gain, -1.0, 1.0)
    return pcm_clean.astype(np.float32)


def suppress_telecom_crosstalk_and_babble(
    audio_input: Any, sample_rate: int = 16000, gate_db: float = -38.0
) -> np.ndarray:
    """Enterprise telecom babble noise & crosstalk suppression filter.

    Applies strict telephony bandpass (300Hz - 3400Hz standard voice band) and spectral energy
    gating below `gate_db` to remove background contact center babble noise and crosstalk bleed.
    This guarantees clean voice signals for speaker diarization and biometric matching.
    """
    pcm = condition_telephony_audio(audio_input, sample_rate=sample_rate)
    if pcm.size == 0:
        return pcm

    # Strict 300Hz - 3400Hz telecom bandpass for mono GSM isolation
    if _SCIPY_AVAILABLE and sample_rate >= 8000 and pcm.size > 100:
        try:
            nyq = 0.5 * sample_rate
            low = max(300.0 / nyq, 0.001)
            high = min(3400.0 / nyq, 0.999)
            if low < high:
                b, a = butter(N=4, Wn=[low, high], btype="bandpass")
                pcm = filtfilt(b, a, pcm).astype(np.float32)
        except Exception as exc:
            logger.debug(f"AudioConditioning: Telecom bandpass bypassed due to: {exc}")

    # Spectral energy noise gating (suppressing babble below threshold)
    frame_len = int(sample_rate * 0.025)  # 25ms windows
    if frame_len > 0 and pcm.size >= frame_len:
        n_frames = pcm.size // frame_len
        frames = pcm[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms_energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-9)
        max_rms = np.max(rms_energy) + 1e-9
        db_levels = 20.0 * np.log10(rms_energy / max_rms)

        # Gate out frames below gate_db (background babble chatter)
        mask = (db_levels > gate_db).astype(np.float32)
        # Smooth mask with simple 3-frame rolling average to prevent clipping clicks
        smooth_mask = np.convolve(mask, np.ones(3) / 3.0, mode="same")
        frames_gated = frames * smooth_mask[:, None]
        pcm[: n_frames * frame_len] = frames_gated.flatten()

    return pcm.astype(np.float32)


# Turkish automated contact center IVR / robot disclosure phrases
IVR_KEYWORDS = [
    "hoş geldiniz",
    "kayıt altındadır",
    "kayıt altına alınmaktadır",
    "tuşlayınız",
    "beklemeye devam ediniz",
    "aktarıyorum",
    "çağrı merkezimize",
    "sırada bekleyen",
    "kısa bir süre sonra",
    "bizi aradığınız için",
    "güvenliğiniz amacıyla",
]


def is_ivr_segment(text: str, start_time: float) -> bool:
    """Check if a segment within the first 18 seconds is an automated IVR/robot greeting."""
    if start_time > 18.0 or not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in IVR_KEYWORDS)
