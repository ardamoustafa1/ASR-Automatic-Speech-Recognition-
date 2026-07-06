"""
Unit tests for Vodafone Enterprise SOTA Features:
1. Telephony audio conditioning and bandpass filtering
2. IVR / Robot speech detection
3. Semantic + Acoustic dual-guard role auto-correction
4. Simultaneous speech (interruption) detection
"""
import numpy as np
import pytest
from asr_pro.services.audio_conditioning import condition_telephony_audio, is_ivr_segment
from asr_pro.core.semantic_role_guard import enforce_semantic_role_guard
from asr_pro.core.keyword_engine import SegmentInput


def test_audio_conditioning_numpy():
    """Verify telephony audio conditioning normalizes DC offset and RMS amplitude."""
    # Create synthetic noisy signal with DC offset
    raw_pcm = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32) * 0.01 + 0.5
    clean_pcm = condition_telephony_audio(raw_pcm, sample_rate=16000)
    
    assert len(clean_pcm) == 16000
    assert clean_pcm.dtype == np.float32
    # DC offset should be removed (~0 mean)
    assert abs(np.mean(clean_pcm)) < 1e-4
    # RMS should be boosted towards target ~0.10
    rms = np.sqrt(np.mean(clean_pcm**2))
    assert rms > 0.05 and rms <= 1.0


def test_ivr_segment_detection():
    """Verify IVR/robot speech detection identifies telecom greetings within first 18 seconds."""
    assert is_ivr_segment("Vodafone'a hoş geldiniz, sesiniz kayıt altındadır.", 2.5) is True
    assert is_ivr_segment("Lütfen beklemeye devam ediniz, müşteri temsilcisine aktarıyorum.", 12.0) is True
    # Should not trigger after 18 seconds even if phrase matches
    assert is_ivr_segment("Vodafone'a hoş geldiniz", 25.0) is False
    assert is_ivr_segment("Merhaba ben Arda, nasıl yardımcı olabilirim?", 5.0) is False


def test_semantic_role_guard_and_interruption():
    """Verify semantic role auto-correction flips roles and flags interruption moments."""
    segments = [
        SegmentInput(start=0.0, end=4.0, text="Merhaba, Vodafone müşteri hizmetleri ben Arda, nasıl yardımcı olabilirim?", speaker="SPEAKER_01"), # Wrong assigned to customer
        SegmentInput(start=3.5, end=7.0, text="Faturam çok yüksek geldi, iptal ettirmek istiyorum şikayetçiyim!", speaker="SPEAKER_00"), # Overlap interruption + wrong assigned to agent
        SegmentInput(start=8.0, end=10.0, text="Sistemimize bakıyorum hemen kontrol ediyorum efendim.", speaker="SPEAKER_00"), # Correct agent
    ]
    
    refined = enforce_semantic_role_guard(segments, agent_id="SPEAKER_00", customer_id="SPEAKER_01")
    
    assert len(refined) == 3
    # First segment should be auto-corrected to agent (SPEAKER_00)
    assert refined[0].speaker == "SPEAKER_00"
    assert refined[0].auto_corrected is True
    assert refined[0].is_interruption is False
    
    # Second segment should be auto-corrected to customer (SPEAKER_01) AND flagged as interruption
    assert refined[1].speaker == "SPEAKER_01"
    assert refined[1].auto_corrected is True
    assert refined[1].is_interruption is True
    
    # Third segment is already correct agent, no changes
    assert refined[2].speaker == "SPEAKER_00"
    assert refined[2].auto_corrected is False
    assert refined[2].is_interruption is False
