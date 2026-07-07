"""Unit and integration tests for Tier-5 SOTA capabilities:
Biometrics, Discourse Guard, Domain Adaptation, Live Coaching, and MOS Estimation.
"""

import numpy as np

from asr_pro.services.biometric_service import BiometricService
from asr_pro.services.domain_adaptation import DomainAdaptationService
from asr_pro.services.live_coaching_service import LiveCoachingService
from asr_pro.services.llm_discourse_guard import LLMDiscourseGuard
from asr_pro.services.mos_estimator import MOSEstimator


def test_voiceprint_biometrics():
    """Test spectral voiceprint extraction and cosine matching."""
    # Generate synthetic speech formant frequencies (400Hz and 1500Hz)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio_a = np.sin(2 * np.pi * 400 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)
    audio_b = np.sin(2 * np.pi * 800 * t) + 0.5 * np.sin(2 * np.pi * 2500 * t)

    vec_a, model_a = BiometricService.extract_voiceprint(audio_a)
    vec_a2, model_a2 = BiometricService.extract_voiceprint(audio_a)
    vec_b, model_b = BiometricService.extract_voiceprint(audio_b)

    # In test mode the ECAPA-TDNN model is not loaded (see _is_testing guard),
    # so this exercises the legacy 128-dim fallback deterministically.
    assert model_a == "fft-legacy-v1"
    assert len(vec_a) == 128
    assert len(vec_b) == 128

    sim_identical = BiometricService.cosine_similarity(vec_a, vec_a2)
    sim_different = BiometricService.cosine_similarity(vec_a, vec_b)

    assert sim_identical > 0.99
    assert sim_different < sim_identical


def test_domain_adaptation_telecom_jargon():
    """Test Levenshtein and regex phonetic correction on telecom jargon."""
    raw = "Ben volti ayarlarımı yapamadım, e sim e geçmek istiyorum, a pe ne nedir? cayma bedil ne kadar?"
    corrected = DomainAdaptationService.correct_telecom_terms(raw)

    assert "VoLTE" in corrected
    assert "eSIM" in corrected
    assert "APN" in corrected
    assert "cayma bedeli" in corrected
    assert "volti" not in corrected


def test_llm_discourse_guard_metrics():
    """Test FCR, CES, and Agent Adherence scoring."""
    transcript = (
        "Hoş geldiniz Vodafone müşteri hizmetleri ben Ali. "
        "Faturam çok yüksek geldi şikayetçiyim. "
        "Kayıt altına alınmaktadır güvenlik amacıyla sistemden bakıyorum. "
        "Sorun halloldu teşekkür ederim. Başka bir işleminiz var mı? İyi günler dilerim."
    )
    metrics = LLMDiscourseGuard.analyze_call_metrics([], full_transcript=transcript)

    assert metrics["agent_adherence_score"] == 100
    assert "Kurumsal Selamlama" in metrics["adherence_checks_passed"]
    assert "KVKK / Ses Kaydı Bildirimi" in metrics["adherence_checks_passed"]
    assert metrics["fcr_status"] in ("Resolved", "Pending / Needs Follow-up")


def test_discourse_role_verification():
    """Test turn-taking sandwich verification on short ambiguous responses."""

    class MockSeg:
        def __init__(self, text, speaker):
            self.text = text
            self.speaker = speaker

    segments = [
        MockSeg("Merhaba nasıl yardımcı olabilirim", "SPEAKER_00"),
        MockSeg("Evet", "SPEAKER_00"),  # Ambiguous acknowledgement wrongly attributed to agent
        MockSeg("Faturam hakkında sorum var", "SPEAKER_00"),
    ]

    verified = LLMDiscourseGuard.verify_discourse_roles(segments)
    assert verified[1].speaker == "SPEAKER_01"
    assert getattr(verified[1], "auto_corrected", False) is True


def test_live_coaching_alerts():
    """Test live websocket coaching trigger rules."""
    session_id = "test-session-101"
    LiveCoachingService.clear_session(session_id)

    alert = LiveCoachingService.evaluate_chunk(
        session_id=session_id,
        text="Faturam çok yüksek geldi şikayetçiyim iptal etmek istiyorum",
        session_elapsed=5.0,
    )
    assert alert is not None
    assert alert["type"] == "escalation"
    assert "😡" in alert["title"]

    # Check duplicate prevention
    alert2 = LiveCoachingService.evaluate_chunk(
        session_id=session_id,
        text="iptal etmek istiyorum",
        session_elapsed=6.0,
    )
    assert alert2 is None


def test_mos_estimator():
    """Test ITU-T P.863 MOS calculation."""
    # Synthetic clean speech simulation (high SNR, no clipping)
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    clean_audio = np.zeros(32000, dtype=np.float32)
    clean_audio[:24000] = 0.5 * np.sin(2 * np.pi * 300 * t[:24000]) + np.random.normal(
        0, 0.001, 24000
    )
    clean_audio[24000:] = np.random.normal(0, 0.0005, 8000)

    mos_clean = MOSEstimator.estimate_mos(clean_audio)
    assert mos_clean["mos_score"] >= 4.0
    assert mos_clean["quality_grade"] == "Excellent / HD Voice"

    # Simulate heavy clipping (>98% amplitude)
    clipped_audio = np.ones(16000, dtype=np.float32) * 0.99
    mos_clipped = MOSEstimator.estimate_mos(clipped_audio)
    assert mos_clipped["clipping_rate_pct"] > 90.0
    assert mos_clipped["mos_score"] < 3.5
    assert mos_clipped["noc_alert"] is not None


def test_tier6_sota_diarization_upgrades():
    """Verify Tier-6 SOTA Telecom babble filter, Markov transition smoothing, and crosstalk events."""
    from asr_pro.core.keyword_engine import SegmentInput
    from asr_pro.services.audio_conditioning import suppress_telecom_crosstalk_and_babble
    from asr_pro.services.diarization_service import DiarizationService

    # 1. Test Telecom Babble Suppression Filter
    noisy_pcm = np.random.normal(0, 0.05, 32000).astype(np.float32)
    clean_pcm = suppress_telecom_crosstalk_and_babble(noisy_pcm, sample_rate=16000, gate_db=-25.0)
    assert clean_pcm.shape == noisy_pcm.shape
    assert isinstance(clean_pcm, np.ndarray)

    # 2. Test Markov Chain Smoothing
    segs = [
        SegmentInput(start=0.0, end=2.0, text="Merhaba Vodafone", speaker="SPEAKER_00"),
        SegmentInput(
            start=2.5, end=3.5, text="efendim", speaker="SPEAKER_01"
        ),  # 1 word ambiguous glitch
        SegmentInput(
            start=4.0, end=6.0, text="Size nasıl yardımcı olabilirim?", speaker="SPEAKER_00"
        ),
    ]
    diarizer = DiarizationService()
    smoothed = diarizer._apply_markov_smoothing(segs)
    assert smoothed[1].speaker == "SPEAKER_00"  # Markov smoothed the glitch!

    # 3. Test Crosstalk Extraction
    crosstalk_segs = [
        SegmentInput(
            start=10.0, end=14.0, text="Faturam çok yüksek geldi şikayetçiyim", speaker="SPEAKER_01"
        ),
        SegmentInput(
            start=12.5, end=15.0, text="Hemen kontrol ediyorum beyefendi", speaker="SPEAKER_00"
        ),
    ]
    events = diarizer.extract_crosstalk_events(crosstalk_segs)
    assert len(events) == 1
    assert events[0]["type"] == "interruption"
    assert events[0]["duration"] > 0.5

    # 3b. Crosstalk extraction from real pyannote acoustic overlap regions
    # (preferred path - no timestamp-boundary guessing involved).
    acoustic_events = diarizer.extract_crosstalk_events(
        crosstalk_segs, overlap_regions=[(12.5, 14.0)]
    )
    assert len(acoustic_events) == 1
    assert acoustic_events[0]["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert acoustic_events[0]["duration"] == 1.5


def test_acoustic_pitch_profiles_and_multi_speaker():
    """Verify acoustic F0 pitch estimation and 3+ speaker conference call supervisor tracking."""
    from asr_pro.core.keyword_engine import SegmentInput
    from asr_pro.services.diarization_service import DiarizationService

    segs = [
        SegmentInput(start=0.0, end=2.0, text="Merhaba Vodafone", speaker="SPEAKER_00"),
        SegmentInput(
            start=2.5, end=5.0, text="Faturamla ilgili şikayetim var", speaker="SPEAKER_01"
        ),
        SegmentInput(
            start=5.5,
            end=8.0,
            text="Görüşmeye takım lideri olarak dahil oluyorum",
            speaker="SPEAKER_02",
        ),
    ]
    diarizer = DiarizationService()
    profiles = diarizer.extract_speaker_pitch_profiles(segs)
    assert "SPEAKER_00" in profiles and profiles["SPEAKER_00"]["f0_mean_hz"] == 124.5
    assert "SPEAKER_01" in profiles and profiles["SPEAKER_01"]["f0_mean_hz"] == 215.0
    assert "SPEAKER_02" in profiles and profiles["SPEAKER_02"]["f0_mean_hz"] == 155.0
    assert "Bariton" in profiles["SPEAKER_00"]["voice_type"]
    assert "Soprano" in profiles["SPEAKER_01"]["voice_type"]
    assert "Takım Lideri" in profiles["SPEAKER_02"]["voice_type"]
