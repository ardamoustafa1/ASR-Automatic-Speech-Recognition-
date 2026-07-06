import os

import pytest

from asr_pro.core.empathy_engine import analyze_soft_skills
from asr_pro.core.keyword_engine import SegmentInput

MODEL_AVAILABLE = os.environ.get("ASR_TEST_NO_MODEL") != "1"


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_empathy_high_score():
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Merhaba, sizi anlıyorum.",
            segment_index=0,
            speaker="SPEAKER_01",
        ),
        SegmentInput(
            start=2.5,
            end=5.0,
            text="Yaşadığınız durum için çok üzgünüm.",
            segment_index=1,
            speaker="SPEAKER_01",
        ),
        SegmentInput(
            start=5.5,
            end=8.0,
            text="Hemen kontrol ediyorum ve yardımcı olacağım.",
            segment_index=2,
            speaker="SPEAKER_01",
        ),
    ]

    result = analyze_soft_skills(segments, agent_speaker_id="SPEAKER_01", use_ai=False)

    assert result.score > 85
    assert len(result.active_listening_hits) >= 1
    assert len(result.compassion_hits) >= 1
    assert len(result.solution_hits) >= 1
    assert result.interruption_count == 0


def test_empathy_low_score_defensive():
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Yapacak bir şey yok hanımefendi.",
            segment_index=0,
            speaker="SPEAKER_01",
        ),
        SegmentInput(
            start=2.5,
            end=5.0,
            text="Size daha önce de söyledim, sistem böyle çalışıyor.",
            segment_index=1,
            speaker="SPEAKER_01",
        ),
    ]

    result = analyze_soft_skills(segments, agent_speaker_id="SPEAKER_01", use_ai=False)

    assert result.score <= 40
    assert len(result.defensive_hits) >= 2


def test_interruption_penalty():
    segments = [
        # Müşteri 0-3 saniye arası konuşuyor
        SegmentInput(
            start=0.0,
            end=3.0,
            text="Ben bunu kabul etmiyorum iptal edeceğim",
            segment_index=0,
            speaker="SPEAKER_00",
        ),
        # Temsilci müşteri lafını bitirmeden 2.0. saniyede araya giriyor! (Overlap 1.0 saniye, > 0.2)
        SegmentInput(
            start=2.0,
            end=5.0,
            text="Sizi anlıyorum efendim ama dinleyin",
            segment_index=1,
            speaker="SPEAKER_01",
        ),
    ]

    result = analyze_soft_skills(segments, agent_speaker_id="SPEAKER_01", use_ai=False)

    assert result.interruption_count == 1
    assert "Agresif Söz Kesme" in result.analysis_summary


def test_crisis_management_bonus_simulated():
    # To test AI logic purely, we mock it or pass use_ai=False.
    # We will test the structural output of the engine.
    pass


def test_stereo_cross_channel_jitter_not_counted_as_interruption():
    """Regression test: found via real stereo call-center audio - independently
    transcribed left/right channels produce sub-second timestamp overlap from
    VAD boundary imprecision alone, not real talk-overs. A brief overlap must
    not be scored as an interruption."""
    segments = [
        SegmentInput(
            start=0.0,
            end=3.0,
            text="Faturamda bir sorun var da onu konuşmak istiyorum",
            segment_index=0,
            speaker="SPEAKER_00",
        ),
        # Only 0.3s of overlap - typical cross-channel transcription jitter,
        # well under the real-interruption threshold.
        SegmentInput(
            start=2.7,
            end=5.0,
            text="Tabii efendim hemen bakıyorum",
            segment_index=1,
            speaker="SPEAKER_01",
        ),
    ]

    result = analyze_soft_skills(segments, agent_speaker_id="SPEAKER_01", use_ai=False)

    assert result.interruption_count == 0


def test_implausible_mega_overlap_not_counted_as_interruption():
    """Regression test: found via real stereo call-center audio - some Whisper
    segments have grossly incorrect end timestamps that produce multi-second
    "overlaps" (observed up to 11s in real recordings). No human interruption
    lasts that long; this is a corrupted timestamp, not a real talk-over."""
    segments = [
        SegmentInput(
            start=0.0,
            end=12.0,  # corrupted/over-extended end timestamp
            text="Faturamda bir sorun var da onu konuşmak istiyorum",
            segment_index=0,
            speaker="SPEAKER_00",
        ),
        SegmentInput(
            start=3.0,
            end=6.0,
            text="Tabii efendim hemen bakıyorum",
            segment_index=1,
            speaker="SPEAKER_01",
        ),
    ]

    result = analyze_soft_skills(segments, agent_speaker_id="SPEAKER_01", use_ai=False)

    assert result.interruption_count == 0


def test_backchannel_ack_not_counted_as_interruption():
    """A one-word backchannel acknowledgment ("Evet") is not a real utterance
    being cut off, even if a large overlap is measured against it."""
    segments = [
        SegmentInput(
            start=0.0,
            end=0.4,
            text="Evet",
            segment_index=0,
            speaker="SPEAKER_00",
        ),
        SegmentInput(
            start=0.1,
            end=3.0,
            text="Sizi anlıyorum efendim hemen yardımcı oluyorum",
            segment_index=1,
            speaker="SPEAKER_01",
        ),
    ]

    result = analyze_soft_skills(segments, agent_speaker_id="SPEAKER_01", use_ai=False)

    assert result.interruption_count == 0
