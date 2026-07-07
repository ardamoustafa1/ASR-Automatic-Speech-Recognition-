"""Regression coverage for the diarization degenerate-speaker safeguard.

When speaker diarization fails to distinguish an agent from a customer
(e.g. a very short call or overlapping stereo channels collapse to a single
detected speaker), `analyze_soft_skills` falls back to scoring the whole
transcript as the agent and `analyze_churn_risk` falls back to analyzing the
whole transcript unfiltered. Silently trusting either result risks
attributing one party's speech to the other. `save_conversation_with_analysis`
must detect this and downgrade confidence / flag it instead of reporting a
confident-looking score.
"""

from unittest.mock import patch

import pytest

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.services.conversation_service import save_conversation_with_analysis


def _fake_diarizer_single_speaker(segments_data, audio_path=None):
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Merhaba yardımcı olur musunuz?",
            speaker="SPEAKER_00",
            segment_index=0,
        ),
        SegmentInput(
            start=2.0,
            end=4.0,
            text="Faturamda hata var, iptal edeceğim.",
            speaker="SPEAKER_00",
            segment_index=1,
        ),
    ]
    return segments, "SPEAKER_00", None, "pyannote", []  # degenerate: only one speaker identified


def _fake_diarizer_two_speakers(segments_data, audio_path=None):
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Merhaba nasıl yardımcı olabilirim?",
            speaker="SPEAKER_00",
            segment_index=0,
        ),
        SegmentInput(
            start=2.0,
            end=4.0,
            text="Faturamda hata var, iptal edeceğim.",
            speaker="SPEAKER_01",
            segment_index=1,
        ),
    ]
    return segments, "SPEAKER_00", "SPEAKER_01", "pyannote", []


@pytest.mark.usefixtures("setup_db")
def test_degenerate_speaker_separation_downgrades_confidence(db_session):
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = (
            _fake_diarizer_single_speaker
        )
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript="Merhaba yardımcı olur musunuz? Faturamda hata var, iptal edeceğim.",
        )

    assert result["diarization"]["speaker_separation_reliable"] is False
    assert result["churn"]["confidence"] == "Düşük"
    assert "güvenilir değil" in result["empathy"]["summary"]


@pytest.mark.usefixtures("setup_db")
def test_reliable_speaker_separation_is_not_flagged(db_session):
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = (
            _fake_diarizer_two_speakers
        )
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript="Merhaba nasıl yardımcı olabilirim? Faturamda hata var, iptal edeceğim.",
        )

    assert result["diarization"]["speaker_separation_reliable"] is True
    assert result["diarization"]["method"] == "pyannote"
    assert "güvenilir değil" not in result["empathy"]["summary"]


def _fake_diarizer_two_speakers_degraded_method(segments_data, audio_path=None):
    """Two distinct speaker labels, but from a non-acoustic (text heuristic) method."""
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Merhaba nasıl yardımcı olabilirim?",
            speaker="SPEAKER_00",
            segment_index=0,
        ),
        SegmentInput(
            start=2.0,
            end=4.0,
            text="Faturamda hata var, iptal edeceğim.",
            speaker="SPEAKER_01",
            segment_index=1,
        ),
    ]
    return segments, "SPEAKER_00", "SPEAKER_01", "text_heuristic", []


@pytest.mark.usefixtures("setup_db")
def test_degraded_method_with_two_speakers_still_flagged_unreliable(db_session):
    """Two distinct speaker labels alone must not imply reliability - the acoustic
    method matters. A text-heuristic split (no real acoustic signal) must be
    downgraded even when it happens to produce two different speaker IDs,
    otherwise a Vodafone-scale deployment would silently trust guessed splits."""
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = (
            _fake_diarizer_two_speakers_degraded_method
        )
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript="Merhaba nasıl yardımcı olabilirim? Faturamda hata var, iptal edeceğim.",
        )

    assert result["diarization"]["speaker_separation_reliable"] is False
    assert result["diarization"]["method"] == "text_heuristic"
    assert result["churn"]["confidence"] == "Düşük"
