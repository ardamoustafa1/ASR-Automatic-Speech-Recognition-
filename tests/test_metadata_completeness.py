"""Regression coverage for churn/empathy engine data that was computed but
silently dropped before reaching metadata_json - a reviewer needs WHICH
competitor was named and WHICH specific phrases drove the empathy score, not
just the final numeric scores. See conversation_service.py's
final_metadata["competitors_mentioned"] / ["customer_average_wpm"] /
["empathy_breakdown"].
"""

from unittest.mock import patch

import pytest

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.services.conversation_service import save_conversation_with_analysis


def _fake_diarizer(segments_data, audio_path=None):
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Sizi anlıyorum, çok üzgünüm, hemen çözüm bulalım.",
            speaker="SPEAKER_00",
            segment_index=0,
        ),
        SegmentInput(
            start=2.0,
            end=5.0,
            text="Rakip firma Turkcell bana daha iyi teklif verdi, geçmeyi düşünüyorum.",
            speaker="SPEAKER_01",
            segment_index=1,
        ),
    ]
    return segments, "SPEAKER_00", "SPEAKER_01", "pyannote", []


@pytest.mark.usefixtures("setup_db")
def test_competitor_mentions_and_wpm_are_persisted(db_session):
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = _fake_diarizer
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript=(
                "Sizi anlıyorum, çok üzgünüm, hemen çözüm bulalım. "
                "Rakip firma Turkcell bana daha iyi teklif verdi, geçmeyi düşünüyorum."
            ),
        )

    from asr_pro.db.models import Conversation

    conv = db_session.query(Conversation).filter(Conversation.id == result["conversation_id"]).one()
    meta = conv.metadata_json

    assert "competitors_mentioned" in meta
    assert any("turkcell" in c.lower() for c in meta["competitors_mentioned"])
    assert "customer_average_wpm" in meta
    assert isinstance(meta["customer_average_wpm"], int)


@pytest.mark.usefixtures("setup_db")
def test_empathy_breakdown_is_persisted_with_matched_phrases(db_session):
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = _fake_diarizer
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript="Sizi anlıyorum, çok üzgünüm, hemen çözüm bulalım.",
        )

    from asr_pro.db.models import Conversation

    conv = db_session.query(Conversation).filter(Conversation.id == result["conversation_id"]).one()
    breakdown = conv.metadata_json["empathy_breakdown"]

    assert breakdown["active_listening_hits"], "agent said 'sizi anlıyorum' - must be captured"
    assert breakdown["compassion_hits"], "agent said 'çok üzgünüm' - must be captured"
    assert "interruption_count" in breakdown
    assert "agent_wpm_avg" in breakdown


def _fake_diarizer_toxic(segments_data, audio_path=None):
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Siktir git buradan, seni orospu çocuğu!",
            speaker="SPEAKER_01",
            segment_index=0,
        ),
    ]
    return segments, "SPEAKER_00", "SPEAKER_01", "pyannote", []


@pytest.mark.usefixtures("setup_db")
def test_toxicity_result_is_persisted(db_session):
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = (
            _fake_diarizer_toxic
        )
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript="Siktir git buradan, seni orospu çocuğu!",
        )

    from asr_pro.db.models import Conversation

    conv = db_session.query(Conversation).filter(Conversation.id == result["conversation_id"]).one()
    toxicity = conv.metadata_json["toxicity"]
    assert toxicity["is_clean"] is False
    assert "siktir" in toxicity["matched_terms"]
    assert toxicity["toxicity_rate"] > 0


@pytest.mark.usefixtures("setup_db")
def test_crm_summary_persisted_when_enabled(db_session):
    """CRM summary is skipped under the default test-mode flag (real
    zero-shot model calls are unstable in a heavy multi-native-library test
    process) - this test forces it on with a fake classifier to verify the
    persistence wiring itself, independent of the real model."""

    class _FakeClassifier:
        def predict(self, text, labels=None, hypothesis=None):
            return {
                "labels": list(labels or ["Bilinmiyor"]),
                "scores": [1.0] + [0.0] * (len(labels or []) - 1),
            }

    with (
        patch(
            "asr_pro.services.diarization_service.DiarizationService.get_instance"
        ) as mock_get_instance,
        patch("asr_pro.config._is_testing", False),
        patch(
            "asr_pro.core.sentiment_engine.SentimentClassifier.get_instance",
            return_value=_FakeClassifier(),
        ),
    ):
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = _fake_diarizer
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript=(
                "Sizi anlıyorum, çok üzgünüm, hemen çözüm bulalım. "
                "Rakip firma Turkcell bana daha iyi teklif verdi, geçmeyi düşünüyorum."
            ),
        )

    from asr_pro.db.models import Conversation

    conv = db_session.query(Conversation).filter(Conversation.id == result["conversation_id"]).one()
    summary = conv.metadata_json.get("call_summary")
    assert summary is not None
    # The fake classifier just echoes labels[0] of whatever label set it was
    # given, so the exact value isn't meaningful here - what matters is that
    # generate_crm_summary ran for real and its four fields all persisted.
    assert summary["intent"]
    assert summary["issue"]
    assert summary["action"]
    assert summary["resolution"]
    assert summary["executive_summary"]
