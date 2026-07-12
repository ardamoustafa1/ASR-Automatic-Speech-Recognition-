"""End-to-end check: PII spoken during a call must never reach the database.

Covers the full save_conversation_with_analysis path - a card number read
aloud during identity verification has to be masked in both the persisted
segment rows and the conversation's full transcript (PCI-DSS 3.4 / KVKK).
"""

from unittest.mock import patch

import pytest

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.db.models import Conversation, TranscriptSegmentRow
from asr_pro.services.conversation_service import save_conversation_with_analysis

VALID_PAN = "4111111111111111"


def _fake_diarizer_with_pan(segments_data, audio_path=None):
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Kimlik doğrulama için kart numaranızı alabilir miyim?",
            speaker="SPEAKER_00",
            segment_index=0,
        ),
        SegmentInput(
            start=2.0,
            end=6.0,
            text=f"Tabii, kart numaram {VALID_PAN} efendim.",
            speaker="SPEAKER_01",
            segment_index=1,
        ),
    ]
    return segments, "SPEAKER_00", "SPEAKER_01", "pyannote", []


@pytest.mark.usefixtures("setup_db")
def test_card_number_never_persisted(db_session):
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = (
            _fake_diarizer_with_pan
        )
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript=f"Kimlik doğrulama için kart numaranızı alabilir miyim? Tabii, kart numaram {VALID_PAN} efendim.",
        )

    conv = db_session.query(Conversation).filter(Conversation.id == result["conversation_id"]).one()
    rows = (
        db_session.query(TranscriptSegmentRow)
        .filter(TranscriptSegmentRow.conversation_id == conv.id)
        .all()
    )

    assert VALID_PAN not in conv.full_transcript
    assert "**** 1111" in conv.full_transcript
    for row in rows:
        assert VALID_PAN not in row.text
    assert conv.metadata_json["pii_redaction"]["masked_counts"] == {"pan": 1}
