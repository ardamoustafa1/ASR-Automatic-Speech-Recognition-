"""Regression coverage for two foreign-key integrity bugs found while running
the app against a FK-enforcing database (PostgreSQL in production; the SQLite
test engine now enables `PRAGMA foreign_keys=ON` in conftest.py to match it).

1. save_conversation_with_analysis inserted keyword_hits and
   transcript_segments in the same flush with no relationship() linking the
   two mapped classes, so SQLAlchemy did not guarantee segment rows landed
   before the hit rows referencing them by segment_id - every conversation
   with at least one keyword hit failed to save against PostgreSQL.
2. DELETE /conversations/{id} (the GDPR "right to be forgotten" endpoint)
   deleted the conversation row directly with no ON DELETE CASCADE on the
   child tables, so it always failed once any transcript_segments/
   keyword_hits rows existed for that conversation.

Both were invisible under the old test setup because SQLite does not enforce
foreign keys by default, unlike PostgreSQL.
"""

from unittest.mock import patch

import pytest

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.db.models import Conversation, KeywordHit, TranscriptSegmentRow
from asr_pro.services.conversation_service import save_conversation_with_analysis


def _fake_diarizer_single_segment(segments_data, audio_path=None):
    segments = [
        SegmentInput(
            start=0.0,
            end=3.0,
            text="Fatura itirazı için aramıştım, iade istiyorum.",
            speaker="SPEAKER_00",
            segment_index=0,
        )
    ]
    return segments, "SPEAKER_00", None, "text_heuristic", []


@pytest.mark.usefixtures("setup_db")
def test_save_conversation_with_keyword_hits_does_not_violate_fk(db_session):
    """A conversation whose text matches keyword rules must persist cleanly -
    this reproduces the exact 500 seen live via /conversations/analyze."""
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = (
            _fake_diarizer_single_segment
        )
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript="Fatura itirazı için aramıştım, iade istiyorum.",
            sector="omni",
        )

    assert result["hit_count"] > 0
    hits = (
        db_session.query(KeywordHit)
        .filter(KeywordHit.conversation_id == result["conversation_id"])
        .all()
    )
    assert len(hits) == result["hit_count"]
    # Every persisted hit's segment_id must resolve to a real, already-inserted row.
    segment_ids = {
        s.id
        for s in db_session.query(TranscriptSegmentRow)
        .filter(TranscriptSegmentRow.conversation_id == result["conversation_id"])
        .all()
    }
    for hit in hits:
        assert hit.segment_id in segment_ids


@pytest.mark.usefixtures("setup_db")
def test_delete_conversation_with_children_succeeds(db_session):
    """Deleting a conversation that has transcript_segments/keyword_hits rows
    must not raise a FK violation (GDPR right-to-be-forgotten path)."""
    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance"
    ) as mock_get_instance:
        mock_get_instance.return_value.assign_speakers_to_segments.side_effect = (
            _fake_diarizer_single_segment
        )
        mock_get_instance.return_value.extract_crosstalk_events.return_value = []
        mock_get_instance.return_value.extract_speaker_pitch_profiles.return_value = {}
        result = save_conversation_with_analysis(
            db=db_session,
            segments_data=[],
            full_transcript="Fatura itirazı için aramıştım, iade istiyorum.",
            sector="omni",
        )

    conv_id = result["conversation_id"]
    assert result["hit_count"] > 0  # sanity: there ARE children to cascade-delete

    db_session.query(KeywordHit).filter(KeywordHit.conversation_id == conv_id).delete(
        synchronize_session=False
    )
    db_session.query(TranscriptSegmentRow).filter(
        TranscriptSegmentRow.conversation_id == conv_id
    ).delete(synchronize_session=False)
    conv = db_session.query(Conversation).filter(Conversation.id == conv_id).first()
    db_session.delete(conv)
    db_session.commit()  # must not raise IntegrityError

    assert db_session.query(Conversation).filter(Conversation.id == conv_id).first() is None
    assert (
        db_session.query(TranscriptSegmentRow)
        .filter(TranscriptSegmentRow.conversation_id == conv_id)
        .count()
        == 0
    )
    assert db_session.query(KeywordHit).filter(KeywordHit.conversation_id == conv_id).count() == 0
