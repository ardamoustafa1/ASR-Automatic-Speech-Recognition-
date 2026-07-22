"""Tests for the semantic + acoustic dual-guard role correction engine."""

from __future__ import annotations

from unittest.mock import patch

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.semantic_role_guard import enforce_semantic_role_guard


def test_returns_empty_list_unchanged():
    assert enforce_semantic_role_guard([], "agent-1", "cust-1") == []


def test_flips_customer_complaint_wrongly_assigned_to_agent():
    segments = [
        SegmentInput(
            start=0.0, end=2.0, text="şikayetçiyim, faturam çok yüksek", speaker="agent-1"
        ),
    ]
    result = enforce_semantic_role_guard(segments, "agent-1", "cust-1")
    assert result[0].speaker == "cust-1"
    assert result[0].auto_corrected is True


def test_flips_agent_script_wrongly_assigned_to_customer():
    segments = [
        SegmentInput(
            start=0.0,
            end=2.0,
            text="Vodafone müşteri hizmetleri, nasıl yardımcı olabilirim",
            speaker="cust-1",
        ),
    ]
    result = enforce_semantic_role_guard(segments, "agent-1", "cust-1")
    assert result[0].speaker == "agent-1"
    assert result[0].auto_corrected is True


def test_no_correction_when_no_strong_phrase_present():
    segments = [SegmentInput(start=0.0, end=2.0, text="evet tabii", speaker="agent-1")]
    result = enforce_semantic_role_guard(segments, "agent-1", "cust-1")
    assert result[0].speaker == "agent-1"
    assert result[0].auto_corrected is False


def test_no_correction_when_agent_or_customer_id_missing():
    segments = [SegmentInput(start=0.0, end=2.0, text="şikayetçiyim", speaker="agent-1")]
    result = enforce_semantic_role_guard(segments, None, None)
    assert result[0].speaker == "agent-1"
    assert result[0].auto_corrected is False


def test_detects_interruption_between_overlapping_segments():
    segments = [
        SegmentInput(start=0.0, end=5.0, text="konuşuyorum", speaker="agent-1"),
        # Starts well before the previous segment ends -> interruption.
        SegmentInput(start=4.5, end=6.0, text="dur bekle", speaker="cust-1"),
    ]
    result = enforce_semantic_role_guard(segments, "agent-1", "cust-1")
    assert result[0].is_interruption is False
    assert result[1].is_interruption is True


def test_no_interruption_when_gap_is_clean():
    segments = [
        SegmentInput(start=0.0, end=5.0, text="konuşuyorum", speaker="agent-1"),
        SegmentInput(start=5.5, end=6.0, text="tamam", speaker="cust-1"),
    ]
    result = enforce_semantic_role_guard(segments, "agent-1", "cust-1")
    assert result[1].is_interruption is False


def test_handles_dict_segments():
    segments = [{"start": 0.0, "end": 2.0, "text": "şikayetçiyim", "speaker": "agent-1"}]
    result = enforce_semantic_role_guard(segments, "agent-1", "cust-1")
    assert result[0]["speaker"] == "cust-1"
    assert result[0]["auto_corrected"] is True


def test_handles_plain_object_segments():
    class PlainSegment:
        def __init__(self, start, end, text, speaker):
            self.start = start
            self.end = end
            self.text = text
            self.speaker = speaker

    segments = [PlainSegment(0.0, 2.0, "şikayetçiyim", "agent-1")]
    result = enforce_semantic_role_guard(segments, "agent-1", "cust-1")
    assert result[0].speaker == "cust-1"


def test_dataclass_replace_failure_is_swallowed(caplog):
    segments = [SegmentInput(start=0.0, end=2.0, text="şikayetçiyim", speaker="agent-1")]
    with patch("dataclasses.replace", side_effect=RuntimeError("boom")):
        result = enforce_semantic_role_guard(segments, "agent-1", "cust-1")
    # Original segment is preserved (not corrected) when replace() blows up.
    assert result[0].speaker == "agent-1"
