"""Additional coverage for LLMDiscourseGuard: interruption/CES scoring and every
segment-shape branch of verify_discourse_roles (dataclass, namedtuple, dict, and
the plain-object exception path).
"""

from __future__ import annotations

from collections import namedtuple

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.services.llm_discourse_guard import LLMDiscourseGuard

NamedSeg = namedtuple("NamedSeg", ["text", "speaker"])


def test_analyze_call_metrics_counts_interruptions_and_bumps_ces():
    segments = [
        {"text": "a", "is_interruption": True},
        {"text": "b", "is_interruption": True},
        {"text": "c", "is_interruption": False},
        {"text": "d", "is_interruption": False},
    ]
    metrics = LLMDiscourseGuard.analyze_call_metrics(segments, full_transcript="konuşma metni")
    # 2/4 = 50% interruption ratio > 0.15 threshold -> CES bumped by 1.0 from baseline 1.0
    assert metrics["ces_score"] == 2.0
    assert "Söz kesme sayısı: 2" in metrics["ces_explanation"]


def test_analyze_call_metrics_no_ces_bump_below_interruption_threshold():
    segments = [{"text": "a", "is_interruption": False} for _ in range(10)]
    metrics = LLMDiscourseGuard.analyze_call_metrics(segments, full_transcript="konuşma")
    assert metrics["ces_score"] == 1.0


def test_verify_discourse_roles_corrects_dataclass_segment():
    segments = [
        SegmentInput(
            start=0.0, end=1.0, text="Merhaba nasıl yardımcı olabilirim", speaker="SPEAKER_00"
        ),
        SegmentInput(start=1.0, end=1.5, text="Evet", speaker="SPEAKER_00"),
        SegmentInput(start=1.5, end=2.5, text="Faturam hakkında sorum var", speaker="SPEAKER_00"),
    ]
    result = LLMDiscourseGuard.verify_discourse_roles(segments)
    assert result[1].speaker == "SPEAKER_01"
    assert result[1].auto_corrected is True


def test_verify_discourse_roles_corrects_namedtuple_segment():
    segments = [
        NamedSeg("Merhaba nasıl yardımcı olabilirim", "SPEAKER_00"),
        NamedSeg("Evet", "SPEAKER_00"),
        NamedSeg("Faturam hakkında sorum var", "SPEAKER_00"),
    ]
    result = LLMDiscourseGuard.verify_discourse_roles(segments)
    assert result[1].speaker == "SPEAKER_01"


def test_verify_discourse_roles_corrects_dict_segment():
    segments = [
        {"text": "Merhaba nasıl yardımcı olabilirim", "speaker": "SPEAKER_00"},
        {"text": "Evet", "speaker": "SPEAKER_00"},
        {"text": "Faturam hakkında sorum var", "speaker": "SPEAKER_00"},
    ]
    result = LLMDiscourseGuard.verify_discourse_roles(segments)
    assert result[1]["speaker"] == "SPEAKER_01"
    assert result[1]["auto_corrected"] is True


def test_verify_discourse_roles_swallows_exception_on_immutable_object():
    class ImmutableSeg:
        __slots__ = ("text",)

        def __init__(self, text):
            self.text = text

        @property
        def speaker(self):
            return "SPEAKER_00"

    segments = [
        ImmutableSeg("Merhaba nasıl yardımcı olabilirim"),
        ImmutableSeg("Evet"),
        ImmutableSeg("Faturam hakkında sorum var"),
    ]
    # speaker has no setter and the object has no __dict__ (via __slots__), so
    # `seg.speaker = alternate_spk` raises - must be swallowed, not propagate.
    result = LLMDiscourseGuard.verify_discourse_roles(segments)
    assert result[1].speaker == "SPEAKER_00"  # left uncorrected after the failed attempt


def test_verify_discourse_roles_noop_for_short_segment_lists():
    segments = [SegmentInput(start=0.0, end=1.0, text="Evet", speaker="SPEAKER_00")]
    result = LLMDiscourseGuard.verify_discourse_roles(segments)
    assert result == segments
