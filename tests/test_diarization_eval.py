"""Tests for the DER (Diarization Error Rate) measurement harness.

Uses synthetic reference/hypothesis annotations (no audio/model needed) to
verify compute_der() reports a sane, known DER for a hand-constructed
mismatch - this is what proves the harness itself is trustworthy before it's
used to score real diarization output.
"""

from __future__ import annotations

from pyannote.core import Annotation, Segment

from asr_pro.services.diarization_eval import (
    DERResult,
    compute_der,
    turns_to_annotation,
)
from asr_pro.services.diarization_service import SpeakerTurn


def test_compute_der_perfect_match_is_zero():
    reference = Annotation(uri="call_1")
    reference[Segment(0.0, 5.0)] = "A"
    reference[Segment(5.0, 10.0)] = "B"

    hypothesis = Annotation(uri="call_1")
    hypothesis[Segment(0.0, 5.0)] = "SPEAKER_00"
    hypothesis[Segment(5.0, 10.0)] = "SPEAKER_01"

    result = compute_der(reference, hypothesis)
    assert isinstance(result, DERResult)
    assert result.der == 0.0
    assert result.reference_speakers == 2
    assert result.hypothesis_speakers == 2


def test_compute_der_speaker_confusion():
    reference = Annotation(uri="call_2")
    reference[Segment(0.0, 5.0)] = "A"
    reference[Segment(5.0, 10.0)] = "B"

    # DER's optimal-mapping step is label-agnostic (it permutes hypothesis
    # labels to best match reference), so merely renaming/swapping speaker IDs
    # produces zero confusion - only collapsing two real speakers into one
    # detected speaker (under-clustering, the failure mode min_speakers/
    # max_speakers guards against) registers as genuine confusion: the
    # optimal mapping can match one half correctly, but the other reference
    # speaker has no distinct hypothesis label left to map to.
    hypothesis = Annotation(uri="call_2")
    hypothesis[Segment(0.0, 10.0)] = "SPEAKER_00"

    result = compute_der(reference, hypothesis)
    assert result.der > 0.4
    assert result.hypothesis_speakers == 1
    assert result.reference_speakers == 2


def test_compute_der_missed_detection():
    reference = Annotation(uri="call_3")
    reference[Segment(0.0, 5.0)] = "A"
    reference[Segment(5.0, 10.0)] = "B"

    # Hypothesis only detected the first half - the second half is a pure miss.
    hypothesis = Annotation(uri="call_3")
    hypothesis[Segment(0.0, 5.0)] = "SPEAKER_00"

    result = compute_der(reference, hypothesis)
    assert 0.4 < result.der < 0.6


def test_turns_to_annotation_converts_speaker_turns():
    turns = [
        SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.0, end=4.0, speaker="SPEAKER_01"),
    ]
    annotation = turns_to_annotation(turns, uri="test")
    assert annotation.uri == "test"
    assert set(annotation.labels()) == {"SPEAKER_00", "SPEAKER_01"}
    assert annotation.label_duration("SPEAKER_00") == 2.0


def test_turns_to_annotation_skips_zero_duration_turns():
    turns = [
        SpeakerTurn(start=0.0, end=0.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=1.0, end=2.0, speaker="SPEAKER_01"),
    ]
    annotation = turns_to_annotation(turns)
    assert set(annotation.labels()) == {"SPEAKER_01"}
