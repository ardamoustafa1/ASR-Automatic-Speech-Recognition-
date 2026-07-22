"""Tests for the DER (Diarization Error Rate) measurement harness.

Uses synthetic reference/hypothesis annotations (no audio/model needed) to
verify compute_der() reports a sane, known DER for a hand-constructed
mismatch - this is what proves the harness itself is trustworthy before it's
used to score real diarization output.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pyannote.core import Annotation, Segment

from asr_pro.services.diarization_eval import (
    DERBatchResult,
    DERResult,
    compute_der,
    evaluate_audio_against_rttm,
    evaluate_directory,
    load_reference_rttm,
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


def test_der_batch_result_mean_der_empty_is_zero():
    assert DERBatchResult().mean_der == 0.0


def test_der_batch_result_mean_der_averages_per_file():
    batch = DERBatchResult(
        per_file=[
            DERResult(uri="a", der=0.2, components={}, reference_speakers=2, hypothesis_speakers=2),
            DERResult(uri="b", der=0.4, components={}, reference_speakers=2, hypothesis_speakers=2),
        ]
    )
    assert batch.mean_der == pytest.approx(0.3)


def _write_rttm(path, uri, turns):
    with open(path, "w") as f:
        for start, duration, speaker in turns:
            f.write(f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")


def test_load_reference_rttm_reads_speaker_turns(tmp_path):
    rttm_path = tmp_path / "call_1.rttm"
    _write_rttm(rttm_path, "call_1", [(0.0, 5.0, "A"), (5.0, 5.0, "B")])

    annotation = load_reference_rttm(str(rttm_path))

    assert set(annotation.labels()) == {"A", "B"}


def test_load_reference_rttm_raises_for_empty_file(tmp_path):
    rttm_path = tmp_path / "empty.rttm"
    rttm_path.write_text("")

    with pytest.raises(ValueError, match="No annotations found"):
        load_reference_rttm(str(rttm_path))


def test_evaluate_audio_against_rttm_scores_diarization_output(tmp_path):
    rttm_path = tmp_path / "call_1.rttm"
    _write_rttm(rttm_path, "call_1", [(0.0, 5.0, "A"), (5.0, 5.0, "B")])
    audio_path = str(tmp_path / "call_1.wav")

    mock_turns = [
        SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=5.0, end=10.0, speaker="SPEAKER_01"),
    ]
    mock_service = MagicMock()
    mock_service.diarize.return_value = mock_turns

    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance",
        return_value=mock_service,
    ):
        result = evaluate_audio_against_rttm(audio_path, str(rttm_path))

    assert isinstance(result, DERResult)
    assert result.der == 0.0
    mock_service.diarize.assert_called_once_with(audio_path)


def test_evaluate_directory_returns_empty_result_when_no_rttm_files(tmp_path):
    result = evaluate_directory(str(tmp_path), str(tmp_path))
    assert result.per_file == []
    assert result.overall_der == 0.0


def test_evaluate_directory_skips_rttm_without_matching_audio(tmp_path):
    rttm_path = tmp_path / "call_1.rttm"
    _write_rttm(rttm_path, "call_1", [(0.0, 5.0, "A")])
    # No matching call_1.wav/.mp3/.flac/.m4a in tmp_path.
    result = evaluate_directory(str(tmp_path), str(tmp_path))
    assert result.per_file == []


def test_evaluate_directory_scores_matched_audio_rttm_pairs(tmp_path):
    rttm_path = tmp_path / "call_1.rttm"
    _write_rttm(rttm_path, "call_1", [(0.0, 5.0, "A"), (5.0, 5.0, "B")])
    audio_path = tmp_path / "call_1.wav"
    audio_path.write_bytes(b"")  # existence is all evaluate_directory checks

    mock_turns = [
        SpeakerTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=5.0, end=10.0, speaker="SPEAKER_01"),
    ]
    mock_service = MagicMock()
    mock_service.diarize.return_value = mock_turns

    with patch(
        "asr_pro.services.diarization_service.DiarizationService.get_instance",
        return_value=mock_service,
    ):
        result = evaluate_directory(str(tmp_path), str(tmp_path))

    assert len(result.per_file) == 1
    assert result.per_file[0].uri == "call_1"
    assert result.per_file[0].der == 0.0
    assert result.overall_der == 0.0
