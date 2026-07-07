"""Diarization Error Rate (DER) measurement harness.

Without a quantitative accuracy metric, "we improved diarization" is an
unverifiable claim. This module scores DiarizationService's output against
human-annotated reference transcripts (RTTM format, the standard used by
NIST/DIHARD/VoxConverse diarization benchmarks) using pyannote.metrics'
DiarizationErrorRate - the same metric pyannote itself reports for its
published model benchmarks, so results are directly comparable.

DER = (false alarm + missed detection + speaker confusion) / total reference
speech time. Lower is better; 0.0 is perfect. Published pyannote-3.1 DER on
clean benchmark data is roughly 10-15%; noisy/overlapping telephony audio
typically scores worse - only measuring against your own real call recordings
tells you the number that matters for a specific deployment.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

from loguru import logger
from pyannote.core import Annotation, Segment

from asr_pro.services.diarization_service import DiarizationService, SpeakerTurn


def turns_to_annotation(turns: list[SpeakerTurn], uri: str = "hypothesis") -> Annotation:
    """Convert DiarizationService speaker turns into a pyannote Annotation."""
    annotation = Annotation(uri=uri)
    for turn in turns:
        if turn.end > turn.start:
            annotation[Segment(turn.start, turn.end)] = turn.speaker
    return annotation


def load_reference_rttm(rttm_path: str) -> Annotation:
    """Load a human-annotated reference diarization from an RTTM file.

    RTTM (Rich Transcription Time Marked) is the standard reference format
    for diarization benchmarks - one row per speaker turn:
    `SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>`.
    """
    from pyannote.database.util import load_rttm

    annotations_by_uri = load_rttm(rttm_path)
    if not annotations_by_uri:
        raise ValueError(f"No annotations found in RTTM file: {rttm_path}")
    # An RTTM file conventionally holds a single recording's reference.
    return next(iter(annotations_by_uri.values()))


@dataclass
class DERResult:
    uri: str
    der: float
    components: dict[str, float]
    reference_speakers: int
    hypothesis_speakers: int


@dataclass
class DERBatchResult:
    per_file: list[DERResult] = field(default_factory=list)
    overall_der: float = 0.0

    @property
    def mean_der(self) -> float:
        if not self.per_file:
            return 0.0
        return sum(r.der for r in self.per_file) / len(self.per_file)


def compute_der(reference: Annotation, hypothesis: Annotation) -> DERResult:
    """Compute Diarization Error Rate for a single reference/hypothesis pair."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    components = metric.compute_components(reference, hypothesis)
    der = metric.compute_metric(components)
    return DERResult(
        uri=reference.uri or "unknown",
        der=float(der),
        components={k: float(v) for k, v in components.items()},
        reference_speakers=len(reference.labels()),
        hypothesis_speakers=len(hypothesis.labels()),
    )


def evaluate_audio_against_rttm(audio_path: str, rttm_path: str) -> DERResult:
    """Run DiarizationService on `audio_path` and score it against `rttm_path`."""
    reference = load_reference_rttm(rttm_path)
    reference.uri = os.path.splitext(os.path.basename(audio_path))[0]

    turns = DiarizationService.get_instance().diarize(audio_path)
    hypothesis = turns_to_annotation(turns, uri=reference.uri)
    return compute_der(reference, hypothesis)


def evaluate_directory(audio_dir: str, rttm_dir: str) -> DERBatchResult:
    """Batch-evaluate every <name>.wav / <name>.rttm pair found in the two directories.

    Reports per-file DER plus an aggregate DER computed with pyannote's
    accumulation semantics (weighted by reference speech duration, not a
    naive mean of per-file rates) via DiarizationErrorRate's running total.
    """
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    result = DERBatchResult()

    rttm_files = sorted(glob.glob(os.path.join(rttm_dir, "*.rttm")))
    if not rttm_files:
        logger.warning(f"DER evaluation: no .rttm files found in {rttm_dir}")
        return result

    for rttm_path in rttm_files:
        name = os.path.splitext(os.path.basename(rttm_path))[0]
        audio_candidates = [
            p
            for ext in ("wav", "mp3", "flac", "m4a")
            for p in [os.path.join(audio_dir, f"{name}.{ext}")]
            if os.path.exists(p)
        ]
        if not audio_candidates:
            logger.warning(
                f"DER evaluation: no matching audio for reference {rttm_path}, skipping."
            )
            continue
        audio_path = audio_candidates[0]

        reference = load_reference_rttm(rttm_path)
        reference.uri = name
        turns = DiarizationService.get_instance().diarize(audio_path)
        hypothesis = turns_to_annotation(turns, uri=name)

        # __call__ (not compute_components alone) accumulates components into
        # the metric's running total, which abs(metric) below reports as the
        # duration-weighted overall DER across every file evaluated.
        components = metric(reference, hypothesis, uri=name, detailed=True)
        der = components[metric.metric_name_]
        result.per_file.append(
            DERResult(
                uri=name,
                der=float(der),
                components={k: float(v) for k, v in components.items() if k != metric.metric_name_},
                reference_speakers=len(reference.labels()),
                hypothesis_speakers=len(hypothesis.labels()),
            )
        )
        logger.info(f"DER evaluation: {name} -> DER={der:.3f}")

    result.overall_der = float(abs(metric))
    return result
