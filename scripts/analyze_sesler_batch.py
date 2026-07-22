#!/usr/bin/env python3
"""
Real-audio batch validation pipeline for churn/empathy/diarization calibration.

Runs the full analysis pipeline (transcription -> speaker diarization ->
churn risk -> empathy) against real stereo call-center recordings, WITHOUT
writing anything to the application database. This is a read-only validation
pass: it surfaces where the models actually break on real speech (crosstalk,
accents, ASR mistranscription, unseen competitor phrasing) that no synthetic
unit test can catch, and produces score distributions for calibration review.

Deliberately does not call save_conversation_with_analysis() - the audio in
sesler/ contains real customer calls and must not be seeded into the demo
DB/UI. Output is a JSONL report file only.

Usage:
  python3 scripts/analyze_sesler_batch.py [--limit N] [--audio-dir sesler]
                                           [--output data/sesler_analysis_results.jsonl]
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _already_processed(output_path: Path) -> set[str]:
    """Filenames already present in the report, so a re-run skips them."""
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["filename"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def analyze_one(audio_path: Path) -> dict:
    from asr_pro.core.churn_engine import analyze_churn_risk
    from asr_pro.core.compliance_engine import analyze_compliance_risk
    from asr_pro.core.empathy_engine import analyze_soft_skills
    from asr_pro.core.keyword_engine import SegmentInput
    from asr_pro.services.asr_service import ASRService
    from asr_pro.services.diarization_service import DiarizationService

    start = time.monotonic()

    asr = ASRService.get_instance()
    raw_segments, duration = asr.transcribe(str(audio_path), language="tr")

    segment_inputs = [
        SegmentInput(
            start=s.start,
            end=s.end,
            text=s.text,
            speaker=getattr(s, "speaker", None),
            segment_index=i,
        )
        for i, s in enumerate(raw_segments)
    ]

    diarizer = DiarizationService.get_instance()
    aligned_segments, agent_id, customer_id = diarizer.assign_speakers_to_segments(
        segment_inputs, audio_path=str(audio_path)
    )

    speaker_separation_reliable = bool(agent_id) and bool(customer_id) and agent_id != customer_id

    empathy_result = analyze_soft_skills(aligned_segments, agent_speaker_id=agent_id)
    churn_result = analyze_churn_risk(aligned_segments, customer_speaker_id=customer_id)
    compliance_violations = analyze_compliance_risk(
        aligned_segments, domain_key="telecom", agent_speaker_id=agent_id
    )

    full_transcript = " ".join(s.text for s in aligned_segments if s.text)
    elapsed = time.monotonic() - start

    return {
        "filename": audio_path.name,
        "audio_duration_sec": round(duration, 1),
        "processing_sec": round(elapsed, 1),
        "segment_count": len(aligned_segments),
        "agent_speaker_id": agent_id,
        "customer_speaker_id": customer_id,
        "speaker_separation_reliable": speaker_separation_reliable,
        "churn": {
            "risk_score": churn_result.risk_score,
            "is_high_risk": churn_result.is_high_risk,
            "confidence": churn_result.confidence,
            "risk_breakdown": churn_result.risk_breakdown,
            "competitors_mentioned": list(churn_result.competitors_mentioned),
            "average_wpm": churn_result.average_wpm,
        },
        "empathy": {
            "score": empathy_result.score,
            "summary": empathy_result.analysis_summary,
        },
        "compliance_violations": [
            {
                "severity": v.severity,
                "category": v.category,
                "segment_text": v.segment_text,
            }
            for v in compliance_violations
        ],
        "transcript_preview": full_transcript[:400],
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default="sesler")
    parser.add_argument("--output", default="data/sesler_analysis_results.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files")
    args = parser.parse_args()

    audio_dir = WORKSPACE_ROOT / args.audio_dir
    output_path = WORKSPACE_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(audio_dir.glob("*.wav"))
    if args.limit:
        files = files[: args.limit]

    done = _already_processed(output_path)
    pending = [f for f in files if f.name not in done]

    print(
        f"[analyze_sesler_batch] {len(files)} total files, {len(done)} already done, "
        f"{len(pending)} to process."
    )

    with open(output_path, "a", encoding="utf-8") as out:
        for idx, audio_path in enumerate(pending, 1):
            print(f"[{idx}/{len(pending)}] {audio_path.name} ...", flush=True)
            try:
                result = analyze_one(audio_path)
            except Exception as exc:
                result = {
                    "filename": audio_path.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"  FAILED: {result['error']}", flush=True)
            else:
                print(
                    f"  ok - churn={result['churn']['risk_score']} "
                    f"empathy={result['empathy']['score']} "
                    f"reliable_split={result['speaker_separation_reliable']}",
                    flush=True,
                )
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

    print(f"[analyze_sesler_batch] Done. Report at {output_path}")


if __name__ == "__main__":
    main()
