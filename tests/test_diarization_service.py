from __future__ import annotations

"""Unit tests for DiarizationService and Agent/Customer role identification."""

from asr_pro.config import settings
from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.services.diarization_service import DiarizationService


def test_diarization_service_singleton():
    s1 = DiarizationService.get_instance()
    s2 = DiarizationService.get_instance()
    assert s1 is s2


def test_role_identification_agent_greeting():
    service = DiarizationService.get_instance()

    segments = [
        SegmentInput(
            start=0.0,
            end=3.0,
            text="Merhaba, ASR-Pro Müşteri Hizmetlerine hoş geldiniz, ben Arda, nasıl yardımcı olabilirim?",
            speaker="SPEAKER_00",
        ),
        SegmentInput(
            start=3.5,
            end=6.0,
            text="Merhaba, faturamla ilgili bir sorun yaşadım kontrol edebilir misiniz?",
            speaker="SPEAKER_01",
        ),
        SegmentInput(
            start=6.5,
            end=10.0,
            text="Tabii ki hemen kontrol ediyorum, anlayışınız için teşekkür ederim.",
            speaker="SPEAKER_00",
        ),
    ]

    aligned, agent_id, customer_id, method, overlap_regions = service.assign_speakers_to_segments(
        segments
    )
    assert len(aligned) == 3
    assert agent_id == "SPEAKER_00"
    assert customer_id == "SPEAKER_01"


def test_heuristic_speaker_alternation():
    service = DiarizationService.get_instance()

    segments = [
        SegmentInput(
            start=0.0, end=2.0, text="Hoş geldiniz size nasıl yardımcı olabilirim?", speaker=None
        ),
        SegmentInput(
            start=4.0, end=6.0, text="İnternet paketimi yükseltmek istiyorum.", speaker=None
        ),  # 2.0s pause -> new speaker
    ]

    aligned, agent_id, customer_id, method, overlap_regions = service.assign_speakers_to_segments(
        segments
    )
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[1].speaker == "SPEAKER_01"
    assert agent_id == "SPEAKER_00"
    assert customer_id == "SPEAKER_01"


def test_stereo_audio_diarization_and_alignment(tmp_path):
    import wave

    import numpy as np

    file_path = str(tmp_path / "test_stereo.wav")
    sr = 16000
    duration = 2.0
    n_samples = int(duration * sr)

    # Left channel: active 0.0 to 1.0s (SPEAKER_00)
    left_ch = np.zeros(n_samples, dtype=np.int16)
    left_ch[:sr] = np.random.normal(0, 10000, sr).astype(np.int16)

    # Right channel: active 1.0 to 2.0s (SPEAKER_01)
    right_ch = np.zeros(n_samples, dtype=np.int16)
    right_ch[sr:] = np.random.normal(0, 10000, n_samples - sr).astype(np.int16)

    stereo_data = np.empty((n_samples * 2,), dtype=np.int16)
    stereo_data[0::2] = left_ch
    stereo_data[1::2] = right_ch

    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo_data.tobytes())

    service = DiarizationService.get_instance()
    assert service.is_stereo_audio(file_path) is True

    segments = [
        SegmentInput(start=0.0, end=0.9, text="Merhaba, hoş geldiniz.", speaker=None),
        SegmentInput(start=1.1, end=1.9, text="Teşekkürler, bir sorunum vardı.", speaker=None),
    ]

    aligned, agent_id, customer_id, method, overlap_regions = service.assign_speakers_to_segments(
        segments, audio_path=file_path
    )
    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[1].speaker == "SPEAKER_01"
    assert method in ("pyannote", "stereo_energy")

    turns = service.diarize(file_path)
    assert len(turns) >= 2
    assert any(t.speaker == "SPEAKER_00" for t in turns)
    assert any(t.speaker == "SPEAKER_01" for t in turns)


def test_deduplicate_assigned_segments():
    service = DiarizationService.get_instance()
    segs = [
        SegmentInput(start=0.0, end=2.0, text="Efendim?", speaker="SPEAKER_01"),
        SegmentInput(start=2.0, end=4.0, text="Efendim?", speaker="SPEAKER_01"),
        SegmentInput(start=4.0, end=6.0, text="Efendim?", speaker="SPEAKER_01"),
        SegmentInput(start=6.0, end=8.0, text="İyi günler.", speaker="SPEAKER_01"),
    ]
    cleaned = service._deduplicate_assigned_segments(segs)
    assert len(cleaned) == 2
    assert cleaned[0].text == "Efendim?"
    assert cleaned[1].text == "İyi günler."


def test_stereo_physical_assignment_bypass(tmp_path):
    import wave
    from unittest.mock import patch

    import numpy as np

    file_path = str(tmp_path / "bypass_stereo.wav")
    sr = 16000
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(sr * 2 * 2, dtype=np.int16).tobytes())

    service = DiarizationService.get_instance()
    segments = [
        SegmentInput(start=0.0, end=1.0, text="Merhaba agent", speaker="SPEAKER_00"),
        SegmentInput(start=0.5, end=1.5, text="Merhaba customer", speaker="SPEAKER_01"),
    ]

    with patch.object(
        service,
        "diarize_with_overlap",
        side_effect=Exception("Should not be called when bypassing!"),
    ):
        aligned, agent_id, customer_id, method, overlap_regions = (
            service.assign_speakers_to_segments(segments, audio_path=file_path)
        )
        assert len(aligned) == 2
        assert aligned[0].speaker == "SPEAKER_00"
        assert aligned[1].speaker == "SPEAKER_01"
        assert method == "stereo_physical"


def test_finetuned_segmentation_swap_is_noop_when_unset():
    service = DiarizationService.get_instance()
    original = settings.diarization_finetuned_segmentation_path
    try:
        settings.diarization_finetuned_segmentation_path = ""
        service._maybe_swap_finetuned_segmentation()  # must not raise
    finally:
        settings.diarization_finetuned_segmentation_path = original


def test_finetuned_segmentation_swap_warns_on_missing_checkpoint(tmp_path, caplog):
    service = DiarizationService.get_instance()
    original = settings.diarization_finetuned_segmentation_path
    try:
        settings.diarization_finetuned_segmentation_path = str(tmp_path / "does_not_exist.ckpt")
        service._maybe_swap_finetuned_segmentation()  # must not raise, just warn
    finally:
        settings.diarization_finetuned_segmentation_path = original
