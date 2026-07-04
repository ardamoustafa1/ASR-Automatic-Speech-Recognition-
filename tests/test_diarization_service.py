from __future__ import annotations

"""Unit tests for DiarizationService and Agent/Customer role identification."""

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

    aligned, agent_id, customer_id = service.assign_speakers_to_segments(segments)
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

    aligned, agent_id, customer_id = service.assign_speakers_to_segments(segments)
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[1].speaker == "SPEAKER_01"
    assert agent_id == "SPEAKER_00"
    assert customer_id == "SPEAKER_01"
