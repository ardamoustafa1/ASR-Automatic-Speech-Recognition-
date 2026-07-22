"""Tests for DomainAdaptationService.adapt_segments across segment shapes."""

from __future__ import annotations

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.services.domain_adaptation import DomainAdaptationService


def test_correct_terms_returns_falsy_text_unchanged():
    assert DomainAdaptationService.correct_terms("") == ""
    assert DomainAdaptationService.correct_terms(None) is None


def test_adapt_segments_corrects_dataclass_segments_in_place():
    segments = [
        SegmentInput(start=0.0, end=1.0, text="zaten modafon yanımdan"),
        SegmentInput(start=1.0, end=2.0, text="düz metin, düzeltme yok"),
    ]
    result = DomainAdaptationService.adapt_segments(segments)
    assert result[0].text == "zaten Vodafone yanımdan"
    assert result[1].text == "düz metin, düzeltme yok"


def test_adapt_segments_corrects_dict_segments_in_place():
    segments = [{"start": 0.0, "end": 1.0, "text": "romin ücreti"}]
    result = DomainAdaptationService.adapt_segments(segments)
    assert result[0]["text"] == "roaming ücreti"


def test_adapt_segments_corrects_plain_object_segments():
    class PlainSegment:
        def __init__(self, text):
            self.text = text

    seg = PlainSegment("havele yaptım")
    result = DomainAdaptationService.adapt_segments([seg])
    assert "havale" in result[0].text


def test_adapt_segments_skips_empty_text_segments():
    segments = [{"start": 0.0, "end": 1.0, "text": ""}]
    result = DomainAdaptationService.adapt_segments(segments)
    assert result[0]["text"] == ""


def test_adapt_segments_returns_empty_list_unchanged():
    assert DomainAdaptationService.adapt_segments([]) == []
