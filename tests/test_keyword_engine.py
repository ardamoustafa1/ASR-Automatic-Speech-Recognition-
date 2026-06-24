"""Tests for keyword detection engine."""

import pytest

from asr_pro.core.keyword_engine import (
    SegmentInput,
    RuleInput,
    analyze_keywords,
    evaluate_rule_on_text,
)


@pytest.fixture
def zam_rule():
    return RuleInput(
        id="rule-zam",
        name="Zam Tespiti",
        keywords=("zam",),
        match_mode="semantic",
        severity="warning",
    )


@pytest.fixture
def fatura_rule():
    return RuleInput(
        id="rule-fatura",
        name="Fatura Tespiti",
        keywords=("fatura", "fatura itirazı"),
        match_mode="semantic",
        severity="info",
    )


def test_zam_not_matching_zaman(zam_rule):
    hits = evaluate_rule_on_text("Bu zamanda görüşmek üzere, iyi günler.", zam_rule)
    assert len(hits) == 0


def test_zam_detected_in_context(zam_rule):
    hits = evaluate_rule_on_text("Tarifeye zam yapıldığını duydum.", zam_rule)
    assert len(hits) >= 1
    assert any(h.matched_text for h in hits)


def test_semantic_synonym_fiyat_artisi(zam_rule):
    hits = evaluate_rule_on_text("Fiyat artışı çok yüksek olmuş.", zam_rule)
    assert len(hits) >= 1


def test_fatura_itirazi(fatura_rule):
    hits = evaluate_rule_on_text("Fatura itirazı için aramıştım.", fatura_rule)
    assert len(hits) >= 1


def test_multi_segment_analysis(zam_rule, fatura_rule):
    segments = [
        SegmentInput(start=0, end=5, text="Merhaba nasıl yardımcı olabilirim?", segment_index=0),
        SegmentInput(start=5, end=15, text="Fatura itirazı var, zam da yapılmış.", segment_index=1),
    ]
    hits = analyze_keywords(segments, [zam_rule, fatura_rule])
    assert len(hits) >= 2


def test_fuzzy_match():
    rule = RuleInput(
        id="fuzzy",
        name="Fuzzy",
        keywords=("iptal",),
        match_mode="fuzzy",
        fuzzy_threshold=0.85,
    )
    hits = evaluate_rule_on_text("iptali istiyorum", rule)
    assert len(hits) >= 1


def test_empty_segment():
    rule = RuleInput(id="x", name="X", keywords=("test",))
    hits = analyze_keywords([SegmentInput(start=0, end=0, text="   ")], [rule])
    assert hits == []
