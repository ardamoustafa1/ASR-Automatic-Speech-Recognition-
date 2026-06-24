"""Tests for the Top-Tier Churn Risk Engine (Acoustic + Temporal)."""

import os
import pytest

MODEL_AVAILABLE = os.environ.get("ASR_TEST_NO_MODEL") != "1"

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.sentiment_engine import SentimentClassifier
from asr_pro.core.churn_engine import analyze_churn_risk


@pytest.fixture(scope="session", autouse=True)
def preload_model():
    """Ensure the model is loaded once before tests."""
    if MODEL_AVAILABLE:
        SentimentClassifier.get_instance()._load_model()


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_acoustic_stress_wpm():
    # Scenario: Speaking 15 words in 3 seconds -> 300 WPM (Extreme Stress)
    fast_text = "Sizi kaç kere aramam gerekiyor neden kimse telefonlarımı açmıyor bu nasıl bir rezalet hemen üyeliğimi sonlandırın"
    seg = SegmentInput(start=0.0, end=3.0, text=fast_text, segment_index=0)
    
    result = analyze_churn_risk([seg])
    
    assert result.is_high_risk is True
    assert result.average_wpm > 250
    assert result.insights[0].acoustic_stress_multiplier == 1.4  # High stress multiplier applied


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_temporal_weighting():
    # Comparing the same threat at the beginning vs the end of the call
    threat = "Eğer böyle devam ederse başka bir firmaya geçiş yapacağım."
    
    # 10 segments total to force chunking (chunk size is 5)
    neutral_segs = [SegmentInput(start=i*5, end=(i+1)*5, text="Sadece normal bir cümle.", segment_index=i) for i in range(10)]
    
    segments_early = list(neutral_segs)
    segments_early[0] = SegmentInput(start=0, end=5, text=threat, segment_index=0)
    
    segments_late = list(neutral_segs)
    segments_late[-1] = SegmentInput(start=45, end=50, text=threat, segment_index=9)
    
    res_early = analyze_churn_risk(segments_early)
    res_late = analyze_churn_risk(segments_late)
    
    # The late threat should carry more risk score due to temporal weighting
    assert res_late.risk_score > res_early.risk_score


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_competitor_ner():
    # Scenario: Mentioning Vodafone
    seg = SegmentInput(start=0, end=5, text="Sizin çekim gücünüzden bıktım, yarın Turkcell bayisine gidip hattımı taşıyacağım.", segment_index=0)
    result = analyze_churn_risk([seg])
    
    assert "turkcell" in result.competitors_mentioned
    assert result.is_high_risk is True


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_speaker_isolation():
    # Scenario: Agent says "iptal edeceğim" -> Should not trigger churn
    segments = [
        SegmentInput(start=0, end=5, text="Talebiniz üzerine eski paketinizi iptal edeceğim.", segment_index=0, speaker="SPEAKER_01")
    ]
    # We pass "SPEAKER_00" as the customer. The engine should ignore SPEAKER_01.
    result = analyze_churn_risk(segments, customer_speaker_id="SPEAKER_00")
    
    assert result.is_high_risk is False
    assert result.risk_score == 0.0
    assert len(result.insights) == 0
