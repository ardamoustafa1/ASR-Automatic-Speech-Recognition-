"""Tests for the Top-Tier Churn Risk Engine (Acoustic + Temporal)."""

import os

import pytest

MODEL_AVAILABLE = os.environ.get("ASR_TEST_NO_MODEL") != "1"

from asr_pro.core.churn_engine import analyze_churn_risk
from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.sentiment_engine import SentimentClassifier


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
    neutral_segs = [
        SegmentInput(start=i * 5, end=(i + 1) * 5, text="Sadece normal bir cümle.", segment_index=i)
        for i in range(10)
    ]

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
    seg = SegmentInput(
        start=0,
        end=5,
        text="Sizin çekim gücünüzden bıktım, yarın Turkcell bayisine gidip hattımı taşıyacağım.",
        segment_index=0,
    )
    result = analyze_churn_risk([seg])

    assert "turkcell" in result.competitors_mentioned
    assert result.is_high_risk is True


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_risk_score_is_length_invariant():
    """Regression test: a call should not score higher purely because it has
    more chunks. Previously, cumulative_risk += final_segment_risk * 0.6 summed
    without normalization, so any call long enough to produce several
    qualifying chunks saturated to 100% regardless of severity."""
    mild_complaint = "Bu hizmetten pek memnun değilim, belki başka seçenekleri düşünebilirim."
    neutral = "Bugün hava durumu hakkında konuşalım."

    short_segments = [
        SegmentInput(start=i * 5, end=(i + 1) * 5, text=mild_complaint, segment_index=i)
        for i in range(5)  # 1 chunk
    ]
    long_segments = [
        SegmentInput(start=i * 5, end=(i + 1) * 5, text=mild_complaint, segment_index=i)
        for i in range(20)  # 4 chunks, same repeated content
    ]

    short_result = analyze_churn_risk(short_segments)
    long_result = analyze_churn_risk(long_segments)

    # A longer call repeating the exact same signal should not inflate the
    # score just because it produced more chunks.
    assert long_result.risk_score <= short_result.risk_score + 0.1
    assert long_result.risk_score < 1.0

    # A single low-key "normal conversation" chunk should never be high risk.
    neutral_segments = [
        SegmentInput(start=i * 5, end=(i + 1) * 5, text=neutral, segment_index=i) for i in range(5)
    ]
    neutral_result = analyze_churn_risk(neutral_segments)
    assert neutral_result.is_high_risk is False


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_speaker_isolation():
    # Scenario: Agent says "iptal edeceğim" -> Should not trigger churn
    segments = [
        SegmentInput(
            start=0,
            end=5,
            text="Talebiniz üzerine eski paketinizi iptal edeceğim.",
            segment_index=0,
            speaker="SPEAKER_01",
        )
    ]
    # We pass "SPEAKER_00" as the customer. The engine should ignore SPEAKER_01.
    result = analyze_churn_risk(segments, customer_speaker_id="SPEAKER_00")

    assert result.is_high_risk is False
    assert result.risk_score == 0.0
    assert len(result.insights) == 0


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_zero_false_positive_agent_and_neutral_speech():
    """Verify that agent explanations and neutral customer sentences never trigger false positive churn alarms."""
    # These exact sentences caused false positive alarms before Layer 1 & 2 guards:
    false_positive_segs = [
        SegmentInput(
            0,
            4,
            "İyiyim, teşekkürler. Ben de iyiyim, teşekkür ederim. Tüzeye tarifeleriniz için aramıştım.",
            0,
        ),
        SegmentInput(
            5,
            9,
            "Şu an mevcut tarife 50 GB sosyal paket sırası kullanıyoruz. Tarifelerimizin sağ olsun fiyatı 1050 TL.",
            1,
        ),
        SegmentInput(
            10, 14, "Ayrıca hangi şey var mı? Gigabyte'ı. 840. Bir paket daha vardı normalde.", 2
        ),
        SegmentInput(
            15,
            19,
            "Uygulamadan yatabiliyor muyum? Bu 840'ı 840'dan mı yapabiliyorsunuz? Yok. Sizler oradan tekrar.",
            3,
        ),
        SegmentInput(
            20,
            24,
            "Aranmışsınız, işaretlisiniz. Bugün de son günü olduğu için mağaza danışmanlarımıza dikmişler.",
            4,
        ),
        SegmentInput(
            25,
            29,
            "Ay sonu olduğu için tekrardan arattık aranıyorsunuz mağaza danışmanlarım olarak.",
            5,
        ),
    ]

    result = analyze_churn_risk(false_positive_segs)

    # 0 Hata / Sıfır Yanlış Alarm: None of these sentences should create a churn insight!
    assert len(result.insights) == 0
    assert result.is_high_risk is False
    assert result.risk_score < 0.20


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")
def test_enterprise_competitor_and_root_cause_diagnostics():
    """Verify that true threats get correctly flagged with severity badges and root cause diagnosis."""
    true_threat_segs = [
        SegmentInput(
            0,
            5,
            "950 TL mi gözüküyordu o? Rakip A1, S&S her şey için ötesi var. O 990 TL bu sana. Ben bir düşünüyorum.",
            0,
        )
    ]

    result = analyze_churn_risk(true_threat_segs)

    assert len(result.insights) == 1
    insight = result.insights[0]
    assert "a1" in result.competitors_mentioned or "rakip" in result.competitors_mentioned
    assert insight.severity == "🔥 Kritik Tehdit"
    assert "Rakip Firma" in insight.trigger_reason
