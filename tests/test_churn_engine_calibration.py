"""Golden-set calibration suite for the churn risk engine.

Asserts realistic Turkish call-center scenarios land in the expected risk
*band* (low / medium / high) rather than exact values (the underlying model
is probabilistic), and that the score is monotonic with respect to explicit
churn signal. This is the regression evidence handed to enterprise
buyers/compliance reviewers to demonstrate the score behaves sanely.
"""

import os
from unittest.mock import patch

import pytest

MODEL_AVAILABLE = os.environ.get("ASR_TEST_NO_MODEL") != "1"

from asr_pro.config import settings
from asr_pro.core.churn_engine import analyze_churn_risk
from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.sentiment_engine import SentimentClassifier

pytestmark = pytest.mark.skipif(not MODEL_AVAILABLE, reason="HuggingFace model required")


@pytest.fixture(scope="session", autouse=True)
def preload_model():
    if MODEL_AVAILABLE:
        SentimentClassifier.get_instance()._load_model()


def _segments(texts: list[str], wpm: float = 120.0) -> list[SegmentInput]:
    segs = []
    for i, t in enumerate(texts):
        duration = max(0.5, len(t.split()) / (wpm / 60))
        segs.append(SegmentInput(start=i * 6, end=i * 6 + duration, text=t, segment_index=i))
    return segs


LOW_RISK_SCENARIOS = {
    "happy_resolved_call": [
        "Merhaba, size nasıl yardımcı olabilirim.",
        "Faturamla ilgili küçük bir sorum vardı, hemen çözüldü.",
        "Çok teşekkür ederim, harika bir hizmetti.",
        "Rica ederim, iyi günler dilerim.",
    ],
    "neutral_small_talk": [
        "Bugün hava çok güzeldi, değil mi?",
        "Evet, hafta sonu için planlarım var.",
        "Faturamı öğrenmek için aramıştım sadece.",
        "Tabii, hemen kontrol ediyorum.",
    ],
    "resolved_complaint": [
        "Geçen ay faturamda bir hata vardı, biraz canım sıkılmıştı.",
        "Anlıyorum, hemen düzeltiyorum.",
        "Teşekkürler, bu sorunu çözdüğünüz için memnun oldum.",
        "Başka bir konuda yardımcı olabilir miyim, hayır teşekkürler.",
    ],
}

HIGH_RISK_SCENARIOS = {
    "explicit_cancellation_threat": [
        "Bu hizmetten gerçekten çok memnun değilim.",
        "Sürekli aynı sorunu yaşıyorum, artık çok sinirlendim.",
        "Eğer bu sorun çözülmezse üyeliğimi hemen iptal edeceğim.",
        "Başka bir firmaya geçmeyi ciddi ciddi düşünüyorum.",
    ],
    "competitor_switch_with_frustration": [
        "Sizin hizmetinizden gerçekten bıktım artık.",
        "Rakip firmalar çok daha iyi teklifler sunuyor.",
        "Yarın Turkcell bayisine gidip hattımı taşıyacağım.",
        "Bu son şansınızdı, karar verdim bile.",
    ],
}


@pytest.mark.parametrize("scenario", LOW_RISK_SCENARIOS.values(), ids=LOW_RISK_SCENARIOS.keys())
def test_low_risk_scenarios_stay_low(scenario):
    result = analyze_churn_risk(_segments(scenario))
    assert result.risk_score < 0.5, f"Expected low risk, got {result.risk_score}"
    assert result.is_high_risk is False


@pytest.mark.parametrize("scenario", HIGH_RISK_SCENARIOS.values(), ids=HIGH_RISK_SCENARIOS.keys())
def test_high_risk_scenarios_flagged(scenario):
    result = analyze_churn_risk(_segments(scenario))
    assert result.risk_score >= 0.5, f"Expected elevated risk, got {result.risk_score}"


def test_score_is_monotonic_with_explicit_signal():
    """Appending an explicit cancellation line to an otherwise-neutral call
    must not decrease the score - a basic sanity property any calibrated
    model should satisfy."""
    neutral = [
        "Bugün hava durumu hakkında konuşalım.",
        "Faturamı öğrenmek için aramıştım.",
        "Teşekkürler, anladım.",
    ]
    escalated = neutral + ["Eğer bu sorun çözülmezse hemen üyeliğimi iptal edeceğim."]

    neutral_result = analyze_churn_risk(_segments(neutral))
    escalated_result = analyze_churn_risk(_segments(escalated))

    assert escalated_result.risk_score >= neutral_result.risk_score


def test_confidence_reflects_evidence_volume():
    """A score backed by a single short chunk should carry lower confidence
    than one backed by several agreeing chunks."""
    short_call = _segments(["Bu hizmetten hiç memnun değilim, iptal edeceğim."])
    long_call = _segments(
        [
            "Bu hizmetten hiç memnun değilim, iptal edeceğim.",
            "Sürekli aynı sorunu yaşıyorum.",
            "Başka firmalara bakıyorum artık.",
            "Gerçekten sabrım taştı.",
            "Bu son uyarım, karar verdim.",
            "Yarın hattımı taşıyacağım.",
            "Sizinle çalışmaktan vazgeçtim.",
            "İyi günler, kararımı verdim.",
            "Bu konuşmayı burada bitiriyorum.",
            "Teşekkürler, hoşça kalın.",
        ]
    )

    short_result = analyze_churn_risk(short_call)
    long_result = analyze_churn_risk(long_call)

    assert short_result.confidence == "Düşük"
    assert long_result.confidence in ("Orta", "Yüksek")


def test_own_company_name_excluded_from_competitor_bonus():
    """Regression test: found via real Vodafone outbound sales call audio -
    an agent introducing their own employer ("Vodafone'dan arıyorum") must
    not be scored as a competitor mention when the deployment's own company
    name is configured."""
    segments = _segments(
        [
            "Merhabalar, Vodafone'dan arıyorum.",
            "Size özel bir kampanyamız var, bilgi vermek istedim.",
            "Paketinizi yükseltmek ister misiniz?",
            "Teşekkürler, iyi günler.",
        ]
    )

    with patch.object(settings, "churn_own_company_names", ""):
        result_without_exclusion = analyze_churn_risk(segments)
    assert "vodafone" in result_without_exclusion.competitors_mentioned

    with patch.object(settings, "churn_own_company_names", "vodafone"):
        result_with_exclusion = analyze_churn_risk(segments)
    assert "vodafone" not in result_with_exclusion.competitors_mentioned
    assert result_with_exclusion.risk_breakdown["competitor_bonus"] == 0.0
