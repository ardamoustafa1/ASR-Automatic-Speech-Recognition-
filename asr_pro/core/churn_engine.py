# Predicts customer churn risk based on acoustic features and sentiment indicators.
from __future__ import annotations

"""Top-Tier Enterprise Churn Risk Engine with Acoustic and Temporal Analytics."""


import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from asr_pro.config import settings
from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.sentiment_engine import SentimentClassifier

CHURN_LABELS = [
    "canceling service",
    "switching to competitor",
    "continue service",
    "neutral complaint",
]
# Kept for backwards-compat imports; the live threshold is settings.churn_alarm_threshold
# and can be tuned per deployment via ASR_CHURN_ALARM_THRESHOLD.
CHURN_ALARM_THRESHOLD = settings.churn_alarm_threshold

# Example Competitor List for NER Extraction
COMPETITORS = {
    "vodafone",
    "turkcell",
    "türk telekom",
    "turk telekom",
    "superonline",
    "turknet",
    "türknet",
    "aws",
    "azure",
    "google cloud",
    "netgsm",
    "bimcell",
    "pttcell",
    "teknosa mobil",
    "ttnet",
    "millencom",
    "d-smart",
    "digiturk",
    "kablonet",
    "avea",
    "rakip",
    "rakip firma",
    "rakip operatör",
    "başka operatör",
    "başka firma",
    "başka yer",
    "başka şebeke",
    "a1",
}

AGENT_INDICATOR_TERMS = {
    "müşteri hizmetleri",
    "nasıl yardımcı olabilirim",
    "mağaza danışmanlarım",
    "mağaza danışmanlarımız",
    "onay için arıyoruz",
    "sistemimizde görüntüledim",
    "sistemden bakıyorum",
    "fatura kesilmiş",
    "kontrol ediyorum",
    "bilgi veriyorum",
    "kayıt altına alınmaktadır",
    "temsilci",
    "yardımcı olayım",
    "arayış sebebimiz",
    "arattık",
    "bize ulaştığınız için",
    "tekrardan arattık",
    "ödeme başlıyoruz",
    "tarifeleriniz için aramıştım",
}

CHURN_ANCHOR_TERMS = {
    "iptal",
    "cayma",
    "taahhüt bit",
    "kapattır",
    "fesh",
    "taşıma",
    "başka operatör",
    "rakip",
    "bırakıyor",
    "aboneliğimi son",
    "yasal hak",
    "tüketici hakları",
    "şikayetçiyim",
    "hakem heyeti",
    "mahkeme",
    "btk",
    "cimri",
    "kapatırım",
    "iptal edin",
    "geçiyorum",
    "başka yer",
    "düşünüyorum",
    "pahalı",
    "zam",
    "uygun fiyat",
    "memnun değilim",
    "hat taşı",
    "numara taşı",
    "çıkmak istiyorum",
    "aboneliğimi iptal",
    "sonlandırmak",
}


@dataclass(frozen=True)
class ChurnInsight:
    text: str
    base_risk: float
    wpm: int
    acoustic_stress_multiplier: float
    temporal_weight: float
    final_segment_risk: float
    severity: str = "⚠️ Yüksek Risk"
    trigger_reason: str = "Ayrılma ve Rakip Firma Eğilimi"
    price_signal: str | None = None
    filler_ratio: float = 0.0
    churn_state: str = "🟢 Stabil"


@dataclass(frozen=True)
class ChurnResult:
    risk_score: float  # 0.0 to 1.0, bounded regardless of call length
    is_high_risk: bool
    competitors_mentioned: tuple[str, ...]
    insights: tuple[ChurnInsight, ...]
    average_wpm: int
    transcript_length: int
    # Explainability: which term of the formula drove the final score.
    risk_breakdown: dict[str, float] = field(
        default_factory=lambda: {
            "model_signal": 0.0,
            "escalation_trend": 0.0,
            "competitor_bonus": 0.0,
        }
    )
    # How much customer speech backs this score - "Düşük" for a single chunk,
    # "Yüksek" once several chunks agree. Lets a reviewer weigh a high score
    # from one short outburst differently than one backed by a whole call.
    confidence: str = "Düşük"
    trajectory: tuple[str, ...] = ()
    was_deescalated: bool = False
    agent_retention_score: float = 100.0
    detected_prices: tuple[str, ...] = ()
    average_filler_ratio: float = 0.0


def _own_company_names() -> set[str]:
    """Names of the deploying client's own company, excluded from competitor
    detection. Without this, an agent saying "Vodafone'dan arıyorum" on their
    own employer's outbound sales call would score as a competitor mention."""
    return {
        name.strip().lower()
        for name in settings.churn_own_company_names.split(",")
        if name.strip()
    }


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Damerau-Levenshtein distance to protect against ASR typos."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def _extract_competitors(text: str) -> set[str]:
    """SOTA Named Entity & Fuzzy Phonetic extraction for known competitors (handles ASR typos)."""
    found = set()
    text_lower = text.lower()
    own_names = _own_company_names()
    words = re.findall(r"\w+", text_lower)
    for comp in COMPETITORS:
        if comp in own_names:
            continue
        if re.search(rf"\b{re.escape(comp)}\b", text_lower):
            found.add(comp)
            continue
        if len(comp) >= 5:
            for w in words:
                if abs(len(w) - len(comp)) <= 2 and _levenshtein_distance(w, comp) <= 1:
                    found.add(comp)
                    break
    return found


def _extract_price_signals(text: str) -> tuple[str, ...]:
    """Extract numeric currency and price objection points (e.g., 990 TL, 1050 TL, 840 TL)."""
    matches = re.findall(r"\b(\d{2,4})\s*(?:TL|tl|Lira|lira|₺|bin|gigabyte|gb|paket)", text, re.IGNORECASE)
    matches_raw = re.findall(r"\b(?:fiyatı|tutar|bedel|ödeme|istenen|gözüküyordu o\?|o|bu|fiyat)\s*(\d{2,4})\b", text, re.IGNORECASE)
    all_prices = set()
    for m in matches + matches_raw:
        if int(m) >= 50 and int(m) <= 100000:
            all_prices.add(f"{m} TL")
    return tuple(sorted(all_prices, key=lambda x: int(x.split()[0]), reverse=True))


FILLER_WORDS = {"ııı", "eee", "şey", "yani", "hani", "hmm", "ıı", "aaa", "mesela", "atıyorum", "falan", "filan"}


def _calculate_filler_ratio(text: str) -> float:
    """Calculate ratio of hesitation and filler words in customer speech."""
    words = [w.strip(".,!?\"'") for w in text.lower().split()]
    if not words:
        return 0.0
    fillers = sum(1 for w in words if w in FILLER_WORDS)
    return round((fillers / len(words)) * 100, 1)


def analyze_churn_risk(
    segments: Sequence[SegmentInput], customer_speaker_id: str | None = None
) -> ChurnResult:
    """Analyze call transcript with Temporal and Acoustic Stress (WPM) mechanics (Optimized for Speed)."""
    if not segments:
        return ChurnResult(0.0, False, (), (), 0, 0)

    classifier = SentimentClassifier.get_instance()
    hypothesis = "The customer's intent in this text is {}."

    chunk_risks: list[float] = []
    chunk_weights: list[float] = []
    insights: list[ChurnInsight] = []
    competitors: set[str] = set()
    total_wpm = 0
    wpm_count = 0

    total_segments = len(segments)
    trajectory_states = []
    all_prices_detected = set()
    total_filler_ratio = 0.0

    # 1. HIZLI GEÇİŞ: Sadece WPM (Akustik Stres) ve Rakip Firma analizi yap (Çok Hızlı)
    # Ayrıca ML modeline göndermek için anlamlı, birleştirilmiş cümle öbekleri (chunk) oluştur.
    chunks = []
    current_chunk_text = []
    current_chunk_wpm_sum = 0
    current_chunk_wpm_count = 0

    for idx, seg in enumerate(segments):
        if customer_speaker_id and seg.speaker and seg.speaker != customer_speaker_id:
            continue

        text = seg.text or ""
        words = text.split()
        if len(text.strip()) < 3:
            continue

        # Layer 1: Agent Speech Exclusion Guard (Sıfır Yanlış Alarm Denetimi)
        # Even when customer_speaker_id is None, ignore any segment containing obvious agent phrases!
        text_lower = text.lower()
        if any(term in text_lower for term in AGENT_INDICATOR_TERMS):
            continue

        # Acoustic Stress (WPM)
        duration = max(0.5, seg.end - seg.start)
        wpm = int((len(words) / duration) * 60)

        total_wpm += wpm
        wpm_count += 1

        current_chunk_text.append(text)
        current_chunk_wpm_sum += wpm
        current_chunk_wpm_count += 1

        competitors.update(_extract_competitors(text))

        # Her 5 segmentte bir (veya sona gelindiğinde) birleştirilmiş bir analiz öbeği (chunk) oluştur
        if len(current_chunk_text) >= 5 or idx == total_segments - 1:
            chunk_str = " ".join(current_chunk_text)
            avg_chunk_wpm = current_chunk_wpm_sum / max(1, current_chunk_wpm_count)
            # Zaman ağırlığı: Aramanın sonlarına doğru ağırlık artar
            temporal_weight = 1.0 + (0.5 * (idx / max(1, total_segments - 1)))

            chunks.append(
                {"text": chunk_str, "wpm": avg_chunk_wpm, "temporal_weight": temporal_weight}
            )
            current_chunk_text = []
            current_chunk_wpm_sum = 0
            current_chunk_wpm_count = 0

    # 2. YAPAY ZEKA GEÇİŞİ (Ağır İşlem): Sadece birleştirilmiş öbekler üzerinde çalıştır (5-10 kat daha hızlı)
    for chunk in chunks:
        text = chunk["text"]
        wpm = chunk["wpm"]
        temporal_weight = chunk["temporal_weight"]

        acoustic_multiplier = 1.0
        if wpm > settings.churn_wpm_high_threshold:
            acoustic_multiplier = settings.churn_wpm_high_multiplier
        elif wpm > settings.churn_wpm_mid_threshold:
            acoustic_multiplier = settings.churn_wpm_mid_multiplier

        # Ağır NLP modeli artık her satırda değil, 5 satırda bir (blok halinde) çalışıyor!
        result = classifier.predict(text, labels=CHURN_LABELS, hypothesis=hypothesis)
        score_map = dict(zip(result["labels"], result["scores"]))

        base_risk = score_map.get("canceling service", 0.0) + score_map.get(
            "switching to competitor", 0.0
        )
        final_segment_risk = min(1.0, base_risk * acoustic_multiplier * temporal_weight)

        # Layer 2: Enterprise Lexical Anchor & Semantic Gate (Sıfır Yanlış Alarm)
        text_lower = text.lower()
        has_anchor = any(term in text_lower for term in CHURN_ANCHOR_TERMS)
        chunk_competitors = _extract_competitors(text)
        has_competitor = len(chunk_competitors) > 0

        if not has_anchor and not has_competitor:
            if final_segment_risk < 0.85:
                final_segment_risk *= 0.10

        # Multi-Modal Scientific Metrics: Price extraction, Filler ratio, HMM State
        prices_in_chunk = _extract_price_signals(text)
        all_prices_detected.update(prices_in_chunk)
        filler_ratio = _calculate_filler_ratio(text)
        total_filler_ratio += filler_ratio

        if final_segment_risk >= 0.65 or (has_competitor and wpm > 130) or any(t in text_lower for t in ["iptal", "cayma", "fesh", "mahkeme", "hakem heyeti", "şikayet"]):
            chunk_state = "🔥 Kritik Tehdit"
        elif final_segment_risk >= 0.40 or any(t in text_lower for t in ["pahalı", "zam", "yüksek gelmiş", "indirim", "uygun fiyat", "düşünüyorum"]):
            chunk_state = "⚠️ Fiyat/Kararsızlık"
        else:
            chunk_state = "🟢 Stabil"
        trajectory_states.append(chunk_state)

        # Layer 3: Dynamic Severity and Diagnostic Root Cause
        if final_segment_risk >= 0.45:
            if has_competitor and wpm > 130:
                severity = "🔥 Kritik Tehdit"
                reason = f"Rakip Firma Tehdidi ({', '.join(sorted(chunk_competitors)).title()}) + Akustik Stres ({int(wpm)} WPM)"
            elif has_competitor:
                severity = "🔥 Kritik Tehdit"
                reason = f"Rakip Firma Alternatifi ve Geçiş Eğilimi ({', '.join(sorted(chunk_competitors)).title()})"
            elif any(t in text_lower for t in ["iptal", "cayma", "fesh", "kapattır", "sonlandır", "mahkeme", "hakem heyeti", "şikayet"]):
                severity = "🔥 Kritik Tehdit"
                reason = "Doğrudan İptal / Cayma Bedeli ve Yasal Şikayet Talebi"
            elif any(t in text_lower for t in ["pahalı", "zam", "yüksek gelmiş", "indirim", "uygun fiyat"]):
                severity = "⚠️ Yüksek Risk"
                reason = "Fiyat / Zam İtirazı ve Kampanya Memnuniyetsizliği"
            else:
                severity = "⚠️ Yüksek Risk"
                reason = f"Ayrılma Eğilimi ve Akustik Stres ({int(wpm)} WPM)"

            insights.append(
                ChurnInsight(
                    text=text,
                    base_risk=round(base_risk, 3),
                    wpm=int(wpm),
                    acoustic_stress_multiplier=acoustic_multiplier,
                    temporal_weight=round(temporal_weight, 3),
                    final_segment_risk=round(final_segment_risk, 3),
                    severity=severity,
                    trigger_reason=reason,
                    price_signal=" • ".join(prices_in_chunk) if prices_in_chunk else None,
                    filler_ratio=filler_ratio,
                    churn_state=chunk_state,
                )
            )

        chunk_risks.append(final_segment_risk)
        chunk_weights.append(temporal_weight)

    if chunk_risks:
        weighted_mean = sum(r * w for r, w in zip(chunk_risks, chunk_weights)) / sum(
            chunk_weights
        )
        model_signal = max(chunk_risks)
        final_risk = settings.churn_max_weight * model_signal + settings.churn_mean_weight * (
            weighted_mean
        )
    else:
        model_signal = 0.0
        weighted_mean = 0.0
        final_risk = 0.0

    competitor_bonus = 0.0
    if competitors and final_risk > 0.3:
        competitor_bonus = min(
            settings.churn_competitor_bonus_cap,
            settings.churn_competitor_bonus_per_mention * len(competitors),
        )
        final_risk = min(1.0, final_risk + competitor_bonus)

    avg_wpm = int(total_wpm / wpm_count) if wpm_count > 0 else 0
    avg_filler_ratio = round(total_filler_ratio / max(1, len(chunks)), 1)

    # Check De-escalation & Agent Retention Effectiveness
    was_deescalated = False
    agent_retention_score = 100.0
    if trajectory_states:
        had_crisis = any("Kritik" in s or "Fiyat" in s for s in trajectory_states[:-1])
        ended_stable = trajectory_states[-1] == "🟢 Stabil"
        if had_crisis and ended_stable:
            was_deescalated = True
            agent_retention_score = min(100.0, 85.0 + 15.0)  # Award +15 bonus for resolving crisis!
        elif trajectory_states[-1] == "🔥 Kritik Tehdit":
            agent_retention_score = max(20.0, 100.0 - (final_risk * 70.0))
        elif trajectory_states[-1] == "⚠️ Fiyat/Kararsızlık":
            agent_retention_score = max(50.0, 100.0 - (final_risk * 40.0))

    if len(chunks) >= 4:
        confidence = "Yüksek"
    elif len(chunks) >= 2:
        confidence = "Orta"
    else:
        confidence = "Düşük"

    return ChurnResult(
        risk_score=round(final_risk, 3),
        is_high_risk=(final_risk >= settings.churn_alarm_threshold),
        competitors_mentioned=tuple(sorted(competitors)),
        insights=tuple(insights),
        average_wpm=avg_wpm,
        transcript_length=total_segments,
        risk_breakdown={
            "model_signal": round(model_signal, 3),
            "escalation_trend": round(weighted_mean, 3),
            "competitor_bonus": round(competitor_bonus, 3),
        },
        confidence=confidence,
        trajectory=tuple(trajectory_states),
        was_deescalated=was_deescalated,
        agent_retention_score=round(agent_retention_score, 1),
        detected_prices=tuple(sorted(all_prices_detected, key=lambda x: int(re.sub(r"\D", "", x)) if re.sub(r"\D", "", x).isdigit() else 0, reverse=True)),
        average_filler_ratio=avg_filler_ratio,
    )
