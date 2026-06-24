from __future__ import annotations
"""Top-Tier Enterprise Churn Risk Engine with Acoustic and Temporal Analytics."""


import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.sentiment_engine import SentimentClassifier

CHURN_LABELS = ["canceling service", "switching to competitor", "continue service", "neutral complaint"]
CHURN_ALARM_THRESHOLD = 0.75

# Example Competitor List for NER Extraction
COMPETITORS = {
    "vodafone", "turkcell", "türk telekom", "turk telekom", "superonline", 
    "turknet", "türknet", "aws", "azure", "google cloud"
}


@dataclass(frozen=True)
class ChurnInsight:
    text: str
    base_risk: float
    wpm: int
    acoustic_stress_multiplier: float
    temporal_weight: float
    final_segment_risk: float


@dataclass(frozen=True)
class ChurnResult:
    risk_score: float  # 0.0 to 1.0 (cumulative)
    is_high_risk: bool
    competitors_mentioned: tuple[str, ...]
    insights: tuple[ChurnInsight, ...]
    average_wpm: int
    transcript_length: int


def _extract_competitors(text: str) -> set[str]:
    """Simple Named Entity extraction for known competitors."""
    found = set()
    text_lower = text.lower()
    for comp in COMPETITORS:
        if re.search(rf"\b{re.escape(comp)}\b", text_lower):
            found.add(comp)
    return found


def analyze_churn_risk(segments: Sequence[SegmentInput], customer_speaker_id: Optional[str] = None) -> ChurnResult:
    """Analyze call transcript with Temporal and Acoustic Stress (WPM) mechanics (Optimized for Speed)."""
    if not segments:
        return ChurnResult(0.0, False, (), (), 0, 0)

    classifier = SentimentClassifier.get_instance()
    hypothesis = "The customer's intent in this text is {}."
    
    cumulative_risk = 0.0
    insights: list[ChurnInsight] = []
    competitors: set[str] = set()
    total_wpm = 0
    wpm_count = 0
    
    total_segments = len(segments)
    
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
            
            chunks.append({
                "text": chunk_str,
                "wpm": avg_chunk_wpm,
                "temporal_weight": temporal_weight
            })
            current_chunk_text = []
            current_chunk_wpm_sum = 0
            current_chunk_wpm_count = 0

    # 2. YAPAY ZEKA GEÇİŞİ (Ağır İşlem): Sadece birleştirilmiş öbekler üzerinde çalıştır (5-10 kat daha hızlı)
    for chunk in chunks:
        text = chunk["text"]
        wpm = chunk["wpm"]
        temporal_weight = chunk["temporal_weight"]
        
        acoustic_multiplier = 1.0
        if wpm > 190:
            acoustic_multiplier = 1.4
        elif wpm > 160:
            acoustic_multiplier = 1.2
            
        # Ağır NLP modeli artık her satırda değil, 5 satırda bir (blok halinde) çalışıyor!
        result = classifier.predict(text, labels=CHURN_LABELS, hypothesis=hypothesis)
        score_map = dict(zip(result["labels"], result["scores"]))
        
        base_risk = score_map.get("canceling service", 0.0) + score_map.get("switching to competitor", 0.0)
        final_segment_risk = base_risk * acoustic_multiplier * temporal_weight
        
        if final_segment_risk > 0.3:
            insights.append(ChurnInsight(
                text=text,
                base_risk=round(base_risk, 3),
                wpm=int(wpm),
                acoustic_stress_multiplier=acoustic_multiplier,
                temporal_weight=round(temporal_weight, 3),
                final_segment_risk=round(final_segment_risk, 3)
            ))
            cumulative_risk += (final_segment_risk * 0.6)

    final_risk = min(1.0, cumulative_risk)
    
    if competitors and final_risk > 0.4:
        final_risk = max(final_risk, 0.8)

    avg_wpm = int(total_wpm / wpm_count) if wpm_count > 0 else 0

    return ChurnResult(
        risk_score=round(final_risk, 3),
        is_high_risk=(final_risk >= CHURN_ALARM_THRESHOLD),
        competitors_mentioned=tuple(sorted(competitors)),
        insights=tuple(insights),
        average_wpm=avg_wpm,
        transcript_length=total_segments
    )

