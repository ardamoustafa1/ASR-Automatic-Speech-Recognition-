# Core engine responsible for monitoring regulatory compliance in customer-agent conversations.
from __future__ import annotations

"""Enterprise Compliance & Regulatory Monitoring Engine with Zero-Error Tolerance."""


from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.sentiment_engine import SentimentClassifier

SeverityLevel = Literal["CRITICAL", "HIGH", "MEDIUM"]


@dataclass(frozen=True)
class ComplianceViolation:
    severity: SeverityLevel
    category: str
    reason: str
    segment_text: str
    timestamp_start: float
    timestamp_end: float


# Yasal Uyumluluk (Red Flag) Kırmızı Çizgi İfadeleri (Sektörlere Göre)
RED_FLAG_PATTERNS = {
    "finance": {
        "Yanıltıcı Yatırım Vaadi (SPK İhlali)": [
            "kesin kazandırır",
            "garanti getiri",
            "sıfır risk",
            "zarar etmezsiniz",
            "kesinlikle düşmez",
        ],
        "Veri Gizliliği (KVKK/BDDK İhlali)": [
            "kredi kartı şifre",
            "şifrenizi paylaş",
            "üç haneli güvenlik",
            "kartınızın arkasındaki",
        ],
    },
    "telecom": {
        "Sözleşme/İptal Engelleme (BTK İhlali)": [
            "iptal edemezsiniz",
            "taahhüt bozamazsınız",
            "çıkış yapamazsınız",
        ],
        "Agresif Ceza Tehdidi": [
            "cayma bedeli çok yüksek",
            "ciddi ceza öder",
            "mahkemelik olursunuz",
        ],
    },
    "health": {
        "Umut Tacirliği / Garanti (Sağlık Bak. İhlali)": [
            "kesin iyileştir",
            "yüzde yüz tedavi",
            "garanti veriyorum",
            "mucizevi sonuç",
        ]
    },
    "insurance": {
        "Kapsam Saptırma": ["her şeyi karşılar", "hiçbir şart yok", "tüm masrafları biz ödüyoruz"]
    },
    "general": {
        "Agresif Satış / Etik İhlal": [
            "hemen almazsanız yanarsınız",
            "bundan daha iyisini bulamazsın",
            "sen beni dinle",
        ]
    },
}

# The `sector` string a caller passes through save_conversation_with_analysis
# is shared across multiple engines with historically different vocabularies
# (DomainAdaptationService uses "banking"/"telecom"/"omni"; this engine's
# RED_FLAG_PATTERNS predates that and only recognizes "finance"/"telecom"/
# "health"/"insurance"/"general"). Without this alias table, a call tagged
# "banking" - the exact value a bank-facing UI would send - silently matched
# zero domain patterns (RED_FLAG_PATTERNS.get("banking", {}) == {}), so
# SPK/BDDK red-flag detection never fired for banking calls. "omni" (mixed
# contact center, no single sector) maps to "general" rather than a specific
# regulator's patterns, matching its meaning elsewhere in the system.
_DOMAIN_KEY_ALIASES = {
    "banking": "finance",
    "bank": "finance",
    "billing": "telecom",
    "omni": "general",
}


def _canonical_domain_key(domain_key: str) -> str:
    key = (domain_key or "general").strip().lower()
    return _DOMAIN_KEY_ALIASES.get(key, key)


# Sahte Alarmları Önleyen Olumsuzluk Kalkanı (False Positive Negators)
FALSE_POSITIVE_NEGATORS = {
    "değil",
    "değildir",
    "yok",
    "yoktur",
    "olmaz",
    "yasak",
    "yasaktır",
    "imkansız",
    "yapamayız",
    "vermiyoruz",
}

NLP_COMPLIANCE_LABELS = [
    "misleading promise",
    "aggressive sales tactic",
    "asking for password",
    "denying cancellation",
    "neutral conversation",
]


def _is_negated(text: str, match_index: int, match_length: int) -> bool:
    """Eşleşen ifadenin çevresindeki (3-4 kelime) kelimelere bakarak olumsuzluk eki var mı diye kontrol eder."""
    # Basitçe eşleşen cümlenin sağına ve soluna 50 karakterlik bir pencere açıyoruz
    window_start = max(0, text.rfind(" ", 0, max(0, match_index)) - 50)
    window_end = text.find(" ", min(len(text), match_index + match_length + 50))
    if window_end == -1:
        window_end = len(text)

    context_window = text[window_start:window_end].lower()

    # Noktalama işaretlerini temizleyelim (nokta, virgül vb. kelimeye bitişik kalmasın)
    import string

    context_window = context_window.translate(str.maketrans("", "", string.punctuation))

    context_words = set(context_window.split())

    # Eğer kalkan kelimelerden biri etrafta geçiyorsa bu bir uyarı/açıklama cümlesidir, ihlal değildir!
    if context_words.intersection(FALSE_POSITIVE_NEGATORS):
        return True
    return False


def _fuzzy_match_phrase(pattern: str, text: str) -> tuple[bool, int, int]:
    """ASR (Ses Tanıma) hatalarını affeden Token-Overlap tabanlı esnek eşleşme."""
    pattern_lower = pattern.lower()
    text_lower = text.lower()

    # Birebir geçiyorsa zaten en hızlısı bu (0ms)
    idx = text_lower.find(pattern_lower)
    if idx != -1:
        return True, idx, len(pattern_lower)

    # Birebir yoksa ASR kelime atlamış veya harf yutmuş olabilir. (Örn: "kredi kartı şifre" -> "kredi kart şifre")
    pattern_words = pattern_lower.split()
    text_words = text_lower.split()

    # Kayan pencere ile ardışık kelime dizilerini kontrol et (N-gram)
    n = len(pattern_words)
    if n > 1 and len(text_words) >= n:
        for i in range(len(text_words) - n + 1):
            window = text_words[i : i + n]
            # Sırf harf hatası (typo) toleransı için SequenceMatcher ile %85 benzerlik arıyoruz
            similarity = SequenceMatcher(None, " ".join(window), pattern_lower).ratio()
            if similarity > 0.85:
                # Eşleşme bulundu! Başlangıç ve bitiş indexini yaklaşık olarak bul
                approx_start = text_lower.find(window[0])
                approx_length = len(" ".join(window))
                return True, approx_start, approx_length

    return False, -1, 0


def analyze_compliance_risk(
    segments: Sequence[SegmentInput],
    domain_key: str = "general",
    use_ai: bool = True,
    agent_speaker_id: str | None = None,
) -> list[ComplianceViolation]:
    """Sıfır Hata Toleranslı Hibrid Uyum Motoru (Fuzzy Matching + Negation Filter + NLP Confidence Gate).

    Regulatory obligations (KVKK/BTK/SPK etc.) apply to what the AGENT says,
    not the customer - a customer asking "kredi kartı şifremi mi istiyorsunuz?"
    or quoting a red-flag phrase back must not be scored as a violation. When
    `agent_speaker_id` is provided, only that speaker's segments are checked.
    """
    if not segments:
        return []

    if agent_speaker_id:
        segments = [s for s in segments if not s.speaker or s.speaker == agent_speaker_id]
        if not segments:
            return []

    violations = []
    domain_patterns = RED_FLAG_PATTERNS.get(_canonical_domain_key(domain_key), {})
    general_patterns = RED_FLAG_PATTERNS.get("general", {})
    all_patterns = {**domain_patterns, **general_patterns}

    classifier = SentimentClassifier.get_instance() if use_ai else None

    # Risk keywords for pre-filtering (avoid calling ML on every segment)
    risk_keywords = [
        "iptal",
        "şifre",
        "sözleşme",
        "kart",
        "garanti",
        "ödeme",
        "zorunlu",
        "cezası",
        "kazanç",
        "kesin",
        "risk",
    ]

    # 1. FUZZY RED FLAG (DETERMINISTIC) + NEGATION FILTER
    ai_candidate_segments = []  # Collect segments needing AI analysis

    for seg in segments:
        text = seg.text or ""
        text_lower = text.lower()

        violation_found_in_segment = False

        for category, patterns in all_patterns.items():
            for pattern in patterns:
                is_match, m_idx, m_len = _fuzzy_match_phrase(pattern, text)

                if is_match:
                    # Eşleşti ama acaba yasal bir uyarı cümlesi mi? (False Positive Check)
                    if _is_negated(text_lower, m_idx, m_len):
                        continue  # Sahte alarm kalkanı devrede! İhlal yok.

                    violations.append(
                        ComplianceViolation(
                            severity="CRITICAL",
                            category=category,
                            reason="Kritik yasaklı ifade kullanımı tespit edildi.",
                            segment_text=text.strip(),
                            timestamp_start=seg.start,
                            timestamp_end=seg.end,
                        )
                    )
                    violation_found_in_segment = True
                    break

            if violation_found_in_segment:
                break

        # Collect AI candidate segments (only segments with risk keywords, not already flagged)
        words = text.split()
        if (
            not violation_found_in_segment
            and use_ai
            and classifier
            and len(words) > 5
            and any(k in text_lower for k in risk_keywords)
            and not _is_negated(text_lower, 0, len(text_lower))
        ):
            # Truncate text to avoid slow inference on very long segments
            ai_candidate_segments.append((seg, text[:400]))

    # 2. YAPAY ZEKA (CONTEXTUAL) DENETİMİ - Batch processing (much faster)
    if ai_candidate_segments and classifier:
        try:
            hypothesis = "This customer service response represents {}."
            for seg, text in ai_candidate_segments:
                res = classifier.predict(text, labels=NLP_COMPLIANCE_LABELS, hypothesis=hypothesis)
                score_map = dict(zip(res["labels"], res["scores"]))

                # Confidence Gate: Eşik değerleri eskisinden çok daha katı (%80)
                if score_map.get("misleading promise", 0.0) > 0.80:
                    violations.append(
                        ComplianceViolation(
                            severity="HIGH",
                            category="Yanıltıcı / Abartılı Vaat (AI Tespiti)",
                            reason="Yapay zeka, temsilcinin cümlesinde yüksek güvenilirlikle yanıltıcı bir vaat tespit etti.",
                            segment_text=text.strip(),
                            timestamp_start=seg.start,
                            timestamp_end=seg.end,
                        )
                    )
                elif score_map.get("asking for password", 0.0) > 0.85:
                    violations.append(
                        ComplianceViolation(
                            severity="CRITICAL",
                            category="Şüpheli Veri Talebi (KVKK)",
                            reason="Yapay zeka, müşteriden şifre veya hassas veri istenmiş olabileceğini tespit etti.",
                            segment_text=text.strip(),
                            timestamp_start=seg.start,
                            timestamp_end=seg.end,
                        )
                    )
        except Exception:
            pass  # AI analysis failure should not break compliance check

    return sorted(violations, key=lambda v: v.timestamp_start)
