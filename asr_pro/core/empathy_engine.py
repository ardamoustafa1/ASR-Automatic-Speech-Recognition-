# Quantifies agent empathy and active listening skills during customer support interactions.
from __future__ import annotations

"""Absolute Zero-Error Deep AI Empathy & Soft Skill Engine."""


from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.sentiment_engine import SentimentClassifier


@dataclass(frozen=True)
class EmpathyResult:
    score: int  # 0 to 100
    active_listening_hits: tuple[str, ...]
    compassion_hits: tuple[str, ...]
    solution_hits: tuple[str, ...]
    defensive_hits: tuple[str, ...]
    interruption_count: int
    crisis_management_bonus: int
    high_wpm_penalty: int
    agent_wpm_avg: int
    analysis_summary: str


EMPATHY_DICTIONARY = {
    "active_listening": [
        "sizi anlıyorum",
        "çok haklısınız",
        "dinliyorum",
        "teşekkür ederim",
        "kesinlikle",
        "tabii ki",
        "evet anladım",
    ],
    "compassion": [
        "çok üzgünüm",
        "kusura bakmayın",
        "özür dileriz",
        "yaşadığınız durum",
        "rahatsızlık için",
        "mağduriyet",
        "beklettiğim için",
    ],
    "solution": [
        "yardımcı olacağım",
        "hemen kontrol ediyorum",
        "birlikte çözeceğiz",
        "halledeceğim",
        "işleminizi başlatıyorum",
        "hızla çözüyoruz",
        "çözüm sunacağım",
    ],
    "defensive": [
        "sistem böyle",
        "yapacak bir şey yok",
        "daha önce de söyledim",
        "anlattığım gibi",
        "bizim hatamız değil",
        "prosedür gereği",
        "elimizden gelen bu",
    ],
}


def _fuzzy_match(word: str, text: str, threshold: float = 0.85) -> bool:
    word_len = len(word.split())
    text_words = text.lower().split()

    if len(text_words) < word_len:
        return SequenceMatcher(None, text.lower(), word).ratio() > threshold

    for i in range(len(text_words) - word_len + 1):
        window = " ".join(text_words[i : i + word_len])
        if SequenceMatcher(None, window, word).ratio() > threshold:
            return True
    return False


def analyze_soft_skills(
    segments: Sequence[SegmentInput], agent_speaker_id: str | None = None, use_ai: bool = True
) -> EmpathyResult:
    """Absolute Zero-Error (Behavioral AI) Empathy Analysis."""
    if not segments:
        return EmpathyResult(50, (), (), (), (), 0, 0, 0, 0, "Konuşma verisi yok.")

    active_listening_hits = set()
    compassion_hits = set()
    solution_hits = set()
    defensive_hits = set()

    total_wpm = 0
    wpm_count = 0
    high_wpm_segments = 0
    interruption_count = 0
    crisis_bonus_pts = 0

    classifier = SentimentClassifier.get_instance() if use_ai else None

    last_speaker = None
    last_end_time = 0.0
    last_customer_was_angry = False

    # Müşteriyi tespit et (Eğer agent ID yoksa, SPEAKER_00 genelde müşteri, SPEAKER_01 temsilci kabul edilir)
    # Daha akıllısı: Agent kimse, diğeri müşteridir.

    for seg in segments:
        text = seg.text or ""
        text_lower = text.lower()
        if len(text_lower.strip()) < 2:
            continue

        # Kimin konuştuğunu belirle
        is_agent = False
        if agent_speaker_id:
            is_agent = seg.speaker == agent_speaker_id
        else:
            # Otomatik çıkarım (Eğer ID yoksa tüm konuşmaları değerlendiriyoruz, ancak kesmeleri speaker değişimiyle yakalarız)
            # Analiz için tüm segmentleri "agent" gibi puanlıyoruz eğer agent ID verilmemişse.
            is_agent = True

        # 1. AKUSTİK SÖZ KESME (INTERRUPTION) ANALİZİ
        if seg.speaker and last_speaker and seg.speaker != last_speaker:
            # Overlap toleransı: 0.2 saniye. Eğer daha fazla iç içe geçmişlerse söz kesmedir!
            overlap = last_end_time - seg.start
            if overlap > 0.2 and is_agent:
                interruption_count += 1

        # 2. MÜŞTERİ DUYGU DURUMU (Kriz Yönetimi İçin)
        if not is_agent and classifier and len(text.split()) > 3:
            hyp = "This customer is feeling {}."
            res = classifier.predict(text, labels=["angry", "frustrated", "calm"], hypothesis=hyp)
            scores = dict(zip(res["labels"], res["scores"]))
            if scores.get("angry", 0) > 0.6 or scores.get("frustrated", 0) > 0.6:
                last_customer_was_angry = True
            else:
                last_customer_was_angry = False

        if is_agent:
            segment_had_positive_empathy = False

            # Fuzzy Kelime Analizi
            for phrase in EMPATHY_DICTIONARY["active_listening"]:
                if _fuzzy_match(phrase, text_lower):
                    active_listening_hits.add(phrase)
                    segment_had_positive_empathy = True

            for phrase in EMPATHY_DICTIONARY["compassion"]:
                if _fuzzy_match(phrase, text_lower):
                    compassion_hits.add(phrase)
                    segment_had_positive_empathy = True

            for phrase in EMPATHY_DICTIONARY["solution"]:
                if _fuzzy_match(phrase, text_lower):
                    solution_hits.add(phrase)
                    segment_had_positive_empathy = True

            # 3. SENTIMENT MIRRORING (KRİZ YÖNETİMİ BONUSU)
            if last_customer_was_angry and segment_had_positive_empathy:
                crisis_bonus_pts += 15
                last_customer_was_angry = False  # Bonusu aldık, krizi yatıştırdı varsay

            # 4. PASİF AGRESİF NİYET FİLTRESİ (Zero-Error False Positive Engeli)
            for phrase in EMPATHY_DICTIONARY["defensive"]:
                if _fuzzy_match(phrase, text_lower):
                    if classifier and len(text.split()) > 3:
                        # Bu gerçekten defansif mi yoksa teknik açıklama mı?
                        hyp = "This customer service response represents a {}."
                        res = classifier.predict(
                            text,
                            labels=["passive aggressive attitude", "technical explanation"],
                            hypothesis=hyp,
                        )
                        scores = dict(zip(res["labels"], res["scores"]))
                        if scores.get("passive aggressive attitude", 0) > 0.6:
                            defensive_hits.add(phrase)  # Sadece gerçekten agresifse cezalandır!
                    else:
                        defensive_hits.add(phrase)  # AI yoksa doğrudan cezalandır

            # Akustik Baştan Savma Kontrolü (WPM)
            words = text.split()
            duration = max(0.5, seg.end - seg.start)
            wpm = int((len(words) / duration) * 60)
            total_wpm += wpm
            wpm_count += 1
            if wpm > 170:
                high_wpm_segments += 1

        # Güncellemeler
        last_speaker = seg.speaker
        last_end_time = max(last_end_time, seg.end)

    # --- SKOR HESAPLAMA MANTIĞI (Apple / Goldman Sachs Standartları) ---
    base_score = 50

    score_active = min(20, len(active_listening_hits) * 10)
    score_compassion = min(30, len(compassion_hits) * 15)
    score_solution = min(30, len(solution_hits) * 15)

    penalty_defensive = min(40, len(defensive_hits) * 20)
    penalty_wpm = min(20, high_wpm_segments * 5)
    penalty_interruption = min(30, interruption_count * 15)  # Söz kesmek çok büyük cezadır

    bonus_crisis = min(30, crisis_bonus_pts)

    final_score = (
        base_score
        + score_active
        + score_compassion
        + score_solution
        + bonus_crisis
        - penalty_defensive
        - penalty_wpm
        - penalty_interruption
    )
    final_score = max(0, min(100, final_score))

    agent_wpm_avg = int(total_wpm / wpm_count) if wpm_count > 0 else 0

    summary_parts = []
    if final_score >= 85:
        summary_parts.append(
            "🏆 Kusursuz İletişim (Top-Tier): Temsilci mükemmel bir empati sergiledi."
        )
    elif final_score >= 65:
        summary_parts.append(
            "Temsilcinin iletişimi standart seviyede, iyileştirmeye açık alanlar var."
        )
    else:
        summary_parts.append(
            "⚠️ Zayıf İletişim: Müşteriye karşı agresif, umursamaz veya savunmacı bir tutum sergilendi."
        )

    if bonus_crisis > 0:
        summary_parts.append(
            "🚨 Kriz Yönetimi Başarılı: Müşteri öfkeliyken profesyonelce şefkat gösterildi (+Bonus)."
        )
    if penalty_interruption > 0:
        summary_parts.append(
            f"🛑 Agresif Söz Kesme: Temsilci {interruption_count} kez müşterinin lafını bitirmesine izin vermedi (Büyük Ceza)."
        )
    if penalty_wpm > 0:
        summary_parts.append(
            f"⏱️ Baştan Savma: Temsilci {high_wpm_segments} kez çok hızlı konuştu (Rushing)."
        )

    return EmpathyResult(
        score=final_score,
        active_listening_hits=tuple(active_listening_hits),
        compassion_hits=tuple(compassion_hits),
        solution_hits=tuple(solution_hits),
        defensive_hits=tuple(defensive_hits),
        interruption_count=interruption_count,
        crisis_management_bonus=bonus_crisis,
        high_wpm_penalty=penalty_wpm,
        agent_wpm_avg=agent_wpm_avg,
        analysis_summary=" ".join(summary_parts),
    )
