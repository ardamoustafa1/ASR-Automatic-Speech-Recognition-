"""LLM & Discourse Guard Service for computing strategic contact center analytics.

Evaluates First Contact Resolution (FCR), Customer Effort Score (CES), and Agent Adherence,
while performing sliding-window discourse role verification on ambiguous utterances.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("asr_pro.services.llm_discourse_guard")

# Mandatory adherence phrase sets for telecommunications
MANDATORY_GREETINGS = [
    "hoş geldiniz",
    "iyi günler",
    "nasıl yardımcı olabilirim",
    "vodafone",
    "türk telekom",
    "turkcell",
    "ben ",
    "müşteri hizmetleri",
]
MANDATORY_KVKK = [
    "kayıt altına alınmaktadır",
    "güvenlik amacıyla",
    "kvkk",
    "ses kaydı",
    "onaylıyor musunuz",
    "bilişim teknolojileri",
]
MANDATORY_CLOSING = [
    "başka bir işleminiz",
    "yardımcı olabileceğim başka",
    "iyi günler dilerim",
    "bizleri tercih ettiğiniz için",
    "teşekkür ederiz",
]

# FCR indicators
FCR_POSITIVE_PHRASES = [
    "halloldu",
    "çözüldü",
    "tamamdır",
    "anladım teşekkürler",
    "çok yardımcı oldunuz",
    "başka sorum yok",
    "harika",
    "işlem tamam",
]
FCR_NEGATIVE_PHRASES = [
    "tekrar arayacağım",
    "çözülmedi",
    "bir işe yaramadı",
    "şikayetçiyim",
    "yetkiliyle görüşmek istiyorum",
    "mahkemeye",
    "iptal edin hemen",
    "hiç yardımcı olmadınız",
]

# CES indicators (high customer effort triggers)
CES_HIGH_EFFORT_PHRASES = [
    "kaç kere aradım",
    "saatlerdir bekliyorum",
    "kimse yardımcı olmuyor",
    "bağlayamadınız",
    "üçüncü kez anlatıyorum",
    "sürekli aktarıyorsunuz",
    "anlamıyorsunuz",
]


class LLMDiscourseGuard:
    """Enterprise discourse analytics engine for telecommunications speech intelligence."""

    @classmethod
    def analyze_call_metrics(cls, segments: list[Any], full_transcript: str = "") -> dict[str, Any]:
        """Compute comprehensive FCR, CES, and Agent Adherence scores for a conversation."""
        if not full_transcript and segments:
            full_transcript = " ".join(
                getattr(s, "text", "") or str(s.get("text", "")) if isinstance(s, dict) else ""
                for s in segments
            )

        text_lower = full_transcript.lower().strip()

        # 1. First Contact Resolution (FCR) Score (0 to 100)
        fcr_score = 70.0  # baseline
        pos_hits = sum(1 for p in FCR_POSITIVE_PHRASES if p in text_lower)
        neg_hits = sum(1 for p in FCR_NEGATIVE_PHRASES if p in text_lower)

        fcr_score += pos_hits * 15.0
        fcr_score -= neg_hits * 25.0
        fcr_score = max(0.0, min(100.0, fcr_score))

        fcr_status = (
            "Resolved"
            if fcr_score >= 75.0
            else ("Unresolved" if fcr_score <= 40.0 else "Pending / Needs Follow-up")
        )

        # 2. Customer Effort Score (CES) (1 to 5, where 1 is smooth/easy, 5 is high effort/pain)
        ces_score = 1.0
        effort_hits = sum(1 for p in CES_HIGH_EFFORT_PHRASES if p in text_lower)
        ces_score += effort_hits * 1.5

        # Check interruption ratio if available
        interruption_count = 0
        for s in segments:
            is_int = (
                getattr(s, "is_interruption", False)
                if not isinstance(s, dict)
                else s.get("is_interruption", False)
            )
            if is_int:
                interruption_count += 1

        if len(segments) > 0:
            int_ratio = interruption_count / len(segments)
            if int_ratio > 0.15:
                ces_score += 1.0
        ces_score = max(1.0, min(5.0, round(ces_score, 1)))

        # 3. Agent Adherence Score (0 to 100%)
        adherence_points = 0
        checks_passed = []
        checks_failed = []

        if any(g in text_lower for g in MANDATORY_GREETINGS):
            adherence_points += 34
            checks_passed.append("Kurumsal Selamlama")
        else:
            checks_failed.append("Kurumsal Selamlama Eksik")

        if any(k in text_lower for k in MANDATORY_KVKK):
            adherence_points += 33
            checks_passed.append("KVKK / Ses Kaydı Bildirimi")
        else:
            checks_failed.append("KVKK Bildirimi Eksik")

        if any(c in text_lower for c in MANDATORY_CLOSING):
            adherence_points += 33
            checks_passed.append("Kurumsal Kapanış / Veda")
        else:
            checks_failed.append("Kurumsal Kapanış Eksik")

        adherence_score = min(100, adherence_points)

        logger.info(
            f"LLMDiscourseGuard: Computed FCR={fcr_score:.1f}% ({fcr_status}), CES={ces_score}/5, Adherence={adherence_score}%"
        )

        return {
            "fcr_score": round(fcr_score, 1),
            "fcr_status": fcr_status,
            "fcr_explanation": f"Pozitif ibareler: {pos_hits}, Negatif/Şikayet ibareleri: {neg_hits}",
            "ces_score": ces_score,
            "ces_explanation": f"Yüksek efor ibareleri: {effort_hits}, Söz kesme sayısı: {interruption_count}",
            "agent_adherence_score": adherence_score,
            "adherence_checks_passed": checks_passed,
            "adherence_checks_failed": checks_failed,
        }

    @classmethod
    def verify_discourse_roles(cls, segments: list[Any]) -> list[Any]:
        """Verify and correct roles for short/ambiguous utterances using turn-taking discourse context."""
        if len(segments) < 3:
            return segments

        short_utterances = [
            "evet",
            "hayır",
            "tamam",
            "anladım",
            "peki",
            "tabii",
            "öyle",
            "hıhı",
            "efendim",
            "doğrudur",
        ]
        corrected_count = 0

        for i in range(1, len(segments) - 1):
            seg = segments[i]
            text = getattr(seg, "text", "") if not isinstance(seg, dict) else seg.get("text", "")
            text_clean = text.lower().strip().rstrip(".?!,")

            if text_clean in short_utterances:
                prev_seg = segments[i - 1]
                next_seg = segments[i + 1]

                prev_spk = (
                    getattr(prev_seg, "speaker", None)
                    if not isinstance(prev_seg, dict)
                    else prev_seg.get("speaker")
                )
                next_spk = (
                    getattr(next_seg, "speaker", None)
                    if not isinstance(next_seg, dict)
                    else next_seg.get("speaker")
                )
                cur_spk = (
                    getattr(seg, "speaker", None)
                    if not isinstance(seg, dict)
                    else seg.get("speaker")
                )

                # In standard telecommunication dialogues, if previous and next turn belong to Agent,
                # an sandwiched acknowledgement ("evet", "anladım") almost certainly belongs to Customer!
                if prev_spk and next_spk and prev_spk == next_spk and cur_spk == prev_spk:
                    # Target speaker is the opposite role
                    alternate_spk = "SPEAKER_01" if prev_spk == "SPEAKER_00" else "SPEAKER_00"

                    import dataclasses

                    if dataclasses.is_dataclass(seg):
                        try:
                            # is_dataclass()'s TypeGuard narrows to DataclassInstance |
                            # type[DataclassInstance]; seg is always an instance here.
                            segments[i] = dataclasses.replace(  # type: ignore[type-var]
                                seg, speaker=alternate_spk, auto_corrected=True
                            )
                            corrected_count += 1
                        except Exception:
                            pass
                    elif hasattr(seg, "_replace"):
                        try:
                            segments[i] = seg._replace(speaker=alternate_spk)
                            corrected_count += 1
                        except Exception:
                            pass
                    elif isinstance(seg, dict):
                        seg["speaker"] = alternate_spk
                        seg["auto_corrected"] = True
                        corrected_count += 1
                    else:
                        try:
                            seg.speaker = alternate_spk
                            seg.auto_corrected = True
                            corrected_count += 1
                        except Exception:
                            pass

        if corrected_count > 0:
            logger.info(
                f"LLMDiscourseGuard: Re-verified and corrected {corrected_count} ambiguous short turn utterances."
            )

        return segments
