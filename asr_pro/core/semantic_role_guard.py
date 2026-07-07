"""
Semantic + Acoustic Dual-Guard Role Correction Engine.

Evaluates assigned speaker segments against Turkish contact center lexical & sentiment
profiles. If acoustic overlap/crosstalk assigned a customer complaint or question
to the Agent (or an agent support script to the Customer), this guard automatically
flips the assignment to the correct role and flags auto_corrected=True.
Also calculates simultaneous speech (interruption) badges across segments.
"""

from typing import Any

from loguru import logger

# Unambiguous Turkish contact center Agent phrases
AGENT_STRONG_PHRASES = [
    "vodafone müşteri hizmetleri",
    "müşteri temsilcisi",
    "benim adım",
    "nasıl yardımcı olabilirim",
    "yardımcı olacağım",
    "kayıt altına alınmaktadır",
    "kontrol ediyorum",
    "sistemimize bakıyorum",
    "hesabınıza bakıyorum",
    "baktım",
    "işleminizi gerçekleştiriyorum",
    "tarafınıza iletildi",
    "aktarıyorum",
    "yönlendiriyorum",
    "bağlıyorum",
    "hatta kalın",
    "bekleteceğim",
    "güvenlik teyidi",
    "doğum tarihinizi",
    "anne kızlık soyadı",
]

# Unambiguous Turkish Customer complaint & request phrases
CUSTOMER_STRONG_PHRASES = [
    "şikayetçiyim",
    "faturam çok yüksek",
    "iptal ettirmek istiyorum",
    "hat çekmiyor",
    "internetime bağlanamıyorum",
    "paramı iade",
    "neden kestiniz",
    "niye çekildi",
    "rezalet",
    "berbat",
    "bir türlü olmuyor",
    "hâlâ çözülmedi",
    "tüketici haklarına",
    "yetkili biriyle",
    "müdürünüzle görüşmek",
]


def enforce_semantic_role_guard(
    segments: list[Any], agent_id: str | None, customer_id: str | None
) -> list[Any]:
    """Inspect and auto-correct speaker roles based on strong semantic evidence.
    Also computes interruption (söz kesme) moments between speakers.
    """
    if not segments:
        return segments

    corrected_count = 0
    interruption_count = 0
    refined = []

    for idx, seg in enumerate(segments):
        text = getattr(seg, "text", "") or ""
        text_lower = text.lower().strip()
        current_spk = getattr(seg, "speaker", None)

        # 1. Semantic role auto-correction
        new_spk = current_spk
        auto_corrected = getattr(seg, "auto_corrected", False)

        if agent_id and customer_id and agent_id != customer_id and text_lower:
            has_agent_phrase = any(phrase in text_lower for phrase in AGENT_STRONG_PHRASES)
            has_cust_phrase = any(phrase in text_lower for phrase in CUSTOMER_STRONG_PHRASES)

            if current_spk == agent_id and has_cust_phrase and not has_agent_phrase:
                new_spk = customer_id
                auto_corrected = True
                corrected_count += 1
                logger.debug(f"SemanticGuard: Flipped Agent -> Customer for: '{text[:40]}...'")
            elif current_spk == customer_id and has_agent_phrase and not has_cust_phrase:
                new_spk = agent_id
                auto_corrected = True
                corrected_count += 1
                logger.debug(f"SemanticGuard: Flipped Customer -> Agent for: '{text[:40]}...'")

        # 2. Check interruption against previous segment using corrected roles
        is_interrupted = False
        if idx > 0:
            prev_seg = refined[idx - 1]
            prev_spk = getattr(prev_seg, "speaker", None)
            prev_end = float(getattr(prev_seg, "end", 0))
            cur_start = float(getattr(seg, "start", 0))
            if prev_spk and new_spk and prev_spk != new_spk and cur_start < (prev_end - 0.20):
                is_interrupted = True
                interruption_count += 1

        # Update segment attributes
        import dataclasses

        if dataclasses.is_dataclass(seg):
            try:
                replace_kwargs = {"speaker": new_spk}
                if hasattr(seg, "auto_corrected"):
                    replace_kwargs["auto_corrected"] = auto_corrected
                if hasattr(seg, "is_interruption"):
                    replace_kwargs["is_interruption"] = is_interrupted
                seg = dataclasses.replace(seg, **replace_kwargs)
            except Exception as exc:
                logger.debug(f"SemanticGuard dataclass replace failed: {exc}")
        elif hasattr(seg, "_replace"):
            try:
                replace_kwargs = {"speaker": new_spk}
                if hasattr(seg, "_fields") and "auto_corrected" in seg._fields:
                    replace_kwargs["auto_corrected"] = auto_corrected
                if hasattr(seg, "_fields") and "is_interruption" in seg._fields:
                    replace_kwargs["is_interruption"] = is_interrupted
                seg = seg._replace(**replace_kwargs)
            except Exception:
                pass
        elif isinstance(seg, dict):
            seg["speaker"] = new_spk
            seg["auto_corrected"] = auto_corrected
            seg["is_interruption"] = is_interrupted
        else:
            try:
                seg.speaker = new_spk
                if hasattr(seg, "auto_corrected"):
                    seg.auto_corrected = auto_corrected
                if hasattr(seg, "is_interruption"):
                    seg.is_interruption = is_interrupted
            except Exception:
                pass

        refined.append(seg)

    if corrected_count > 0 or interruption_count > 0:
        logger.info(
            f"SemanticRoleGuard: Auto-corrected {corrected_count} roles, detected {interruption_count} interruptions."
        )

    return refined
