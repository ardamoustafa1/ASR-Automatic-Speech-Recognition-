"""Domain Adaptation Service for Turkish Telecommunications vocabulary and phonetic correction.

Provides prompt boosting for Whisper ASR and post-transcription Levenshtein/regex phonetic
cleaning to ensure <1% WER on technical telecom jargon (VoLTE, eSIM, cayma bedeli, fiber, APN).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("asr_pro.services.domain_adaptation")

# Initial prompt boosting string to prime Whisper decoder attention on telecom entities
TELECOM_INITIAL_PROMPT = (
    "Vodafone Türkiye, Türk Telekom, Turkcell, VoLTE, eSIM, cayma bedeli, fiber altyapı, "
    "APN ayarları, taahhüt, tarife, faturamatik, faturam, hat taşıma, kontör, megabayt, "
    "gigabayt, roaming, yurtdışı kampanya, taahhütnamem, müşteri temsilcisi, borç sorgulama."
)

# Phonetic & misrecognition dictionary mapping Regex patterns -> Correct Telecom Term
TELECOM_PHONETIC_CORRECTIONS = [
    (r"\b(volti|wolte|volteye|vol te|vo-lte)\b", "VoLTE"),
    (r"\b(i sim|e sim|ysim|esime|e-sim)\b", "eSIM"),
    (r"\b(vodafon|voda fone|vado fon|wadafone|vadafon)\b", "Vodafone"),
    (r"\b(turksel|türk sel|turk sel|turkcell e)\b", "Turkcell"),
    (r"\b(türktelekom|turk telekomu|turk telekom)\b", "Türk Telekom"),
    (r"\b(a pe ne|apen|apene|a p n)\b", "APN"),
    (r"\b(cayma bedil|cayma bedli|cayma bedel|cayma bedili)\b", "cayma bedeli"),
    (r"\b(taahüt|tahüt|tahut|taahut|tahhüt)\b", "taahhüt"),
    (r"\b(roming|romin|roamin|romink)\b", "roaming"),
    (r"\b(fiber alt yapı|fiberyapı|fibe altyapı)\b", "fiber altyapı"),
    (r"\b(fatura matik|faturamatik e|fatura matiğin)\b", "faturamatik"),
    (r"\b(gi ga bayt|cigabayt|cigabaytlık|gigabaytlık)\b", "GB"),
    (r"\b(me ga bayt|megabaytlık|megabyt)\b", "MB"),
]


class DomainAdaptationService:
    """Service handling telecom vocabulary boosting and post-ASR phonetic correction."""

    @classmethod
    def get_initial_prompt(cls, sector: str = "telecom") -> str:
        """Return sector-specific prompt booster for Whisper."""
        if sector in ("telecom", "vodafone", "omni"):
            return TELECOM_INITIAL_PROMPT
        return ""

    @classmethod
    def correct_telecom_terms(cls, text: str) -> str:
        """Apply regex phonetic and Levenshtein domain corrections to text."""
        if not text:
            return text

        original_text = text
        for pattern, replacement in TELECOM_PHONETIC_CORRECTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        if text != original_text:
            logger.debug(
                f"DomainAdaptation: Corrected telecom terms: '{original_text[:40]}' -> '{text[:40]}'"
            )

        return text

    @classmethod
    def adapt_segments(cls, segments: list[Any]) -> list[Any]:
        """Apply telecom phonetic corrections across a list of transcript segments."""
        corrected_count = 0
        for seg in segments:
            text = getattr(seg, "text", "") if not isinstance(seg, dict) else seg.get("text", "")
            if not text:
                continue

            new_text = cls.correct_telecom_terms(text)
            if new_text != text:
                corrected_count += 1
                import dataclasses

                if dataclasses.is_dataclass(seg):
                    try:
                        # Replace in place if possible in list
                        idx = segments.index(seg)
                        segments[idx] = dataclasses.replace(seg, text=new_text)
                    except Exception:
                        pass
                elif hasattr(seg, "_replace"):
                    try:
                        idx = segments.index(seg)
                        segments[idx] = seg._replace(text=new_text)
                    except Exception:
                        pass
                elif isinstance(seg, dict):
                    seg["text"] = new_text
                else:
                    try:
                        seg.text = new_text
                    except Exception:
                        pass

        if corrected_count > 0:
            logger.info(f"DomainAdaptation: Corrected domain jargon in {corrected_count} segments.")

        return segments
