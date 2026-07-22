"""Domain Adaptation Service for Turkish contact-center vocabulary and phonetic correction.

Provides sector-aware prompt boosting for Whisper ASR (telecom AND banking/finance)
plus post-transcription regex phonetic cleaning of high-value domain terms
(VoLTE, eSIM, cayma bedeli, IBAN, EFT, mevduat...). Corrections are precise,
checksum-free regexes chosen so they are safe to apply across all sectors.
"""

from __future__ import annotations

import contextvars
import logging
import re
from typing import Any

logger = logging.getLogger("asr_pro.services.domain_adaptation")

# Per-request correction counter, exposed to callers (e.g. the upload
# pipeline) as a QA metric - "how many words did the phonetic corrector fix
# in this call". A ContextVar (not an instance/module attribute) so
# concurrent requests sharing this singleton-style service never see each
# other's counts, matching the pattern used for _filtered_segment_count in
# asr_service.py.
_correction_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    "domain_adaptation_correction_count", default=0
)


def reset_correction_counter() -> None:
    _correction_count.set(0)


def get_correction_count() -> int:
    return _correction_count.get()


# Initial prompt boosting string to prime Whisper decoder attention on telecom entities
TELECOM_INITIAL_PROMPT = (
    "Vodafone Türkiye, Türk Telekom, Turkcell, Red tarifesi, VoLTE, eSIM, cayma bedeli, "
    "fiber altyapı, APN ayarları, taahhüt, tarife, sınırsız internet, faturamatik, faturam, "
    "hat taşıma, kontör, megabayt, gigabayt, roaming, yurtdışı kampanya, taahhütnamem, "
    "müşteri temsilcisi, borç sorgulama."
)

# Banking / finance vocabulary for bank deployments (KMH, EFT, IBAN, mevduat...)
BANKING_INITIAL_PROMPT = (
    "IBAN, EFT, FAST, havale, swift, vadesiz hesap, vadeli mevduat, kredili mevduat hesabı, "
    "kredi kartı, ekstre, asgari ödeme, faiz oranı, masraf iadesi, bloke, provizyon, "
    "kart aidatı, taksitlendirme, yapılandırma, BDDK, KVKK, internet bankacılığı, "
    "mobil şube, müşteri numarası, son ödeme tarihi."
)

_SECTOR_PROMPTS = {
    "telecom": TELECOM_INITIAL_PROMPT,
    "vodafone": TELECOM_INITIAL_PROMPT,
    "banking": BANKING_INITIAL_PROMPT,
    "bank": BANKING_INITIAL_PROMPT,
    "finance": BANKING_INITIAL_PROMPT,
    "billing": TELECOM_INITIAL_PROMPT,
    # omni = mixed contact center: prime with both vocabularies.
    "omni": f"{TELECOM_INITIAL_PROMPT} {BANKING_INITIAL_PROMPT}",
}

# Phonetic & misrecognition dictionary mapping Regex patterns -> Correct Term.
# Patterns are matched case-insensitively and MUST be specific enough that a
# false positive is implausible in any sector's transcript.
TELECOM_PHONETIC_CORRECTIONS = [
    (r"\b(volti|wolte|volteye|vol te|vo-lte)\b", "VoLTE"),
    (r"\b(i sim|e sim|ysim|esime|e-sim)\b", "eSIM"),
    (r"\b(vodafon|voda fone|vado fon|wadafone|vadafon|modafon|mudafon)\b", "Vodafone"),
    (r"\b(turksel|türk sel|turk sel|turkcell e)\b", "Turkcell"),
    (r"\b(türktelekom|turk telekomu|turk telekom)\b", "Türk Telekom"),
    (r"\b(a pe ne|apen|apene|a p n)\b", "APN"),
    (r"\b(cayma bedil|cayma bedli|cayma bedel|cayma bedili)\b", "cayma bedeli"),
    (r"\b(taahüt|tahüt|tahut|taahut|tahhüt)\b", "taahhüt"),
    (r"\b(taahütname|tahütname|taahutname)\b", "taahhütname"),
    (r"\b(roming|romin|roamin|romink)\b", "roaming"),
    (r"\b(fiber alt yapı|fiberyapı|fibe altyapı)\b", "fiber altyapı"),
    (r"\b(fatura matik|faturamatik e|fatura matiğin)\b", "faturamatik"),
    (r"\b(gi ga bayt|cigabayt|cigabaytlık|gigabaytlık)\b", "GB"),
    (r"\b(me ga bayt|megabaytlık|megabyt)\b", "MB"),
    # Whisper often mishears "sınırsız" (unlimited) as "sınıfsız" (classless) on
    # 8kHz telephony audio - "sınıfsız" has no legitimate use in contact-center
    # speech, observed 3x in a single production call.
    (r"\bsınıfsız\b", "sınırsız"),
    # Vodafone "Red" tariff spoken as "Red'le" (with Red) gets transcribed as
    # the non-words "redle"/"retle"/"retl" - observed on multiple production
    # calls ("Redle 50 kullanıyormuşuz", "RETL50"). None of these letter
    # sequences are Turkish words, so the rewrite is safe cross-sector.
    (r"\b(redle|retle|retl)(\d+)\b", r"Red'le \2"),
    (r"\b(redle|retle|retl)\b", "Red'le"),
    # Same "Red" tariff mis-heard with the "-li" suffix ("Red'li kullanıyorsunuz")
    # as the non-words "redley'li"/"redleyli"/"retli" - observed on real 8kHz
    # calls ("Redley'li kullanıyormuşuz", "retli 60 GB"). Not Turkish words.
    (r"\b(redley'?li|redleyli|retli)\b", "Red'li"),
    # Bare "Red" product name mis-heard as "redley" ("Redley 60 GB sınırsız").
    (r"\bredley\b", "Red"),
]

BANKING_PHONETIC_CORRECTIONS = [
    (r"\b(ay ban|ayban|i ban|iban numaram)\b", "IBAN"),
    (r"\b(e f t|efte|eftiyle)\b", "EFT"),
    (r"\b(havele|havala)\b", "havale"),
    (r"\b(mevduad|mefduat)\b", "mevduat"),
    (r"\b(kredili mevduat hesabi|ka me ha)\b", "kredili mevduat hesabı"),
    (r"\b(bide ka|bideka)\b", "BDDK"),
    (r"\bazgari\b", "asgari"),
    (r"\b(provizion|provüzyon)\b", "provizyon"),
]

ALL_PHONETIC_CORRECTIONS = TELECOM_PHONETIC_CORRECTIONS + BANKING_PHONETIC_CORRECTIONS


class DomainAdaptationService:
    """Service handling sector vocabulary boosting and post-ASR phonetic correction."""

    @classmethod
    def get_initial_prompt(cls, sector: str = "telecom") -> str:
        """Return sector-specific prompt booster for Whisper."""
        return _SECTOR_PROMPTS.get((sector or "").strip().lower(), _SECTOR_PROMPTS["omni"])

    @classmethod
    def correct_terms(cls, text: str) -> str:
        """Apply all sectors' regex phonetic corrections to text.

        Every pattern is domain-unambiguous, so applying the union is safe
        regardless of which sector the call belongs to.
        """
        if not text:
            return text

        original_text = text
        total_subs = 0
        for pattern, replacement in ALL_PHONETIC_CORRECTIONS:
            text, n_subs = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
            total_subs += n_subs

        if total_subs:
            _correction_count.set(_correction_count.get() + total_subs)
            logger.debug(
                f"DomainAdaptation: Corrected domain terms: '{original_text[:40]}' -> '{text[:40]}'"
            )

        return text

    # Backwards-compatible alias (predates banking corrections).
    @classmethod
    def correct_telecom_terms(cls, text: str) -> str:
        return cls.correct_terms(text)

    @classmethod
    def adapt_segments(cls, segments: list[Any]) -> list[Any]:
        """Apply phonetic corrections across a list of transcript segments."""
        corrected_count = 0
        for seg in segments:
            text = getattr(seg, "text", "") if not isinstance(seg, dict) else seg.get("text", "")
            if not text:
                continue

            new_text = cls.correct_terms(text)
            if new_text != text:
                corrected_count += 1
                import dataclasses

                if dataclasses.is_dataclass(seg):
                    try:
                        # Replace in place if possible in list
                        idx = segments.index(seg)
                        # is_dataclass()'s TypeGuard narrows to DataclassInstance |
                        # type[DataclassInstance]; seg is always an instance here.
                        segments[idx] = dataclasses.replace(seg, text=new_text)  # type: ignore[type-var]
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
