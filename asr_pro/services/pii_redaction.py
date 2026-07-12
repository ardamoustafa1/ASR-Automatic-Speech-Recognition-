"""KVKK / PCI-DSS PII redaction for call transcripts.

Banks and telecom operators may not persist card PANs (PCI-DSS req. 3.4) and
must minimize stored personal data (KVKK / GDPR data minimization). Customers
routinely read their national ID, card number, or IBAN aloud during identity
verification, so raw ASR transcripts are full of exactly the identifiers a
compliance audit flags.

Every match is VALIDATED before masking (TCKN checksum, Luhn, IBAN mod-97) so
ordinary numbers - amounts, dates, package sizes - are never touched. Digits
spoken with separators ("5 3 2 ... " or "5321 65 43") are matched too, since
Whisper transcribes read-aloud digit strings with spaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Candidate digit runs: digits possibly separated by spaces/dots/dashes.
# Bounded so we never regex-scan pathological inputs.
_DIGIT_RUN = re.compile(r"(?<!\d)(?:\d[ .\-]?){9,31}\d(?!\d)")
# Turkish IBAN: TR + 24 digits (2 check + 5 bank + 1 reserve + 16 account).
_IBAN_RUN = re.compile(r"\bTR[ .\-]?(?:\d[ .\-]?){23}\d\b", re.IGNORECASE)


def _digits_of(run: str) -> str:
    return re.sub(r"\D", "", run)


def is_valid_tckn(digits: str) -> bool:
    """Validate a Turkish national identity number (TCKN) checksum."""
    if len(digits) != 11 or not digits.isdigit() or digits[0] == "0":
        return False
    d = [int(c) for c in digits]
    odd = d[0] + d[2] + d[4] + d[6] + d[8]
    even = d[1] + d[3] + d[5] + d[7]
    if (odd * 7 - even) % 10 != d[9]:
        return False
    return sum(d[:10]) % 10 == d[10]


def is_valid_luhn(digits: str) -> bool:
    """Validate a payment card PAN with the Luhn algorithm."""
    if not 13 <= len(digits) <= 19 or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def is_valid_tr_iban(text: str) -> bool:
    """Validate a Turkish IBAN with the ISO 13616 mod-97 check."""
    compact = re.sub(r"[ .\-]", "", text).upper()
    if not re.fullmatch(r"TR\d{24}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def _is_tr_mobile(digits: str) -> bool:
    """Turkish mobile numbers: 05xxxxxxxxx / 5xxxxxxxxx / +905xxxxxxxxx."""
    if len(digits) == 12 and digits.startswith("905"):
        return True
    if len(digits) == 11 and digits.startswith("05"):
        return True
    return len(digits) == 10 and digits.startswith("5")


@dataclass
class RedactionResult:
    text: str
    # Counts per PII category actually masked, e.g. {"tckn": 1, "pan": 2}.
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.counts)


def redact_pii(text: str) -> RedactionResult:
    """Mask validated PII identifiers in a transcript string.

    Masking preserves trailing digits (last 4 for PAN/IBAN/phone, last 2 for
    TCKN) so QA reviewers can still cross-reference against CRM records.
    """
    if not text:
        return RedactionResult(text=text)

    counts: dict[str, int] = {}

    def _bump(kind: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1

    # IBAN first: it contains a 24-digit run that would otherwise be shredded
    # by the generic digit-run pass below.
    def _iban_sub(m: re.Match) -> str:
        if is_valid_tr_iban(m.group(0)):
            _bump("iban")
            return f"TR** **** [IBAN GİZLENDİ] {_digits_of(m.group(0))[-4:]}"
        return m.group(0)

    text = _IBAN_RUN.sub(_iban_sub, text)

    def _digit_sub(m: re.Match) -> str:
        run = m.group(0)
        digits = _digits_of(run)
        if is_valid_luhn(digits) and len(digits) >= 15:
            _bump("pan")
            return f"[KART NO GİZLENDİ] **** {digits[-4:]}"
        if is_valid_tckn(digits):
            _bump("tckn")
            return f"[TCKN GİZLENDİ] *********{digits[-2:]}"
        if _is_tr_mobile(digits):
            _bump("phone")
            return f"[TELEFON GİZLENDİ] *** {digits[-4:]}"
        return run

    text = _DIGIT_RUN.sub(_digit_sub, text)

    if counts:
        logger.info(f"PII redaction masked identifiers: {counts}")
    return RedactionResult(text=text, counts=counts)


def redact_segments(segments: list[Any]) -> dict[str, int]:
    """Redact PII in-place across transcript segments (SegmentInput/dataclass/dict).

    Returns aggregate counts per PII category.
    """
    totals: dict[str, int] = {}
    for idx, seg in enumerate(segments):
        text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        # raw_text (pre-domain-correction audit copy, see asr_service.py) can
        # independently contain a PII match the corrected text no longer
        # does (or vice versa - correction could turn a misheard digit
        # string into a valid-looking one) - both must be scrubbed, or the
        # audit-trail feature becomes its own KVKK/PCI-DSS leak.
        raw_text = (
            seg.get("raw_text", "") if isinstance(seg, dict) else getattr(seg, "raw_text", "")
        )
        new_text, new_raw_text = text, raw_text
        counts: dict[str, int] = {}

        if text:
            result = redact_pii(text)
            if result.redacted:
                new_text = result.text
                for k, v in result.counts.items():
                    counts[k] = counts.get(k, 0) + v
        if raw_text:
            raw_result = redact_pii(raw_text)
            if raw_result.redacted:
                new_raw_text = raw_result.text
                for k, v in raw_result.counts.items():
                    counts[k] = counts.get(k, 0) + v

        if not counts:
            continue
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v

        if isinstance(seg, dict):
            seg["text"] = new_text
            if "raw_text" in seg:
                seg["raw_text"] = new_raw_text
        else:
            import dataclasses

            if (
                dataclasses.is_dataclass(seg)
                and getattr(seg, "__dataclass_params__", None)
                and seg.__dataclass_params__.frozen
            ):
                replace_kwargs = {"text": new_text}
                if hasattr(seg, "raw_text"):
                    replace_kwargs["raw_text"] = new_raw_text
                segments[idx] = dataclasses.replace(seg, **replace_kwargs)
            else:
                try:
                    seg.text = new_text
                    if hasattr(seg, "raw_text"):
                        seg.raw_text = new_raw_text
                except Exception:
                    if dataclasses.is_dataclass(seg):
                        replace_kwargs = {"text": new_text}
                        if hasattr(seg, "raw_text"):
                            replace_kwargs["raw_text"] = new_raw_text
                        segments[idx] = dataclasses.replace(seg, **replace_kwargs)
    return totals
