"""KVKK / PCI-DSS PII redaction: checksum validation and masking behavior."""

from dataclasses import dataclass

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.services.pii_redaction import (
    is_valid_luhn,
    is_valid_tckn,
    is_valid_tr_iban,
    redact_pii,
    redact_segments,
)

# Synthetic but checksum-valid identifiers (never real customer data).
VALID_TCKN = "10000000146"
VALID_PAN = "4111111111111111"  # canonical Visa test PAN
VALID_IBAN = "TR330006100519786457841326"  # TR IBAN example (mod-97 valid)


def test_tckn_checksum():
    assert is_valid_tckn(VALID_TCKN)
    assert not is_valid_tckn("12345678901")
    assert not is_valid_tckn("01000000146")  # leading zero invalid
    assert not is_valid_tckn("1234567890")  # too short


def test_luhn():
    assert is_valid_luhn(VALID_PAN)
    assert not is_valid_luhn("4111111111111112")
    assert not is_valid_luhn("123456")


def test_iban_mod97():
    assert is_valid_tr_iban(VALID_IBAN)
    assert is_valid_tr_iban("TR33 0006 1005 1978 6457 8413 26")
    assert not is_valid_tr_iban("TR330006100519786457841327")


def test_redact_tckn_spoken_with_spaces():
    spoken = " ".join(VALID_TCKN)
    result = redact_pii(f"TC kimlik numaram {spoken} efendim")
    assert result.counts == {"tckn": 1}
    assert VALID_TCKN[:9] not in result.text.replace(" ", "")
    assert "[TCKN GİZLENDİ]" in result.text
    assert result.text.endswith("efendim")


def test_redact_pan_keeps_last4():
    result = redact_pii(f"Kart numaram {VALID_PAN} olacak")
    assert result.counts == {"pan": 1}
    assert "1111111" not in result.text
    assert "**** 1111" in result.text


def test_redact_iban():
    result = redact_pii(f"IBAN {VALID_IBAN} hesabıma iade edin")
    assert result.counts == {"iban": 1}
    assert "1978" not in result.text
    assert result.text.count("1326") == 1


def test_ordinary_numbers_untouched():
    text = "840 TL'den 60 GB paket, son ödeme 2026-07-15, sipariş 123456789"
    result = redact_pii(text)
    assert result.text == text
    assert not result.redacted


def test_phone_masking():
    result = redact_pii("Beni 0532 123 45 67 numaradan arayın")
    assert result.counts == {"phone": 1}
    assert "123 45" not in result.text


def test_redact_segments_frozen_dataclass():
    segments = [
        SegmentInput(start=0, end=1, text=f"Kartım {VALID_PAN}", speaker="SPEAKER_01"),
        SegmentInput(start=1, end=2, text="Teşekkürler", speaker="SPEAKER_00"),
    ]
    counts = redact_segments(segments)
    assert counts == {"pan": 1}
    assert "****" in segments[0].text
    assert segments[0].speaker == "SPEAKER_01"  # other fields preserved
    assert segments[1].text == "Teşekkürler"


def test_redact_segments_mutable_and_dict():
    @dataclass
    class Seg:
        text: str

    segs = [Seg(text=f"TCKN {VALID_TCKN}"), {"text": f"IBAN {VALID_IBAN}"}]
    counts = redact_segments(segs)
    assert counts == {"tckn": 1, "iban": 1}
    assert "[TCKN GİZLENDİ]" in segs[0].text
    assert "[IBAN GİZLENDİ]" in segs[1]["text"]


def test_redact_segments_also_scrubs_raw_text():
    """raw_text (the pre-domain-correction audit copy) is a second place PII
    can independently appear - it must never bypass redaction just because
    the corrected `text` field was already clean, or the audit-trail
    feature becomes its own KVKK/PCI-DSS leak."""
    segments = [
        SegmentInput(
            start=0,
            end=1,
            text="kartım güvenli",  # already clean/corrected
            raw_text=f"kartım {VALID_PAN}",  # PAN only present pre-correction
            speaker="SPEAKER_01",
        )
    ]
    counts = redact_segments(segments)
    assert counts == {"pan": 1}
    assert VALID_PAN not in segments[0].raw_text
    assert "****" in segments[0].raw_text
    assert segments[0].text == "kartım güvenli"  # untouched, no PII there


def test_redact_segments_raw_text_dict_form():
    segs = [{"text": "temiz", "raw_text": f"TCKN {VALID_TCKN}"}]
    counts = redact_segments(segs)
    assert counts == {"tckn": 1}
    assert VALID_TCKN not in segs[0]["raw_text"]
