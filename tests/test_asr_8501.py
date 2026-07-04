import sys
from pathlib import Path

import pytest

# Add the ASR directory to sys.path so its internal imports (like 'import config') work
ASR_DIR = Path(__file__).resolve().parent.parent / "tools" / "legacy_streamlit" / "ASR"
sys.path.insert(0, str(ASR_DIR))

try:
    from logic_handlers import (
        calculate_word_accuracy,
        clamp,
        format_timestamp,
        levenshtein_distance,
        normalize_for_wer,
        resolve_mlx_repo_name,
        sanitize_hallucinatory_repetitions,
    )
except ImportError as e:
    pytest.skip(
        f"Could not import ASR logic handlers. Is the ASR directory set up correctly? {e}",
        allow_module_level=True,
    )


def test_format_timestamp():
    assert format_timestamp(0) == "0:00:00"
    assert format_timestamp(65.5) == "0:01:05"
    assert format_timestamp(3600) == "1:00:00"


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    assert clamp(0.5, 0.0, 1.0) == 0.5
    assert clamp(1.5, 0.0, 1.0) == 1.0


def test_normalize_for_wer():
    assert normalize_for_wer("Merhaba Dünya!") == ["merhaba", "dünya"]
    assert normalize_for_wer("  BOŞLUKLAR   Test  ") == ["boşluklar", "test"]
    assert normalize_for_wer("O'nun arabası 123") == ["o", "nun", "arabası", "123"]
    assert normalize_for_wer("") == []


def test_levenshtein_distance():
    ref = ["bu", "bir", "test"]
    hyp1 = ["bu", "bir", "test"]
    hyp2 = ["bu", "bir", "hata"]
    hyp3 = ["bu", "test"]
    hyp4 = ["bu", "bir", "uzun", "test"]

    assert levenshtein_distance(ref, hyp1) == 0
    assert levenshtein_distance(ref, hyp2) == 1
    assert levenshtein_distance(ref, hyp3) == 1
    assert levenshtein_distance(ref, hyp4) == 1
    assert levenshtein_distance([], hyp1) == 3
    assert levenshtein_distance(ref, []) == 3


def test_calculate_word_accuracy():
    # Exact match
    res = calculate_word_accuracy("bu bir test", "bu bir test")
    assert res["wer"] == 0.0
    assert res["accuracy"] == 100.0
    assert res["edit_distance"] == 0

    # 1 error out of 3 words -> wer = 1/3, acc = 66.67%
    res = calculate_word_accuracy("bu bir test", "bu bir hata")
    assert res["wer"] == 1 / 3
    assert round(res["accuracy"], 2) == 66.67
    assert res["edit_distance"] == 1

    # Completely wrong
    res = calculate_word_accuracy("bu bir test", "tamamen yanlış kelimeler")
    assert res["wer"] == 1.0
    assert res["accuracy"] == 0.0
    assert res["edit_distance"] == 3

    # Empty cases
    res = calculate_word_accuracy("", "test")
    assert res["wer"] == 1.0
    assert res["accuracy"] == 0.0

    res = calculate_word_accuracy("", "")
    assert res["wer"] == 0.0
    assert res["accuracy"] == 100.0


def test_resolve_mlx_repo_name():
    assert resolve_mlx_repo_name("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert resolve_mlx_repo_name("mlx-community/custom-model") == "mlx-community/custom-model"


def test_sanitize_hallucinatory_repetitions():
    # Test consecutive identical sentences
    assert sanitize_hallucinatory_repetitions("Efendim? Efendim? İyi günler. İyi günler.") == "Efendim? İyi günler."
    # Test word stutter loops
    assert sanitize_hallucinatory_repetitions("alo alo alo") == "alo"
    assert sanitize_hallucinatory_repetitions("bursa hanım bursa hanım bursa hanım") == "bursa hanım"
    # Normal text should remain untouched
    assert sanitize_hallucinatory_repetitions("Merhaba, nasılsınız? Ben iyiyim.") == "Merhaba, nasılsınız? Ben iyiyim."
