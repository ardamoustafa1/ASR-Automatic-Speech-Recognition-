"""Toxicity engine: real profanity must be caught, ordinary business Turkish
must never false-positive. The legacy Streamlit tool's word list included
"mal" (merchandise), "hasta" (patient), "parasız" (free), "top" (top-up),
"deli" (used as a harmless intensifier) as "profanity" - this engine is
deliberately narrower and must not repeat that mistake."""

from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.toxicity_engine import analyze_toxicity


def seg(text, speaker="SPEAKER_01", start=0.0, end=2.0):
    return SegmentInput(start=start, end=end, text=text, speaker=speaker, segment_index=0)


def test_ordinary_business_turkish_is_never_flagged():
    segments = [
        seg("Merhaba, hasta olduğum için aradım, randevu almak istiyorum."),
        seg("Bu ay mal teslimatında gecikme oldu, parasız kargo hakkım var mı?"),
        seg("Deli gibi çalıştım ama hala çözemedim, top up yapabilir miyim?"),
        seg(
            "Hayvan barınağı için bağış yapmak istiyorum, İnek Çiftliği kampanyası hakkında bilgi alabilir miyim?"
        ),
        seg("Faturamda bir hata var, fakirlik sınırı altında olduğum için indirim talep ediyorum."),
    ]
    result = analyze_toxicity(segments)
    assert result.is_clean
    assert result.matched_terms == ()
    assert result.toxicity_rate == 0.0


def test_explicit_profanity_is_caught():
    segments = [seg("Siktir git buradan, seni orospu çocuğu!")]
    result = analyze_toxicity(segments)
    assert not result.is_clean
    assert "siktir" in result.matched_terms
    assert "orospu" in " ".join(result.matched_terms)
    assert len(result.flagged_segments) == 1
    assert result.flagged_segments[0]["speaker"] == "SPEAKER_01"


def test_aggressive_insult_terms_caught():
    segments = [seg("Ne şerefsiz bir adamsın, namussuzun tekisin.")]
    result = analyze_toxicity(segments)
    assert not result.is_clean
    assert "şerefsiz" in result.matched_terms


def test_empty_segments_is_clean():
    result = analyze_toxicity([])
    assert result.is_clean
    assert result.toxicity_rate == 0.0


def test_toxicity_rate_is_word_weighted():
    segments = [seg("siktir")]  # 1 word, 1 match -> rate 1.0
    result = analyze_toxicity(segments)
    assert result.toxicity_rate == 1.0
