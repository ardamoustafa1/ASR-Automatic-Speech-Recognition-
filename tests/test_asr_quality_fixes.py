"""Regression tests for the ASR accuracy fixes: dedup safety, VAD gating,
confidence aggregation, sector prompts, and phonetic corrections."""

import numpy as np

from asr_pro.services.asr_service import ASRService, TranscriptionSegment
from asr_pro.services.domain_adaptation import DomainAdaptationService


def seg(start, end, text, speaker=None, logprob=-0.3):
    return TranscriptionSegment(
        start=start, end=end, text=text, speaker=speaker, avg_logprob=logprob
    )


# ─── Deduplication: real speech must survive ─────────────────────────────────


def test_dedup_keeps_repeated_backchannels_across_call():
    """Customers legitimately say 'Evet.' many times per call - a 25s dedup
    window was observed deleting real confirmations from production audio."""
    segments = [
        seg(10.0, 10.5, "Evet."),
        seg(18.0, 18.5, "Evet."),  # 7.5s later: real confirmation
        seg(30.0, 30.5, "Evet."),
    ]
    kept = ASRService._deduplicate_segments(segments)
    assert len(kept) == 3


def test_dedup_drops_immediate_hallucination_loop():
    segments = [
        seg(10.0, 12.0, "Kayıt altına alınmaktadır bu görüşme"),
        seg(12.1, 14.1, "Kayıt altına alınmaktadır bu görüşme"),
        seg(14.2, 16.2, "Kayıt altına alınmaktadır bu görüşme"),
    ]
    kept = ASRService._deduplicate_segments(segments)
    assert len(kept) == 1


def test_dedup_drops_immediate_backchannel_loop_but_keeps_later():
    segments = [
        seg(10.0, 10.4, "Tamam"),
        seg(10.5, 10.9, "Tamam"),  # immediate stutter loop -> drop
        seg(25.0, 25.4, "Tamam"),  # much later, real -> keep
    ]
    kept = ASRService._deduplicate_segments(segments)
    assert [s.start for s in kept] == [10.0, 25.0]


def test_dedup_keeps_short_answer_contained_in_previous():
    """'Evet' following 'Evet, YouTube artı.' is a real answer, not bleed."""
    segments = [
        seg(10.0, 12.0, "Evet, YouTube artı çok güzel bir paket."),
        seg(14.0, 14.4, "Evet."),
    ]
    kept = ASRService._deduplicate_segments(segments)
    assert len(kept) == 2


# ─── VAD speech regions ──────────────────────────────────────────────────────


def test_vad_regions_silence_returns_empty(monkeypatch):
    class FakeVAD:
        loaded = True

        def filter_speech_timestamps(self, pcm, sampling_rate=16000):
            return []

    from asr_pro.services import vad_service

    monkeypatch.setattr(vad_service.VADService, "get_instance", classmethod(lambda cls: FakeVAD()))
    silent = np.zeros(16000, dtype=np.float32)
    assert ASRService._vad_speech_regions(silent) == []


def test_vad_regions_empty_but_loud_falls_back_to_full_decode(monkeypatch):
    """VAD returning [] on clearly non-silent audio means the VAD errored -
    must NOT be interpreted as 'no speech' (would delete the whole call)."""

    class FakeVAD:
        loaded = True

        def filter_speech_timestamps(self, pcm, sampling_rate=16000):
            return []

    from asr_pro.services import vad_service

    monkeypatch.setattr(vad_service.VADService, "get_instance", classmethod(lambda cls: FakeVAD()))
    loud = (np.random.default_rng(7).standard_normal(16000) * 0.2).astype(np.float32)
    assert ASRService._vad_speech_regions(loud) is None


def test_vad_regions_merge_and_pad(monkeypatch):
    class FakeVAD:
        loaded = True

        def filter_speech_timestamps(self, pcm, sampling_rate=16000):
            return [
                {"start": 16000, "end": 32000},
                {"start": 40000, "end": 48000},  # 0.5s gap -> merged
                {"start": 160000, "end": 176000},  # far -> separate region
            ]

    from asr_pro.services import vad_service

    monkeypatch.setattr(vad_service.VADService, "get_instance", classmethod(lambda cls: FakeVAD()))
    pcm = np.ones(200000, dtype=np.float32) * 0.1
    regions = ASRService._vad_speech_regions(pcm)
    assert len(regions) == 2
    # padded 0.3s = 4800 samples
    assert regions[0] == (16000 - 4800, 48000 + 4800)
    assert regions[1][0] == 160000 - 4800


def test_vad_regions_unavailable_returns_none(monkeypatch):
    class FakeVAD:
        loaded = False

    from asr_pro.services import vad_service

    monkeypatch.setattr(vad_service.VADService, "get_instance", classmethod(lambda cls: FakeVAD()))
    assert ASRService._vad_speech_regions(np.ones(16000, dtype=np.float32)) is None


# ─── Confidence aggregation ──────────────────────────────────────────────────


def test_compute_confidence_duration_weighted():
    segments = [
        seg(0.0, 10.0, "uzun ve emin segment", logprob=-0.1),
        seg(10.0, 10.5, "kısa şüpheli", logprob=-2.0),
    ]
    conf = ASRService.compute_confidence(segments)
    assert 0.7 < conf < 0.95  # dominated by the long confident segment


def test_compute_confidence_no_data():
    segments = [seg(0.0, 1.0, "veri yok", logprob=-1.0)]
    assert ASRService.compute_confidence(segments) == 0.0
    assert ASRService.compute_confidence([]) == 0.0


# ─── Sector-aware prompts and corrections ────────────────────────────────────


def test_banking_prompt_selected():
    p = DomainAdaptationService.get_initial_prompt("banking")
    assert "IBAN" in p and "mevduat" in p
    assert "VoLTE" not in p


def test_telecom_prompt_default():
    p = DomainAdaptationService.get_initial_prompt("telecom")
    assert "VoLTE" in p


def test_omni_prompt_has_both():
    p = DomainAdaptationService.get_initial_prompt("omni")
    assert "VoLTE" in p and "IBAN" in p


def test_phonetic_corrections_observed_errors():
    # Errors observed verbatim on production 8kHz call-center audio:
    assert DomainAdaptationService.correct_terms("60 GB sınıfsız sosyal medya") == (
        "60 GB sınırsız sosyal medya"
    )
    assert DomainAdaptationService.correct_terms("zaten modafon yanımdan") == (
        "zaten Vodafone yanımdan"
    )
    assert "havale" in DomainAdaptationService.correct_terms("havele yaptım")


def test_correct_telecom_terms_alias_still_works():
    assert DomainAdaptationService.correct_telecom_terms("romin ücreti") == "roaming ücreti"


# ─── Punctuation-glued repeat loops (no whitespace between repeats) ─────────


def test_sanitize_collapses_glued_digit_loop():
    """Observed live on real noisy telephony audio with large-v3-turbo:
    a single segment decoded as '5.' followed by '7.' repeated 100 times,
    with zero whitespace between repeats - the whitespace-requiring n-gram
    collapse loop never matches this glued form."""
    garbage = "5." + "7." * 100
    assert ASRService._sanitize_text(garbage) == "5.7."


def test_sanitize_glued_loop_does_not_touch_normal_text():
    assert ASRService._sanitize_text("840 TL bir yıl boyunca.") == "840 TL bir yıl boyunca."
    assert ASRService._sanitize_text("normal cümle burada.") == "normal cümle burada."


def test_sanitize_collapses_glued_word_loop():
    garbage = "Merhaba " + "ve.ve.ve.ve.ve.ve." + " efendim"
    out = ASRService._sanitize_text(garbage)
    assert "ve.ve.ve" not in out
    assert out.count("ve.") <= 1


# ─── Word-probability-triggered rescue ───────────────────────────────────────


def _seg_with_words(start, end, text, logprob, words):
    return TranscriptionSegment(start=start, end=end, text=text, avg_logprob=logprob, words=words)


def test_word_suspect_trigger_fires_on_confident_segment_with_garbled_words(monkeypatch):
    """'Katapay' (p=0.28) inside a healthy-logprob segment must trigger a
    rescue attempt - segment-level confidence alone never catches these."""
    svc = _svc_for_rescue()
    calls = []

    def spy(clip, lang, prompt):
        calls.append(1)
        return ("", -10.0, None)  # reject path; we only assert the attempt

    monkeypatch.setattr(svc, "_decode_clip", spy)
    words = [
        {"word": "Katapay", "start": 0.1, "end": 0.5, "probability": 0.28},
        {"word": "taktiğinde", "start": 0.6, "end": 1.0, "probability": 0.35},
        {"word": "paketi", "start": 1.1, "end": 1.5, "probability": 0.95},
    ]
    segments = [_seg_with_words(0.0, 2.0, "Katapay taktiğinde paketi", -0.25, words)]
    svc._second_pass_rescue(_audio(), segments, "tr", "prompt")
    assert calls, "suspect-word segment must be attempted"


def test_word_suspect_rescue_accepts_only_fewer_suspects(monkeypatch):
    svc = _svc_for_rescue()
    fixed_words = [
        {"word": "Paket", "start": 0.1, "end": 0.5, "probability": 0.92},
        {"word": "dediğinizde", "start": 0.6, "end": 1.0, "probability": 0.88},
        {"word": "paketi", "start": 1.1, "end": 1.5, "probability": 0.95},
    ]
    monkeypatch.setattr(
        svc,
        "_decode_clip",
        lambda clip, lang, prompt: ("Paket dediğinizde paketi", -0.20, fixed_words),
    )
    words = [
        {"word": "Katapay", "start": 0.1, "end": 0.5, "probability": 0.28},
        {"word": "taktiğinde", "start": 0.6, "end": 1.0, "probability": 0.35},
    ]
    segments = [_seg_with_words(0.0, 2.0, "Katapay taktiğinde", -0.25, words)]
    out = svc._second_pass_rescue(_audio(), segments, "tr", "prompt")
    assert out[0].text == "Paket dediğinizde paketi"


def test_word_suspect_rescue_rejects_lower_logprob(monkeypatch):
    svc = _svc_for_rescue()
    monkeypatch.setattr(svc, "_decode_clip", lambda clip, lang, prompt: ("farklı metin", -0.50, []))
    words = [
        {"word": "Katapay", "start": 0.1, "end": 0.5, "probability": 0.28},
        {"word": "taktiğinde", "start": 0.6, "end": 1.0, "probability": 0.35},
    ]
    segments = [_seg_with_words(0.0, 2.0, "Katapay taktiğinde", -0.25, words)]
    out = svc._second_pass_rescue(_audio(), segments, "tr", "prompt")
    assert out[0].text == "Katapay taktiğinde"  # decoder liked it less -> keep


def test_short_low_prob_words_do_not_trigger(monkeypatch):
    """Backchannels ('Bu', 'Tamam') score low at boundaries without being
    wrong - only 4+ char content words count toward the trigger."""
    svc = _svc_for_rescue()
    calls = []
    monkeypatch.setattr(svc, "_decode_clip", lambda *a: (calls.append(1), ("", -10.0, None))[1])
    words = [
        {"word": "Bu", "start": 0.1, "end": 0.3, "probability": 0.07},
        {"word": "da", "start": 0.4, "end": 0.5, "probability": 0.20},
        {"word": "paket", "start": 0.6, "end": 1.0, "probability": 0.95},
    ]
    segments = [_seg_with_words(0.0, 2.0, "Bu da paket", -0.25, words)]
    svc._second_pass_rescue(_audio(), segments, "tr", "prompt")
    assert not calls


def test_rescue_attempt_budget_bounds_latency(monkeypatch):
    svc = _svc_for_rescue()
    calls = []
    monkeypatch.setattr(svc, "_decode_clip", lambda *a: (calls.append(1), ("", -10.0, None))[1])
    segments = [seg(i * 2.0, i * 2.0 + 1.5, f"düşük güven {i}", logprob=-0.9) for i in range(20)]
    svc._second_pass_rescue(_audio(seconds=60.0), segments, "tr", "prompt")
    from asr_pro.config import settings

    assert len(calls) == settings.asr_second_pass_max_attempts


# ─── Initial-prompt echo hallucination ───────────────────────────────────────


def _real_prompt_tokens():
    from asr_pro.config import settings
    from asr_pro.services.domain_adaptation import DomainAdaptationService

    prompt = (
        f"{settings.asr_initial_prompt} {DomainAdaptationService.get_initial_prompt('telecom')}"
    )
    return ASRService._prompt_token_set(prompt)


def test_prompt_echo_catches_real_observed_leak():
    """Verbatim leak observed on production call D138202 at 135.8s: Whisper
    emitted the domain-vocabulary prompt as if the agent had said it -
    including competitor names 'Türk Telekom, Turkcell'. Unshippable for a
    telecom client; must always be dropped."""
    leak = "Vodafone Türkiye, Türk Telekom, Turkcell, VoLTE, eSIM, tayfan, borç sorgulama."
    assert ASRService._is_prompt_echo(leak, _real_prompt_tokens())


def test_prompt_echo_catches_mutated_leak():
    """Echoes mutate: a second real leak on the same call rendered 'fiber
    altyapı' as repeated 'fiyat altyapı', diluting exact-token overlap to
    0.64 - which slipped past an earlier 0.7 ratio threshold. Locks in the
    0.6 threshold."""
    mutated = (
        "Vodafone Türkiye, Türk Telekom, Turkcell, VoLTE, eSIM, tayinli, "
        "fiyat altyapı, fiyat altyapı, fiyat altyap"
    )
    assert ASRService._is_prompt_echo(mutated, _real_prompt_tokens())


def test_prompt_echo_catches_third_mutation_via_ordered_run():
    """Third observed mutation of the same hold-stretch echo - bag-of-words
    overlap dilutes to 0.58 here, but the intact ordered prefix (vodafone,
    türkiye, türk, telekom, turkcell, volte, esim) trips the ordered-run
    detector. Mutations change the tail, never the prefix."""
    mutated3 = (
        "Vodafone Türkiye, Türk Telekom, Turkcell, VoLTE, eSIM, "
        "tayinli bir hizmet, tarif ve bilgi ve bilgi"
    )
    assert ASRService._is_prompt_echo(mutated3, _real_prompt_tokens())


def test_prompt_echo_brand_comparison_not_flagged():
    """A customer legitimately comparing operators matches only ~4 prompt
    tokens - the min-5-matched guard must keep this as real speech."""
    assert not ASRService._is_prompt_echo(
        "Vodafone, Türk Telekom, Turkcell karşılaştırması yaptım dün akşam.",
        _real_prompt_tokens(),
    )


def test_prompt_echo_never_flags_real_speech():
    tokens = _real_prompt_tokens()
    real_lines = [
        # Uses several domain terms, but with real connective grammar:
        "Taahhüt süreniz bitiyor, yeni tarifemizde sınırsız internet mevcut efendim.",
        "Sizleri Vodafone'dan arıyorum, müsait misiniz acaba?",
        "Fatura borcunuzu sorgulamak için kimlik doğrulaması yapmam gerekiyor.",
        "840 TL bir yıl boyunca sabit kalıyor Buse Hanım.",
        "Turkcell'den geçiş yapmak istiyorum, kampanyanız var mı?",
    ]
    for line in real_lines:
        assert not ASRService._is_prompt_echo(line, tokens), line


def test_prompt_echo_short_segments_never_flagged():
    tokens = _real_prompt_tokens()
    assert not ASRService._is_prompt_echo("VoLTE eSIM tarife", tokens)
    assert not ASRService._is_prompt_echo("Vodafone Türk Telekom", tokens)


def test_prompt_echo_empty_prompt_is_safe():
    assert not ASRService._is_prompt_echo("herhangi bir metin burada uzun uzun", frozenset())


# ─── Known Whisper Turkish hallucination artifacts ───────────────────────────


def test_known_hallucination_patterns():
    assert ASRService._is_known_hallucination("İzlediğiniz için teşekkür ederim.")
    assert ASRService._is_known_hallucination("Altyazı M.K.")
    assert ASRService._is_known_hallucination("Kanalımıza abone olmayı unutmayın")
    # Real call-center speech must never match:
    assert not ASRService._is_known_hallucination("Teşekkür ederim, iyi günler.")
    assert not ASRService._is_known_hallucination("Aradığınız için teşekkürler efendim.")
    assert not ASRService._is_known_hallucination("840 TL bir yıl boyunca.")


# ─── Second-pass rescue decoding ─────────────────────────────────────────────


def _svc_for_rescue():
    svc = ASRService.__new__(ASRService)  # bypass singleton/device probing
    svc._is_mlx = True
    return svc


def _audio(seconds=10.0):
    return (np.ones(int(seconds * 16000)) * 0.05).astype(np.float32)


def test_second_pass_accepts_better_hypothesis(monkeypatch):
    svc = _svc_for_rescue()
    monkeypatch.setattr(
        svc, "_decode_clip", lambda clip, lang, prompt: ("en özel teklifimizi", -0.15, None)
    )
    segments = [
        seg(0.0, 2.0, "Merhaba, kampanyamız var.", logprob=-0.2),
        seg(2.0, 4.0, "en özürlü teklifimizi", logprob=-0.9),
    ]
    out = svc._second_pass_rescue(_audio(), segments, "tr", "prompt")
    assert out[1].text == "en özel teklifimizi"
    assert out[1].avg_logprob == -0.15
    assert out[0].text == "Merhaba, kampanyamız var."  # confident segment untouched


def test_second_pass_rejects_worse_hypothesis(monkeypatch):
    svc = _svc_for_rescue()
    monkeypatch.setattr(
        svc, "_decode_clip", lambda clip, lang, prompt: ("başka bir şey", -0.88, None)
    )
    segments = [seg(0.0, 2.0, "orijinal metin", logprob=-0.9)]
    out = svc._second_pass_rescue(_audio(), segments, "tr", "prompt")
    assert out[0].text == "orijinal metin"  # margin not met -> keep original


def test_second_pass_rejects_hallucination(monkeypatch):
    svc = _svc_for_rescue()
    monkeypatch.setattr(
        svc,
        "_decode_clip",
        lambda clip, lang, prompt: ("İzlediğiniz için teşekkür ederim.", -0.05, None),
    )
    segments = [seg(0.0, 2.0, "orijinal metin", logprob=-0.9)]
    out = svc._second_pass_rescue(_audio(), segments, "tr", "prompt")
    assert out[0].text == "orijinal metin"


def test_second_pass_skips_confident_and_unknown(monkeypatch):
    svc = _svc_for_rescue()
    calls = []

    def spy(clip, lang, prompt):
        calls.append(prompt)
        return ("x", 0.0, None)

    monkeypatch.setattr(svc, "_decode_clip", spy)
    segments = [
        seg(0.0, 2.0, "güvenli segment", logprob=-0.2),  # above threshold
        seg(2.0, 4.0, "bilinmeyen", logprob=-1.0),  # sentinel: no data
    ]
    svc._second_pass_rescue(_audio(), segments, "tr", "prompt")
    assert calls == []


def test_second_pass_context_prompt_uses_preceding_confident_text(monkeypatch):
    svc = _svc_for_rescue()
    prompts = []

    def spy(clip, lang, prompt):
        prompts.append(prompt)
        return ("", -10.0, None)

    monkeypatch.setattr(svc, "_decode_clip", spy)
    segments = [
        seg(0.0, 2.0, "Tarifeniz yenileniyor efendim.", logprob=-0.2),
        seg(2.0, 4.0, "şüpheli kısım", logprob=-0.8),
    ]
    svc._second_pass_rescue(_audio(), segments, "tr", "DOMAIN")
    assert len(prompts) == 1
    assert "Tarifeniz yenileniyor efendim." in prompts[0]


# ─── words / avg_logprob survive sentence splitting ──────────────────────────


def test_split_preserves_confidence_fields():
    words = [
        {"word": "Merhaba", "start": 0.2, "end": 0.6, "probability": 0.9},
        {"word": "efendim.", "start": 0.7, "end": 1.1, "probability": 0.8},
        {"word": "Nasıl", "start": 3.0, "end": 3.4, "probability": 0.95},
        {"word": "yardımcı", "start": 3.5, "end": 4.0, "probability": 0.9},
        {"word": "olabilirim?", "start": 4.1, "end": 4.8, "probability": 0.85},
    ]
    segment = TranscriptionSegment(
        start=0.0,
        end=5.0,
        text="Merhaba efendim. Nasıl yardımcı olabilirim?",
        avg_logprob=-0.25,
        words=words,
    )
    out = ASRService._split_into_sentences([segment])
    assert len(out) == 2
    assert all(s.avg_logprob == -0.25 for s in out)
    assert out[0].words and out[0].words[0]["word"] == "Merhaba"
