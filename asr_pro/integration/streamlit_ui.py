import re

import streamlit as st

from asr_pro.core.keyword_engine import RuleInput, SegmentInput, analyze_keywords, hits_to_dict

DEFAULT_ENTERPRISE_RULES = [
    RuleInput(
        id="R-APP-01",
        name="Mobil Uygulama Sorunları",
        keywords=(
            "uygulama çöktü",
            "uygulama donuyor",
            "uygulamaya giremiyorum",
            "açılmıyor",
            "hata veriyor",
            "sistem hatası",
        ),
        match_mode="fuzzy",
        severity="high",
        topic_id="T-TECH",
    ),
    RuleInput(
        id="R-BIL-01",
        name="Fatura ve Ücretlendirme İtirazı",
        keywords=(
            "fatura",
            "fazla ücret",
            "haberim yok",
            "kesinti",
            "neden kestiniz",
            "para çekilmiş",
            "itiraz",
            "iptal",
        ),
        match_mode="semantic",
        severity="high",
        topic_id="T-BILL",
    ),
    RuleInput(
        id="R-SHIP-01",
        name="Kargo ve Teslimat Sorunları",
        keywords=(
            "kargom nerede",
            "kargo gelmedi",
            "teslim edilmedi",
            "kurye",
            "gecikme",
            "teslimat",
        ),
        match_mode="fuzzy",
        severity="medium",
        topic_id="T-SHIP",
    ),
    RuleInput(
        id="R-RET-01",
        name="İade ve İptal Talebi",
        keywords=(
            "iade etmek",
            "geri vermek",
            "paramı geri verin",
            "iptal",
            "kapatmak istiyorum",
            "üyeliğimi sonlandır",
        ),
        match_mode="semantic",
        severity="high",
        topic_id="T-RET",
    ),
    RuleInput(
        id="R-CS-01",
        name="Müşteri Memnuniyetsizliği",
        keywords=(
            "rezalet",
            "şikayetçi",
            "tüketici hakları",
            "mahkemeye vereceğim",
            "cimer",
            "berbat",
            "terbiyesizlik",
            "yazıklar olsun",
        ),
        match_mode="fuzzy",
        severity="critical",
        topic_id="T-CHURN",
    ),
]


def execute_enterprise_keyword_analysis(
    segments_data,
    full_transcription,
    sector=None,
    audio_path=None,
    uploaded_name=None,
    asr_confidence=0.0,
    quality_gate_passed=True,
):
    """
    Apple-level top tier integration with the real keyword engine.
    Translates faster-whisper segments to SegmentInput and runs analyze_keywords.
    """
    if not segments_data:
        return {"topics": [], "hits": [], "raw_text": full_transcription}

    segment_inputs = [
        SegmentInput(start=seg.start, end=seg.end, text=seg.text, segment_index=idx)
        for idx, seg in enumerate(segments_data)
    ]

    hits = analyze_keywords(segment_inputs, DEFAULT_ENTERPRISE_RULES, sector=sector or "omni")

    # Extract unique rule names for backward compatibility with 'topics'
    unique_topics = list({hit.rule_name for hit in hits})

    return {"topics": unique_topics, "hits": hits_to_dict(hits), "raw_text": full_transcription}


def highlight_transcript_html(text: str, hits: list[dict]) -> str:
    if not text or not hits:
        return text

    highlighted = text
    for hit in sorted(hits, key=lambda x: len(x.get("matched_text", "")), reverse=True):
        term = hit.get("matched_text", "")
        if term:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(
                f"<mark style='background-color: #ffd54f; padding: 0 4px; border-radius: 3px;'>{term}</mark>",
                highlighted,
            )
    return highlighted


def render_keyword_results(hits: list[dict]):
    if not hits:
        st.info("Anahtar kelime eşleşmesi bulunamadı.")
        return

    for hit in hits:
        with st.container():
            st.markdown(f"**{hit.get('rule_name', hit.get('keyword_name', 'Bilinmeyen'))}**")
            st.markdown(f"Bulunan: `{hit.get('matched_text', '')}`")
            st.caption(
                f"Kural ID: {hit.get('rule_id', '')} | Güven: %{hit.get('confidence', 0) * 100:.0f} | Seviye: {hit.get('severity', 'info').upper()}"
            )
