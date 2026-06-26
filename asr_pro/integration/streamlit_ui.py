import re

import streamlit as st


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
            st.markdown(f"**{hit.get('keyword_name', 'Bilinmeyen')}**")
            st.markdown(f"Bulunan: `{hit.get('matched_text', '')}`")
            st.caption(f"Kural ID: {hit.get('rule_id', '')}")
