"""Streamlit UI helpers for keyword/topic detection."""

from __future__ import annotations

import html


SEVERITY_COLORS = {
    "critical": "#ef4444",
    "warning": "#f59e0b",
    "info": "#38bdf8",
}


def render_keyword_results(keyword_result: dict):
    import streamlit as st

    hits = keyword_result.get("hits", [])
    topics = keyword_result.get("topics", [])
    hit_count = keyword_result.get("hit_count", 0)

    st.markdown("---")
    st.markdown("### Anahtar Kelime & Konu Tespiti")

    if hit_count == 0:
        st.success("Tanımlı anahtar kelime veya konu eşleşmesi bulunamadı.")
        return

    st.warning(f"**{hit_count} anahtar kelime eşleşmesi** tespit edildi.")

    if topics:
        chips = " ".join(
            f'<span style="display:inline-block;padding:4px 12px;margin:4px;border-radius:999px;'
            f'background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);font-size:0.85rem;">'
            f'{html.escape(t["label_tr"])}</span>'
            for t in topics
        )
        st.markdown(f"**Tespit edilen konular:** {chips}", unsafe_allow_html=True)

    rows = ""
    for hit in hits:
        color = SEVERITY_COLORS.get(hit.get("severity", "info"), "#38bdf8")
        m, s = divmod(hit.get("timestamp_sec", 0), 60)
        time_fmt = f"{int(m):02d}:{s:04.1f}"
        rows += f"""<tr>
            <td>{time_fmt}</td>
            <td><span style="color:{color};font-weight:600;">{html.escape(hit.get("rule_name", ""))}</span></td>
            <td>{html.escape(hit.get("matched_text", ""))}</td>
            <td>{html.escape(hit.get("match_type", ""))}</td>
            <td>{hit.get("confidence", 0):.0%}</td>
        </tr>"""

    table = f"""<table class="asr-table">
    <thead><tr>
        <th>Zaman</th><th>Kural</th><th>Eşleşme</th><th>Mod</th><th>Güven</th>
    </tr></thead>
    <tbody>{rows}</tbody></table>"""
    st.markdown(table, unsafe_allow_html=True)

    if keyword_result.get("conversation_id"):
        st.caption(f"Kayıt ID: `{keyword_result['conversation_id']}` — Trend analitiği için API'ye kaydedildi.")


def highlight_transcript_html(segments_data, hits: list[dict]) -> str:
    """Build transcript table with keyword highlights."""
    hit_by_segment: dict[int, list[str]] = {}
    for hit in hits:
        idx = hit.get("segment_index", 0)
        hit_by_segment.setdefault(idx, []).append(hit.get("matched_text", ""))

    rows = ""
    for idx, segment in enumerate(segments_data):
        m, s = divmod(segment.start, 60)
        time_fmt = f"{int(m):02d}:{s:04.1f}"
        text = segment.text.strip()
        for matched in hit_by_segment.get(idx, []):
            if matched and matched.lower() in text.lower():
                import re
                pattern = re.compile(re.escape(matched), re.IGNORECASE)
                text = pattern.sub(
                    lambda m: f'<mark style="background:rgba(245,158,11,0.35);padding:0 2px;border-radius:2px;">{html.escape(m.group(0))}</mark>',
                    text,
                )
        rows += f"<tr><td>{time_fmt}</td><td>{text}</td></tr>"

    return f"""<table class="asr-table">
    <thead><tr><th style="width:120px;">Zaman</th><th>Konuşma Metni</th></tr></thead>
    <tbody>{rows}</tbody></table>"""
