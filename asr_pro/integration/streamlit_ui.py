import streamlit as st

def render_keyword_results(results):
    pass

def highlight_transcript_html(segments_data, hits):
    html = "<div class='transcript-box'>"
    for s in segments_data:
        text = s.text if hasattr(s, 'text') else s.get('text', '')
        # Simple rendering without actual highlighting for now, to avoid missing module errors
        html += f"<p>{text}</p>"
    html += "</div>"
    return html
