import streamlit as st
import base64, re, json, time, psutil, html, math, os, platform, subprocess, sys, tempfile, wave, hashlib, warnings
from datetime import datetime, timedelta
from array import array
from difflib import SequenceMatcher
from pathlib import Path
from config import *
from logic_handlers import *
from asr_pro.core.trend_engine import get_trend_data, detect_anomalies, run_keyword_analysis
from asr_pro.core.keyword_engine import SegmentInput
from asr_pro.core.churn_engine import analyze_churn_risk
from asr_pro.core.sentiment_engine import analyze_sentiment
from asr_pro.core.compliance_engine import analyze_compliance_risk
from asr_pro.core.empathy_engine import analyze_soft_skills
from asr_pro.core.summary_engine import generate_crm_summary, generate_ollama_summary
KEYWORD_DETECTION_ENABLED = True
def display_detection_results(detected_swears, col_display):
    if detected_swears:
        col_display.error(f"**Uyarı: {len(detected_swears)} uygunsuz ifade tespit edildi.**")
        swear_report = [f" - **'{item['word']}'** anı: {item['time']:.2f}s" for item in detected_swears]
        col_display.markdown("\n".join(swear_report))
    else:
        col_display.success("Dosyada uygunsuz ifade tespit edilmedi.")

# --- PDF RAPORLAMA FONKSİYONU ---
PDF_FONT_CANDIDATES = [
    "Roboto-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

def get_font_path():
    """Türkçe karakter destekli yerel fontu bulur."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in PDF_FONT_CANDIDATES:
        path = candidate if os.path.isabs(candidate) else os.path.join(current_dir, candidate)
        if os.path.exists(path):
            return path
    return None

def replace_turkish_chars(text):
    """Türkçe karakterleri Latin-1 uyumlu hale getirir (Fallback)."""
    if not text: return ""
    replacements = {
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C'
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text

def pdf_safe_text(text, allow_unicode=True):
    """PDF motoru desteklemezse metni Latin-1 güvenli hale getirir."""
    if text is None:
        return ""
    text = str(text)
    if allow_unicode:
        return text
    text = replace_turkish_chars(text)
    return text.encode("latin-1", errors="replace").decode("latin-1")

def pdf_output_bytes(pdf):
    """PyFPDF 1.x hem str hem bytes dönebildiği için güvenli byte üretir."""
    raw = pdf.output(dest="S")
    if isinstance(raw, bytes):
        return raw
    return raw.encode("latin-1", errors="replace")

def create_pdf_report(filename, text_content, detected_swears, toxicity_info):
    """Analiz sonuçlarından profesyonel PDF raporu oluşturur."""
    fpdf_core.FPDF_CACHE_MODE = 2
    fpdf_core.FPDF_CACHE_DIR = tempfile.gettempdir()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    
    # Font Ayarı (Türkçe Karakter İçin)
    font_path = get_font_path()
    unicode_font_loaded = False
    
    if font_path and os.path.exists(font_path):
        try:
            font_family_name = "TurkishFont"
            pdf.add_font(font_family_name, "", font_path, uni=True)
            pdf.set_font(font_family_name, size=12)
            unicode_font_loaded = True
        except Exception:
            unicode_font_loaded = False

    if not unicode_font_loaded:
        pdf.set_font("Arial", size=12)
    
    clean = lambda value: pdf_safe_text(value, unicode_font_loaded)


    # Başlık
    pdf.set_font_size(20)
    pdf.cell(0, 10, txt=clean("ASR PRO - ANALİZ RAPORU"), ln=True, align='C')
    pdf.ln(10)

    # Dosya Bilgisi
    pdf.set_font_size(12)
    pdf.cell(0, 10, txt=clean(f"Dosya: {filename}"), ln=True)
    pdf.cell(0, 10, txt=clean(f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}"), ln=True)
    pdf.ln(5)

    # Özet İstatistikler
    pdf.set_font_size(14)
    pdf.cell(0, 10, txt=clean("ÖZET İSTATİSTİKLER"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()) # Çizgi
    pdf.ln(5)
    
    pdf.set_font_size(11)
    pdf.cell(0, 8, txt=clean(f"Uygunsuz İfade Sayısı: {len(detected_swears)}"), ln=True)
    pdf.cell(0, 8, txt=clean(f"Toksisite Durumu: {toxicity_info}"), ln=True)
    pdf.ln(10)

    # Detaylı Döküm
    pdf.set_font_size(14)
    pdf.cell(0, 10, txt=clean("DETAYLI DEŞİFRE VE ZAMAN DAMGALARI"), ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font_size(10)
    pdf.multi_cell(0, 6, txt=clean(text_content))
    
    return pdf_output_bytes(pdf)

def metric_tone(label, value):
    label_text = str(label or "").lower()
    value_text = str(value or "")
    numeric_match = re.search(r"-?\d+(?:\.\d+)?", value_text.replace(",", "."))
    numeric_value = float(numeric_match.group(0)) if numeric_match else None

    if any(key in label_text for key in ("güveni", "kalitesi", "doğruluğu")) and numeric_value is not None:
        if numeric_value >= 92:
            return "good"
        if numeric_value >= 75:
            return "warn"
        return "danger"
    if "word error" in label_text and numeric_value is not None:
        if numeric_value <= 5:
            return "good"
        if numeric_value <= 15:
            return "warn"
        return "danger"
    if "filtrelenen" in label_text and numeric_value is not None:
        return "good" if numeric_value == 0 else "warn"
    if "işlem" in label_text or "gerçek zaman" in label_text:
        return "info"
    return "neutral"

def render_quality_metric_cards(items):
    cards = []
    for label, value, note in items:
        tone = metric_tone(label, value)
        cards.append(
            f'<div class="quality-metric-card {tone}">'
            f'<div class="quality-metric-top">'
            f'<div class="quality-metric-label">{safe_html(label)}</div>'
            f'<div class="quality-metric-dot"></div>'
            f'</div>'
            f'<div class="quality-metric-value">{safe_html(value)}</div>'
            f'<div class="quality-metric-note">{safe_html(note)}</div>'
            f'</div>'
        )
    html_str = f'<div class="quality-metric-grid">{"".join(cards)}</div>'
    st.markdown(html_str, unsafe_allow_html=True)

def render_metric_summary(run_metrics, target_latency_s):
    rtf_value = run_metrics.get("rtf")
    audio_score = run_metrics.get("audio_quality_score")
    render_quality_metric_cards(
        [
            ("İşlem Süresi", f"{run_metrics['elapsed_s']:.1f}s", "Toplam analiz süresi"),
            ("Gerçek Zaman Oranı", f"{rtf_value:.2f}x" if rtf_value is not None else "-", "RTF performans değeri"),
            ("ASR Güveni", f"%{run_metrics['confidence']:.1f}", "Modelin metin güven skoru"),
            ("Filtrelenen Segment", str(run_metrics["filtered_segments"]), "Kalite filtresinden geçen parça"),
            ("Domain Düzeltme", str(run_metrics.get("domain_corrections", 0)), "Sektör sözlüğü düzeltmesi"),
            ("Ses Kalitesi", f"%{audio_score:.0f}" if audio_score is not None else "-", "Kayıt okunabilirlik sinyali"),
        ]
    )

    coverage = run_metrics.get("speech_coverage")
    coverage_text = f"%{coverage * 100:.0f}" if coverage is not None else "-"
    st.markdown(
        f'''
        <div class="quality-context-row">
            <div><span>Ön işleme</span><strong>{safe_html(run_metrics.get('preprocess_label', 'Standart Netleştirme'))}</strong></div>
            <div><span>Konuşma kapsaması</span><strong>{safe_html(coverage_text)}</strong></div>
            <div><span>Seçilen profil</span><strong>{safe_html(run_metrics.get('selected_profile_label', run_metrics.get('profile_label', '-')))}</strong></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if run_metrics.get("quality_retry"):
        retry_profiles = ", ".join(run_metrics.get("retry_profiles", [])) or "ikinci geçiş"
        st.info(f"Kalite kapısı düşük güven sinyali gördü; otomatik yeniden deneme çalıştı: {retry_profiles}.")
    elif run_metrics.get("quality_retry_skipped_for_latency"):
        st.warning("Kalite retry sinyali oluştu ancak 20 saniye hedefini korumak için ikinci geçiş atlandı. Referans metinle WER kontrolü önerilir.")
    if run_metrics.get("preprocess_profile") == AUDIO_PREP_RESCUE:
        st.info("Seçilen sonuç kötü ses kurtarma ön işlemesinden geldi: gürültü azaltma, konuşma normalizasyonu ve dinamik seviyeleme uygulandı.")
    if run_metrics.get("batched_fallback"):
        st.info("Hızlı batch motoru token sınırına yakalandı; sistem otomatik güvenli ASR geçişiyle devam etti.")
    if run_metrics.get("quality_gate_met"):
        st.success(f"Kurumsal kalite kapısı geçti: ASR güveni %{run_metrics.get('confidence', 0):.1f}, ses kalitesi %{run_metrics.get('audio_quality_score', 0):.0f}.")
    else:
        st.warning("Kurumsal kalite kapısı inceleme istiyor. Şirket teslimi için referans metinle WER ölçümü veya kayıt/sözlük iyileştirmesi önerilir.")
    if run_metrics["target_met"]:
        st.success(f"Hedef süre karşılandı: {run_metrics['elapsed_s']:.1f}s / {target_latency_s}s")
    else:
        st.warning(f"Hedef süre aşıldı: {run_metrics['elapsed_s']:.1f}s / {target_latency_s}s. Daha düşük model veya '20 sn Hedef' profili seçilebilir.")
    if run_metrics.get("confidence", 0) < 90:
        st.warning("Kalite sinyali düşük: bu kayıt için referans metinle WER ölçümü veya müşteri sözlüğü iyileştirmesi önerilir.")
    if run_metrics.get("audio_quality_score", 100) < AUDIO_QUALITY_REVIEW_THRESHOLD:
        notes = run_metrics.get("audio_quality_notes") or []
        st.warning("Ses kalite riski: " + " ".join(notes[:2]))

def render_wer_summary(reference_text, full_transcription):
    if not reference_text.strip():
        return None
    wer_stats = calculate_word_accuracy(reference_text, full_transcription)
    render_quality_metric_cards(
        [
            ("Kelime Doğruluğu", f"%{wer_stats['accuracy']:.1f}", "Referansa göre doğru kelime oranı"),
            ("Word Error Rate", f"%{wer_stats['wer'] * 100:.1f}", "Düşük olması gerekir"),
            ("Doğru Kelime Tahmini", f"{wer_stats['correct_estimate']}/{wer_stats['reference_words']}", "Yaklaşık doğru kelime sayısı"),
        ]
    )
    if wer_stats["accuracy"] >= QUALITY_GATE_ACCURACY:
        st.success(f"Kalite kapısı geçti: kelime doğruluğu %{QUALITY_GATE_ACCURACY:.0f} üstünde, WER %{QUALITY_GATE_WER:.0f} altında.")
    else:
        st.error(f"Kalite kapısı geçmedi: kelime doğruluğu %{wer_stats['accuracy']:.1f}. Bu kayıt teslim/üretim setine alınmadan önce sözlük veya ses kalitesi iyileştirilmeli.")
    return wer_stats

def render_reference_gate_hint(reference_text):
    if not reference_text.strip():
        st.info(f"Gerçek WER için referans metin gerekir. Referans girilince kalite kapısı: doğruluk >= %{QUALITY_GATE_ACCURACY:.0f}, WER <= %{QUALITY_GATE_WER:.0f}.")

def render_search_and_transcript(segments_data):
    st.markdown("#### Ses İçi Arama")
    search_query = st.text_input("Ses kaydı içinde kelime ara...", placeholder="Örn: bütçe, toplantı, rakam...")

    if search_query:
        found_count = 0
        st.write(f"**'{search_query}' için sonuçlar:**")
        search_html = '<div class="transcript-container">'
        for segment in segments_data:
            if search_query.lower() in segment.text.lower():
                found_count += 1
                m, s = divmod(segment.start, 60)
                time_fmt = f"{int(m):02d}:{s:04.1f}"
                text_content = html.escape(segment.text.strip())
                pattern = re.compile(re.escape(search_query), re.IGNORECASE)
                highlighted_text = pattern.sub(lambda m: f'<span class="search-highlight">{m.group(0)}</span>', text_content)
                search_html += f"""
                <div class="transcript-row">
                    <div class="transcript-time">{time_fmt}</div>
                    <div class="transcript-text">...{highlighted_text}...</div>
                </div>"""
        search_html += '</div>'
        
        if found_count == 0:
            st.info("Kelime bulunamadı.")
        else:
            st.markdown(search_html, unsafe_allow_html=True)
            st.caption(f"Toplam {found_count} eşleşme bulundu.")

    st.markdown("---")
    st.markdown("#### Deşifre Çıktısı")
    
    html_output = '<div class="transcript-container">'
    for segment in segments_data:
        m, s = divmod(segment.start, 60)
        time_formatted = f"{int(m):02d}:{s:04.1f}"
        text_content = html.escape(segment.text.strip())
        
        # Apply search highlight if there is an active search
        if search_query and search_query.lower() in segment.text.lower():
            pattern = re.compile(re.escape(search_query), re.IGNORECASE)
            text_content = pattern.sub(lambda m: f'<span class="search-highlight">{m.group(0)}</span>', text_content)
            
        html_output += f"""
        <div class="transcript-row">
            <div class="transcript-time">{time_formatted}</div>
            <div class="transcript-text">{text_content}</div>
        </div>"""
    html_output += '</div>'
    
    st.markdown(html_output, unsafe_allow_html=True)

def render_download_buttons(uploaded_name, formatted_text, segments_data, detected_swears, toxicity_label, negative_score):
    col_d1, col_d2, col_d3 = st.columns(3)
    base_name = os.path.splitext(uploaded_name)[0]
    with col_d1:
        st.download_button(
            label="📥 RAPOR (TXT)",
            data=formatted_text,
            file_name=f"ASR_Rapor_{base_name}.txt",
            mime="text/plain"
        )
    with col_d2:
        try:
            srt_content = create_srt(segments_data)
            st.download_button(
                label="🎬 ALTYAZI (SRT)",
                data=srt_content,
                file_name=f"{base_name}.srt",
                mime="text/plain"
            )
        except Exception as report_error:
            st.warning(f"SRT hazırlanamadı: {report_error}")
    with col_d3:
        try:
            tox_info = f"{toxicity_label} (%{negative_score*100:.1f})"
            pdf_bytes = create_pdf_report(uploaded_name, formatted_text, detected_swears, tox_info)
            st.download_button(
                label="📑 RAPOR (PDF)",
                data=pdf_bytes,
                file_name=f"ASR_Rapor_{base_name}.pdf",
                mime="application/pdf"
            )
        except Exception as report_error:
            st.warning(f"PDF hazırlanamadı: {report_error}")

def build_reading_paragraphs(text: str, sentences_per_paragraph: int = 4):
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if len(sentences) <= 1:
        words = normalized.split()
        return [" ".join(words[idx:idx + 70]) for idx in range(0, len(words), 70)]

    paragraphs = []
    for idx in range(0, len(sentences), sentences_per_paragraph):
        paragraphs.append(" ".join(sentences[idx:idx + sentences_per_paragraph]))
    return paragraphs

def render_paragraph_html(text: str):
    paragraphs = build_reading_paragraphs(text)
    if not paragraphs:
        return "<p>Metin bulunamadı.</p>"
    return "".join(f"<p>{safe_html(paragraph)}</p>" for paragraph in paragraphs)

@st.dialog("Ham ASR - Tam Metin", width="large")
def render_raw_asr_dialog(raw_transcription: str):
    words = normalize_for_wer(raw_transcription)
    char_count = len(raw_transcription or "")
    st.markdown(
        f'''
        <div class="raw-asr-fullscreen">
            <div class="raw-asr-fullscreen-head">
                <div>
                    <div class="raw-asr-kicker">Paragraf Okuma Modu</div>
                    <div class="raw-asr-subtitle">Ham ASR metni, denetim için geniş görünümde gösteriliyor.</div>
                </div>
                <div class="raw-asr-stats">
                    <span>{len(words)} kelime</span>
                    <span>{char_count} karakter</span>
                </div>
            </div>
            <div class="raw-asr-fullscreen-body">{render_paragraph_html(raw_transcription)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_raw_asr_panel(raw_transcription: str, key_prefix: str = "raw_asr"):
    words = normalize_for_wer(raw_transcription)
    char_count = len(raw_transcription or "")
    button_key = f"{key_prefix}_fullscreen_{hashlib.sha1((raw_transcription or '').encode('utf-8')).hexdigest()[:12]}"
    _, action_col = st.columns([0.72, 0.28])
    with action_col:
        if st.button("Tam Ekran Oku", key=button_key, use_container_width=True):
            render_raw_asr_dialog(raw_transcription)
    st.markdown(
        f'''
        <div class="raw-asr-panel">
            <div class="raw-asr-toolbar">
                <div>
                    <div class="raw-asr-kicker">Modelden Gelen Ham Metin</div>
                    <div class="raw-asr-subtitle">Domain düzeltmesi uygulanmadan önceki denetim çıktısı</div>
                </div>
                <div class="raw-asr-stats">
                    <span>{len(words)} kelime</span>
                    <span>{char_count} karakter</span>
                </div>
            </div>
            <div class="raw-asr-body">{safe_html(raw_transcription)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_cached_analysis_result(result, reference_text, target_latency_s, enable_wordcloud):
    display_detection_results(result["detected_swears"], st.empty())
    render_metric_summary(result["run_metrics"], target_latency_s)
    render_reference_gate_hint(reference_text)
    render_wer_summary(reference_text, result["full_transcription"])
    raw_transcription = result["run_metrics"].get("raw_transcription", "")
    if raw_transcription and raw_transcription != result["full_transcription"]:
        with st.expander("HAM ASR / DENETİM"):
            render_raw_asr_panel(raw_transcription, key_prefix="cached_raw_asr")
    if enable_wordcloud:
        with st.expander("KAVRAM HARİTASI (WORD CLOUD)", expanded=True):
            wc_fig = create_wordcloud(result["full_transcription"])
            if wc_fig:
                st.pyplot(wc_fig)
    render_search_and_transcript(result["segments_data"])
    render_download_buttons(
        result["uploaded_name"],
        result["formatted_text"],
        result["segments_data"],
        result["detected_swears"],
        result.get("toxicity_label", "Atlandı"),
        result.get("negative_score", 0.0),
    )

# --- SİSTEM METRİKLERİ ---
def get_system_metrics():
    """Sistem metriklerini anlık okur."""
    cpu_percent = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory()
    return cpu_percent, ram_usage

# --- CSS ENJEKSİYONU (ENTERPRISE CONTROL CENTER THEME) ---
def local_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --asr-bg: #09090b;
    --asr-panel: rgba(24, 24, 27, 0.7);
    --asr-stroke: rgba(255, 255, 255, 0.08);
    --asr-text: #f4f4f5;
    --asr-muted: #a1a1aa;
    --asr-accent: #6366f1;
    --asr-accent-hover: #4f46e5;
    --asr-success: #10b981;
    --asr-warning: #f59e0b;
    --asr-danger: #ef4444;
    --asr-radius: 12px;
    --asr-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    --asr-blur: blur(12px);
}

.stApp {
    background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.15), transparent 40%),
                radial-gradient(circle at bottom left, rgba(16, 185, 129, 0.1), transparent 40%),
                #09090b !important;
    background-attachment: fixed !important;
}

/* Hide Default Streamlit Elements to fix white gap */
#MainMenu, footer, header[data-testid="stHeader"] {
    background-color: transparent !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* Fix Unreadable Text in Widgets and Sidebar */
[data-testid="stSidebar"] label, .stRadio label, .stSelectbox label, .stToggle label, p, span, div {
    color: var(--asr-text);
}
[data-testid="stSidebar"] p {
    color: var(--asr-muted) !important;
}
[data-testid="stSidebar"] label p {
    color: var(--asr-muted) !important;
    font-weight: 600 !important;
}

/* Typography */
h1, h2, h3, h4, p, span, div {
    font-family: 'Outfit', sans-serif;
}

h1 {
    font-weight: 800 !important;
    font-size: 2.5rem !important;
    background: linear-gradient(135deg, #ffffff, #a1a1aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(9, 9, 11, 0.85) !important;
    backdrop-filter: var(--asr-blur) !important;
    border-right: 1px solid var(--asr-stroke) !important;
}

.brand-badge {
    background: linear-gradient(135deg, var(--asr-accent), #8b5cf6);
    width: 40px;
    height: 40px;
    border-radius: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: 800;
    color: #fff;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

.brand-title {
    font-weight: 700;
    font-size: 1.1rem;
    color: #fff;
}

/* Cards & Panels (Glassmorphism) */
.panel, .glass-card, .metric-box, .command-header, .kpi-card, .data-item, .feature-pill {
    background: var(--asr-panel) !important;
    backdrop-filter: var(--asr-blur) !important;
    border: 1px solid var(--asr-stroke) !important;
    border-radius: var(--asr-radius) !important;
    padding: 1.2rem;
    box-shadow: var(--asr-shadow) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--asr-accent), #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.5) !important;
    filter: brightness(1.1) !important;
}

/* Progress & Sliders */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--asr-accent), #8b5cf6) !important;
}

/* Fix Truncated Text in Streamlit Metrics */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p {
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
}

/* KPI Grid */
.kpi-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: #fff;
    margin: 0;
    line-height: 1;
}

/* Transcript Segment Overrides */
.speaker-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 0.75rem;
    margin-right: 8px;
}
.speaker-agent { background: rgba(99, 102, 241, 0.2); color: #818cf8; }
.speaker-customer { background: rgba(16, 185, 129, 0.2); color: #34d399; }

/* Quality Metric Cards */
.quality-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
    margin-bottom: 2rem;
}
.quality-metric-card {
    background: var(--asr-panel);
    border: 1px solid var(--asr-stroke);
    border-radius: var(--asr-radius);
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.quality-metric-card.good { border-top: 3px solid var(--asr-success); }
.quality-metric-card.warn { border-top: 3px solid var(--asr-warning); }
.quality-metric-card.danger { border-top: 3px solid var(--asr-danger); }
.quality-metric-card.info { border-top: 3px solid var(--asr-accent); }
.quality-metric-card.neutral { border-top: 3px solid var(--asr-muted); }

.quality-metric-top { display: flex; align-items: center; justify-content: space-between; }
.quality-metric-label { font-size: 0.85rem; color: var(--asr-muted); font-weight: 500; text-transform: uppercase; }
.quality-metric-dot { width: 8px; height: 8px; border-radius: 50%; }
.good .quality-metric-dot { background-color: var(--asr-success); }
.warn .quality-metric-dot { background-color: var(--asr-warning); }
.danger .quality-metric-dot { background-color: var(--asr-danger); }
.info .quality-metric-dot { background-color: var(--asr-accent); }
.neutral .quality-metric-dot { background-color: var(--asr-muted); }

.quality-metric-value { font-size: 1.75rem; font-weight: 700; color: var(--asr-text); }
.quality-metric-note { font-size: 0.75rem; color: var(--asr-muted); line-height: 1.2; }

/* Quality Context Row */
.quality-context-row {
    display: flex; gap: 2rem; padding: 1rem; background: rgba(255,255,255,0.02);
    border-radius: var(--asr-radius); border: 1px solid var(--asr-stroke);
    margin-bottom: 1.5rem;
}
.quality-context-row div { display: flex; flex-direction: column; gap: 0.25rem; }
.quality-context-row span { font-size: 0.75rem; color: var(--asr-muted); text-transform: uppercase; }
.quality-context-row strong { font-size: 0.95rem; color: var(--asr-text); }

/* Top-Tier Transcript Timeline */
.transcript-container {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 1rem;
    background: rgba(15, 15, 18, 0.6);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    margin-top: 1rem;
}
.transcript-row {
    display: flex;
    gap: 1.2rem;
    padding: 0.8rem 1rem;
    border-radius: 12px;
    transition: all 0.3s ease;
    align-items: flex-start;
}
.transcript-row:hover {
    background: rgba(255, 255, 255, 0.04);
}
.transcript-time {
    font-family: 'SF Pro Text', 'Inter', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--asr-accent);
    background: rgba(99, 102, 241, 0.1);
    padding: 0.3rem 0.6rem;
    border-radius: 6px;
    white-space: nowrap;
    min-width: 70px;
    text-align: center;
}
.transcript-text {
    font-size: 1.05rem;
    color: #e4e4e7;
    line-height: 1.6;
    letter-spacing: 0.2px;
    font-weight: 400;
}
.search-highlight {
    background: rgba(16, 185, 129, 0.2);
    color: #fff;
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
    border-bottom: 2px solid var(--asr-success);
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)
def safe_html(value):
    return html.escape(str(value or ""))

def render_sidebar_module_map(active_mode):
    mode_to_group = {
        "Tek Dosya Analizi": "Çağrılar",
        "Canlı Dinleme (Mikrofon)": "Gerçek Zamanlı",
        "Toplu İşlem Merkezi": "Sistem Yönetimi",
        "📈 Trend ve Erken Uyarı Radarı": "Sistem Yönetimi",
    }
    active_group = mode_to_group.get(active_mode, "Çağrılar")
    groups = [
        ("Çağrılar", "Genel Görünüm, Arama, Kayıtlı Aramalar"),
        ("Tanımlamalar", "Tanımlamalar, Alarmlar, Temsilci Hedefleri"),
        ("Temsilci", "Performans Görünümü, Temsilci Listesi"),
        ("Kullanıcı Yönetimi", "Kullanıcılar, Kullanıcı Grupları, Rol Tanımlama"),
        ("Sistem Yönetimi", "Bileşenler, Toplu İşlem, Motor Sağlığı"),
    ]
    html_parts = ['<div class="sidebar-section-title">Ürün Modülleri</div><div class="sidebar-nav">']
    for title, subtitle in groups:
        active = " active" if title == active_group else ""
        html_parts.append(
            f'<div class="nav-group{active}">'
            f'<div class="nav-group-header"><span>{safe_html(title)}</span><span>{"Aktif" if active else "Hazır"}</span></div>'
            f'<div class="nav-subitem">{safe_html(subtitle)}</div>'
            f'</div>'
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

def render_command_header(mode, model_size, profile_key, domain_key, runtime_device):
    profile = ASR_PROFILES.get(profile_key, ASR_PROFILES["smart"])
    domain = get_domain_profile(domain_key)
    st.markdown(
        f'''
        <style>
        .command-header {{ display: flex; flex-direction: column; gap: 1.5rem; padding: 2rem !important; margin-bottom: 2rem !important; }}
        .header-row {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }}
        .eyebrow {{ font-weight: 800; color: var(--asr-accent) !important; letter-spacing: 1px; margin-bottom: 0.5rem; text-transform: uppercase; font-size: 0.8rem; }}
        .command-header h1 {{ margin-bottom: 0.75rem !important; font-size: 1.8rem !important; font-weight: 700; line-height: 1.3; color: white !important; }}
        .subtitle {{ font-size: 1rem; color: var(--asr-muted) !important; max-width: 85%; line-height: 1.6; }}
        .data-strip {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-top: 1rem; border-top: 1px solid var(--asr-border); padding-top: 1.5rem; }}
        .data-item {{ display: flex; flex-direction: column; gap: 0.4rem; padding: 0.5rem; }}
        .data-item strong {{ font-size: 1.3rem; color: #fff !important; font-weight: 800; }}
        .data-item span {{ font-size: 0.8rem; color: var(--asr-muted) !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 700; }}
        .status-line {{ display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: flex-end; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.25rem; margin-bottom: 1.5rem; }}
        .kpi-card {{ display: flex; flex-direction: column; gap: 0.5rem; padding: 1.25rem !important; }}
        .kpi-label {{ font-size: 0.75rem; text-transform: uppercase; font-weight: 800; color: var(--asr-muted) !important; }}
        .kpi-value {{ font-size: 1.75rem !important; font-weight: 800; color: white !important; }}
        .kpi-foot {{ font-size: 0.8rem; color: var(--asr-accent) !important; font-weight: 600; }}
        .panel {{ padding: 1.5rem !important; display: flex; flex-direction: column; gap: 0.5rem; }}
        .panel-title {{ display: flex; justify-content: space-between; align-items: center; font-size: 1.15rem; font-weight: 800; color: white !important; margin-bottom: 0.25rem; }}
        .panel-caption {{ font-size: 0.9rem; color: var(--asr-muted) !important; line-height: 1.5; margin-bottom: 1rem; }}
        .feature-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 0.5rem; }}
        .feature-pill {{ padding: 0.6rem 1rem !important; text-align: center; font-size: 0.85rem; font-weight: 600; border-radius: 8px !important; }}
        .empty-state {{ padding: 3rem 2rem !important; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; gap: 0.5rem; }}
        .empty-state strong {{ font-size: 1.1rem; color: white !important; }}
        .empty-state span {{ font-size: 0.9rem; color: var(--asr-muted) !important; }}
        </style>
        <div class="command-header">
            <div class="header-row">
                <div>
                    <div class="eyebrow">Enterprise Speech Intelligence</div>
                    <h1>Kurumsal ses kayıtlarını denetlenebilir metne dönüştür.</h1>
                    <div class="subtitle">
                        Adaptif kötü ses kurtarma, sektör sözlüğü ve kalite kapısı tek akışta. Hedef: Yüksek kelime doğruluğu ve optimize edilmiş hata oranı.
                    </div>
                </div>
                <div class="status-line">
                    <span class="chip good">Canlı Sistem</span>
                    <span class="chip info">{safe_html(runtime_device.upper())}</span>
                    <span class="chip warn">{safe_html(mode)}</span>
                </div>
            </div>
            <div class="data-strip">
                <div class="data-item"><strong>{safe_html(model_size.upper())}</strong><span>Model Sınıfı</span></div>
                <div class="data-item"><strong>{safe_html(profile.label)}</strong><span>ASR Profili</span></div>
                <div class="data-item"><strong>{safe_html(domain.label)}</strong><span>Sektör Sözlüğü</span></div>
                <div class="data-item"><strong>%{QUALITY_GATE_ACCURACY:.0f}</strong><span>Doğruluk Hedefi</span></div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_kpi_grid(cpu_percent, ram_usage, target_latency_s, runtime_device):
    ram_percent = int(ram_usage.percent)
    active_calls = 4703
    daily_calls = 357
    agents = 94
    avg_duration = "2 dk 59 sn"
    st.markdown(
        f'''
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Temsilci Sayısı</div>
                <div class="kpi-value">{agents}</div>
                <div class="kpi-foot">Bugün aktif</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Çağrı Sayısı</div>
                <div class="kpi-value">{daily_calls}</div>
                <div class="kpi-foot">Günlük işlenen</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Aktif Çağrılar</div>
                <div class="kpi-value">{active_calls}</div>
                <div class="kpi-foot">Kuyrukta bekleyen</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Ortalama Süre</div>
                <div class="kpi-value">{avg_duration}</div>
                <div class="kpi-foot">SLA {target_latency_s} sn analiz</div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-title">
                <span>Operasyon Özeti</span>
                <span class="chip info">CPU %{cpu_percent:.0f} · RAM %{ram_percent} · {safe_html(runtime_device.upper())}</span>
            </div>
            <div class="feature-grid">
                <div class="feature-pill">Zaman damgalı deşifre</div>
                <div class="feature-pill">Toksisite & risk skoru</div>
                <div class="feature-pill">Sektör sözlüğü</div>
                <div class="feature-pill">WER kalite kapısı</div>
                <div class="feature-pill">TXT, SRT, PDF çıktı</div>
                <div class="feature-pill">Toplu işleme</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_panel(title, caption="", right_label=""):
    right_html = f'<span class="chip info">{safe_html(right_label)}</span>' if right_label else ""
    st.markdown(
        f'''
        <div class="panel">
            <div class="panel-title"><span>{safe_html(title)}</span>{right_html}</div>
            <div class="panel-caption">{safe_html(caption)}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def render_empty_state(title, subtitle):
    st.markdown(
        f'''
        <div class="empty-state">
            <div class="blinking-cursor"></div>
            <strong>{safe_html(title)}</strong>
            <span>{safe_html(subtitle)}</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )


# --- CONTROL RAIL (ALWAYS VISIBLE) ---
def render_app():
    local_css()
    control_col, workspace_col = st.columns([0.36, 1.0], gap="large")
    with control_col:
        st.markdown("""
            <div class="sidebar-header">
                <div class="brand-mark">
                    <div class="brand-badge">ASR</div>
                    <div>
                        <div class="brand-title">ASR Command</div>
                        <div class="brand-subtitle">Model, kalite ve raporlama</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">İş akışı</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Seçiniz",
            ("Tek Dosya Analizi", "Canlı Dinleme (Mikrofon)", "Toplu İşlem Merkezi", "📈 Trend ve Erken Uyarı Radarı"),
        )

        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">🧠 Yapay Zeka Yetenekleri</div>', unsafe_allow_html=True)
        
        enable_empathy = st.toggle("💖 Empati ve Soft Skill Analizi", value=True, help="Temsilcinin aktif dinleme, şefkat ve robotik dil kullanımını analiz eder.")
        enable_autonote = st.toggle("📝 CRM Kapanış Notu (Auto-Note)", value=True, help="Çağrıyı edebi bir dille özetler ve CRM'e hazır bilet oluşturur.")
        enable_churn = st.toggle("⚠️ İptal (Churn) ve Risk Analizi", value=True, help="Müşterinin abonelik iptal riskini ve kurumsal uyumsuzlukları ölçer.")
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        ollama_model = st.selectbox("🔒 Lokal LLM Motoru (Özetleme İçin)", ["Kapalı (Sadece Yerel Motor)", "llama3", "llama3.1", "mistral", "gemma"])
        st.caption("Yerel LLM'in açık olması için terminalde 'ollama run llama3' yazmalısınız.")

        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">💻 Donanım ve Model Seçimi</div>', unsafe_allow_html=True)
        
        hardware_engine = st.radio(
            "Altyapı",
            ["🖥️ Windows (Nvidia CUDA / Standart)", "🍏 Mac (Apple Silicon MLX - Çok Hızlı)"],
            index=0,
            help="Sistemi çalıştırdığınız bilgisayara göre motor seçin.",
            label_visibility="collapsed"
        )
        st.session_state["hardware_engine"] = hardware_engine
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

        # Model Seçimi - Tüm modeller (CPU optimizasyonlu)
        model_choices = tuple(MODEL_NAME_MAP.keys())
        default_model_index = model_choices.index(DEFAULT_MODEL_SIZE) if DEFAULT_MODEL_SIZE in model_choices else 0
        model_size = st.selectbox(
            "Model",
            model_choices,
            index=default_model_index,
            help="Large en yüksek doğruluk için, Turbo daha hızlı demo için uygundur."
        )

        st.caption(f"{model_size.upper()}: {MODEL_INFO.get(model_size, '')}")

        apple_mode = st.toggle("🍎 Yüksek Doğruluk (Konsensüs) Modu", value=False, help="Açıkken 3 farklı stratejiyle aynı sesi çözer, segment bazında en iyisini seçer. Yüksek doğruluk sağlar ama daha yavaştır.")
        
        profile_label_to_key = {profile.label: key for key, profile in ASR_PROFILES.items()}
        profile_keys = tuple(profile_label_to_key.values())
        default_profile_key = "mac_turbo_sla" if wants_apple_mlx_engine(hardware_engine) else "ultimate_hybrid"
        default_profile_index = profile_keys.index(default_profile_key) if default_profile_key in profile_keys else 0
        selected_profile_label = st.selectbox(
            "ASR Profili",
            tuple(profile_label_to_key.keys()),
            index=default_profile_index,
            help="Kötü seslerde en güvenli başlangıç için varsayılan profili koruyabilirsin.",
            disabled=apple_mode,
        )
        st.session_state["apple_mode"] = apple_mode
        if apple_mode:
            profile_key = "apex_quality"
            st.caption("🍎 Yüksek Doğruluk Modu aktif: 3 geçişli konsensüs dekodlama ile daha güvenilir sonuçlar.")
        else:
            profile_key = profile_label_to_key[selected_profile_label]
            st.caption(ASR_PROFILES[profile_key].description)

        domain_label_to_key = {profile.label: key for key, profile in DOMAIN_PROFILES.items()}
        selected_domain_label = st.selectbox(
            "Sektör sözlüğü",
            tuple(domain_label_to_key.keys()),
            index=0,
            help="Şirketin sektörüne göre terim ve düzeltme katmanını seçer"
        )
        domain_key = domain_label_to_key[selected_domain_label]
        st.caption(DOMAIN_PROFILES[domain_key].description)

        target_latency_s = st.slider("Hedef Süre (sn)", 5, 60, TARGET_LATENCY_SECONDS, 5)
        st.session_state["target_latency_s"] = target_latency_s
        if apple_mode and wants_apple_mlx_engine(hardware_engine) and target_latency_s <= TARGET_LATENCY_SECONDS:
            st.caption("20 sn hedefinde Apple konsensüs yerine hızlı MLX SLA geçişi çalıştırılır.")
        hotwords_input = st.text_area(
            "Özel kelimeler",
            value="",
            height=70,
            placeholder="Şekerbank, KMH, Vodafone, Yanımda, Red tarifesi, şirket ürünleri...",
            help="Virgül veya satır satır marka, ürün, kampanya, kişi ve teknik terim girin"
        )

        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Analiz katmanları</div>', unsafe_allow_html=True)

        translate_mode = st.checkbox("Çeviri Modu", value=True, help="Türkçe konuşmayı İngilizce'ye çevirerek yazar.")
        enable_nlp = st.checkbox("Toksisite Analizi", value=True, help="Metin toksisite analizi yapar (ek işlem süresi gerektirir).")
        enable_wordcloud = st.checkbox("Kelime Bulutu", value=True, help="Kapalıyken sonuç ekranı daha hızlı açılır.")
        diarization_mode = st.checkbox("Konuşmacı Ayrımı", value=True, help="Konuşmacıları A/B olarak ayırır (HF Token Gerekli).")
        reference_text = st.text_area(
            "Referans Metin (WER)",
            value="",
            height=90,
            placeholder="Gerçek metni buraya yapıştırırsanız WER/doğruluk hesaplanır.",
            help="Boş bırakılırsa gerçek WER kanıtlanamaz; yalnızca ASR güven ve ses kalite skoru gösterilir"
        )
        
        hf_token = ""
        if diarization_mode:
            hf_token = st.text_input("Hugging Face Token", type="password", help="Pyannote modeli için gereklidir.")
            if not hf_token:
                st.caption("Diarization aktif ama token girilmedi.")

        # SYSTEM HEALTH SECTION (Cached)
        st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-title">Sistem sağlığı</div>', unsafe_allow_html=True)

        # Cache'lenmiş Sistem İstatistikleri
        cpu_percent, ram_usage = get_system_metrics()
        ram_used_gb = round(ram_usage.used / (1024 ** 3), 1)
        ram_total_gb = round(ram_usage.total / (1024 ** 3), 1)
        ram_percent = ram_usage.percent
        runtime_device, runtime_compute = describe_runtime_engine(hardware_engine)
        
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">İşlemci Kullanımı</div>
                <div class="metric-value">{cpu_percent}%</div>
                <div class="metric-bar-bg"><div class="metric-bar-fill" style="width: {cpu_percent}%"></div></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Bellek ({ram_used_gb}/{ram_total_gb} GB)</div>
                <div class="metric-value">{ram_used_gb} GB</div>
                <div class="metric-bar-bg"><div class="metric-bar-fill" style="width: {ram_percent}%"></div></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Çalışma Motoru</div>
                <div class="metric-value" style="color:var(--asr-success)">{runtime_device.upper()} / {runtime_compute}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center; color:#94a3b8; font-size:0.72rem; margin-top:0.85rem;">
                <span>SUNUCU</span>
                <span style="color:var(--asr-success)">ÇEVRİMİÇİ</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; color:#94a3b8; font-size:0.72rem; margin-top:0.35rem;">
                <span>MOTOR</span>
                <span>WHISPER AI v3</span>
            </div>
        """, unsafe_allow_html=True)

    with workspace_col:
        render_command_header(mode, model_size, profile_key, domain_key, runtime_device)

        # --------------------------
        # --- TEKLİ DOSYA İŞLEME ---
        # --------------------------
        if mode == "Tek Dosya Analizi":
        
            col1, col2 = st.columns([1, 1.2], gap="large")

            with col1:
                render_panel(
                    "Ses Veri Girişi",
                    "MP3, WAV, M4A veya FLAC kaydı yükleyin. Dosya kurumsal ASR için 16 kHz mono WAV formatına hazırlanır.",
                    "Tek kayıt"
                )
            
                uploaded_file = st.file_uploader("Dosya Seç", type=["mp3", "wav", "m4a", "flac"], label_visibility="collapsed")
            
                audio_path = None
                if uploaded_file is not None:
                    audio_path = os.path.join(TEMP_AUDIO_DIR, uploaded_file.name)
                    with open(audio_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                    st.success(f"Kaynak alındı: {uploaded_file.name}")
                    try:
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format=uploaded_file.type)
                    except Exception:
                        pass


            with col2:
                render_panel(
                    "Analiz Merkezi",
                    "Model yükleme, kalite kapısı, sektör sözlüğü, toksisite ve rapor üretimi bu bölümden yönetilir.",
                    "Hazır"
                )
            
                swear_display_placeholder = st.empty() 
                nlp_display_placeholder = st.empty()
                render_panel(
                    "Çıktı / Deşifre Sonucu",
                    "Analiz başladığında canlı deşifre burada görünür. İşlem bitince metin, kalite skorları ve indirme dosyaları aynı bölümde kalır.",
                    "Bekliyor"
                )
                output_placeholder = st.empty()

                if audio_path is not None:
                    st.markdown("<br>", unsafe_allow_html=True)
                    output_placeholder.markdown(
                        """
                        <div class="output-frame waiting" class="empty-state">
                            <div>
                                <div class="output-kicker" style="color:var(--asr-accent); font-weight:800; text-transform:uppercase; font-size:0.8rem; margin-bottom:0.5rem;">Çıktı alanı hazır</div>
                                <div class="output-title" style="color:var(--asr-text); font-size:1.1rem; font-weight:600; margin-bottom:0.5rem;">Analizi başlatınca sonuç burada oluşacak.</div>
                                <div class="output-copy" style="color:var(--asr-muted); font-size:0.9rem;">Model çalışırken canlı metin akışı, işlem bitince deşifre tablosu ve indirme butonları bu alanda görünecek.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.button("Analizi Başlat", type="primary"):
                        # --- LAZY LOADING ---
                        output_placeholder.markdown(
                            """
                            <div class="output-frame waiting" class="empty-state">
                                <div>
                                    <div class="output-kicker" style="color:var(--asr-accent); font-weight:800; text-transform:uppercase; font-size:0.8rem; margin-bottom:0.5rem;">İşlem başladı</div>
                                    <div class="output-title" style="color:var(--asr-text); font-size:1.1rem; font-weight:600; margin-bottom:0.5rem;">Model yükleniyor ve ses hazırlanıyor.</div>
                                    <div class="output-copy" style="color:var(--asr-muted); font-size:0.9rem;">İlk segment çözüldüğünde canlı deşifre bu kutuya düşecek.</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.spinner("YZ modelleri yükleniyor (Faster-Whisper)..."):
                            model = load_whisper_model(model_size, st.session_state.get("hardware_engine", "Windows"))
                            nlp_classifier = load_toxicity_classifier() if enable_nlp else None
                    
                        with st.spinner("Ses ön işleme ve deşifre çalışıyor..."):
                            try:
                                task_type = "translate" if translate_mode else "transcribe"
                                live_transcript_placeholder = output_placeholder
                                progress_bar = st.progress(0)

                                def update_live_transcript(segment, current_text, elapsed, prep_info):
                                    duration = prep_info.get("duration")
                                    if duration:
                                        progress_bar.progress(min(float(segment.end) / duration, 1.0))
                                    escaped_text = html.escape(current_text[-4000:])
                                    live_transcript_placeholder.markdown(
                                        f"""
                                        <div class="live-output-box" class="glass-card" style="margin-top:1rem;">
                                            {escaped_text}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                if st.session_state.get("apple_mode", False):
                                    formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = consensus_transcribe(
                                        model,
                                        audio_path,
                                        LANGUAGE,
                                        TURKISH_SWEAR_WORDS,
                                        task=task_type,
                                        domain_key=domain_key,
                                        hotwords=hotwords_input,
                                        progress_callback=update_live_transcript,
                                        target_latency_s=target_latency_s,
                                    )
                                else:
                                    formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = transcribe_audio_file(
                                        model,
                                        audio_path,
                                        LANGUAGE,
                                        TURKISH_SWEAR_WORDS,
                                        task=task_type,
                                        profile_key=profile_key,
                                        domain_key=domain_key,
                                        hotwords=hotwords_input,
                                        progress_callback=update_live_transcript,
                                        target_latency_s=target_latency_s,
                                    )
                                progress_bar.progress(1.0)
                                output_placeholder.markdown(
                                    f"""
                                    <div class="output-frame" class="empty-state">
                                        <div class="output-kicker" style="color:var(--asr-accent); font-weight:800; text-transform:uppercase; font-size:0.8rem; margin-bottom:0.5rem;">Deşifre hazır</div>
                                        <div class="output-title" style="color:var(--asr-text); font-size:1.1rem; font-weight:600; margin-bottom:0.5rem;">Çıktı üretildi: {len(segments_data)} segment, {len(normalize_for_wer(full_transcription))} kelime.</div>
                                        <div class="output-copy" style="color:var(--asr-muted); font-size:0.9rem;">Aşağıda kalite metrikleri, deşifre tablosu ve TXT/SRT/PDF indirme seçenekleri yer alıyor.</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            
                                # --- SONUÇLARI GÖSTER ---
                                display_detection_results(detected_swears, swear_display_placeholder)

                                render_metric_summary(run_metrics, target_latency_s)
                                
                                # --- AI CHURN & COMPLIANCE ANALYTICS ---
                                if KEYWORD_DETECTION_ENABLED:
                                    segment_inputs = [
                                        SegmentInput(start=seg.start, end=seg.end, text=seg.text, segment_index=i) 
                                        for i, seg in enumerate(segments_data)
                                    ]
                                    
                                    if enable_churn:
                                        st.markdown("---")
                                        st.markdown("### 🧠 Top-Tier AI Analytics (Churn & Emotion)")
                                        with st.spinner("Yapay Zeka (MPS Neural Engine) analiz ediyor..."):
                                            churn_res = analyze_churn_risk(segment_inputs)
                                            
                                            # Duygu analizi için çağrının son kısımlarını (en güncel duyguyu) alalım
                                            sentiment_res = analyze_sentiment(segment_inputs[-1]) if segment_inputs else None
                                            
                                            # CHURN METRIC
                                            churn_pct = int(churn_res.risk_score * 100)
                                            churn_color = "🔴" if churn_res.is_high_risk else "🟢"
                                            
                                            # WPM METRIC
                                            wpm = churn_res.average_wpm
                                            wpm_color = "⚠️" if wpm > 160 else "⏱️"
                                            
                                            # EMOTION METRIC
                                            emotion = sentiment_res.emotion_category if sentiment_res else "Nötr İletişim"
                                            
                                            st.markdown(
                                                f'''
                                                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; margin-bottom: 1.5rem;">
                                                    <div class="kpi-card" style="padding: 1.25rem;">
                                                        <div class="kpi-label" style="font-size: 0.85rem; color: #a1a1aa; margin-bottom: 0.5rem;">{churn_color} Churn Riski</div>
                                                        <div class="kpi-value" style="font-size: 1.8rem; font-weight: 800; color: white; word-break: break-word; white-space: normal;">%{churn_pct}</div>
                                                    </div>
                                                    <div class="kpi-card" style="padding: 1.25rem;">
                                                        <div class="kpi-label" style="font-size: 0.85rem; color: #a1a1aa; margin-bottom: 0.5rem;">{wpm_color} Akustik Stres</div>
                                                        <div class="kpi-value" style="font-size: 1.8rem; font-weight: 800; color: white; word-break: break-word; white-space: normal;">{wpm} WPM</div>
                                                    </div>
                                                    <div class="kpi-card" style="padding: 1.25rem;">
                                                        <div class="kpi-label" style="font-size: 0.85rem; color: #a1a1aa; margin-bottom: 0.5rem;">🎭 Çağrı Sonu Duygusu</div>
                                                        <div class="kpi-value" style="font-size: 1.8rem; font-weight: 800; color: white; word-break: break-word; white-space: normal; line-height: 1.1;">{emotion}</div>
                                                    </div>
                                                </div>
                                                ''', unsafe_allow_html=True
                                            )
                                            
                                            if churn_res.is_high_risk:
                                                st.error(f"🚨 **YÜKSEK RİSK TESPİT EDİLDİ!** Müşteri çağrı sonuna doğru ayrılma tehdidi veya yoğun memnuniyetsizlik (Akustik: {wpm} WPM) gösterdi.")
                                            if churn_res.competitors_mentioned:
                                                st.warning(f"🏢 **Rekabet Alarmı:** Rakip firma anıldı -> {', '.join(churn_res.competitors_mentioned).title()}")
                                                
                                            # --- COMPLIANCE MONITORING ---
                                            compliance_res = analyze_compliance_risk(segment_inputs, domain_key=domain_key)
                                            if compliance_res:
                                                st.markdown("### 🏛️ Regülasyon ve Uyum Denetimi")
                                                for vio in compliance_res:
                                                    icon = "🛑" if vio.severity == "CRITICAL" else "⚠️"
                                                    color = "#ff4b4b" if vio.severity == "CRITICAL" else "#ffb200"
                                                    
                                                    st.markdown(f"""
                                                    <div style="background-color: rgba(255, 0, 0, 0.05); border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                                                        <h4 style="margin:0; color: {color};">{icon} {vio.severity} İHLAL: {vio.category}</h4>
                                                        <p style="margin:5px 0 0 0; color: #f0f2f6;"><b>Gerekçe:</b> {vio.reason}</p>
                                                        <p style="margin:5px 0 0 0; color: #f0f2f6;"><b>Temsilci Beyanı:</b> <i>"{vio.segment_text}"</i> <span style="color: gray; font-size: 0.8em;">[{vio.timestamp_start:.1f}s - {vio.timestamp_end:.1f}s]</span></p>
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                                    
                                    # --- EMPATHY & SOFT SKILLS ---
                                    if enable_empathy:
                                        empathy_res = analyze_soft_skills(segment_inputs)
                                        st.markdown("### 💖 Empati ve Soft Skill Analizi")
                                        
                                        # Skor Rengi
                                        emp_color = "green" if empathy_res.score >= 80 else "orange" if empathy_res.score >= 50 else "red"
                                        emp_emoji = "🌟" if empathy_res.score >= 80 else "⚠️" if empathy_res.score >= 50 else "🛑"
                                        
                                        st.markdown(f"**Empati Skoru:** `{empathy_res.score} / 100` {emp_emoji}")
                                        st.progress(empathy_res.score / 100.0)
                                        
                                        if empathy_res.analysis_summary:
                                            st.info(empathy_res.analysis_summary)
                                            
                                        emp_cols = st.columns(3)
                                        with emp_cols[0]:
                                            if empathy_res.active_listening_hits:
                                                st.success("✅ **Aktif Dinleme:**\n" + ", ".join(f"'{h}'" for h in empathy_res.active_listening_hits))
                                            else:
                                                st.error("❌ **Aktif Dinleme:** Tespit edilemedi")
                                                
                                        with emp_cols[1]:
                                            if empathy_res.compassion_hits:
                                                st.success("✅ **Şefkat / Özür:**\n" + ", ".join(f"'{h}'" for h in empathy_res.compassion_hits))
                                                
                                            if empathy_res.solution_hits:
                                                st.success("✅ **Çözüm Odaklılık:**\n" + ", ".join(f"'{h}'" for h in empathy_res.solution_hits))
                                                
                                        with emp_cols[2]:
                                            if empathy_res.defensive_hits:
                                                st.error("🛑 **Robotik/Defansif:**\n" + ", ".join(f"'{h}'" for h in empathy_res.defensive_hits))
                                            else:
                                                st.success("✅ **Robotik Dil:** Yok")
                                                
                                # ----------------------------------------

                                render_reference_gate_hint(reference_text)

                                if reference_text.strip():
                                    wer_stats = render_wer_summary(reference_text, full_transcription)
                                    formatted_text += f"Kelime Doğruluğu: %{wer_stats['accuracy']:.1f}\n"
                                    formatted_text += f"WER: %{wer_stats['wer'] * 100:.1f}\n"
                                    formatted_text += f"Doğru Tahmini: {wer_stats['correct_estimate']}/{wer_stats['reference_words']}\n"

                                raw_transcription = run_metrics.get("raw_transcription", "")
                                if raw_transcription and raw_transcription != full_transcription:
                                    with st.expander("Ham ASR / Denetim"):
                                        render_raw_asr_panel(raw_transcription, key_prefix="single_raw_asr")

                                # NLP analizi sadece aktifse
                                toxicity_label, negative_score = ("Atlandı", 0.0)
                                if enable_nlp:
                                    with st.spinner("Toksisite analizi yapılıyor..."):
                                            if nlp_classifier:
                                                toxicity_label, negative_score = analyze_toxicity(full_transcription, nlp_classifier)
                                            
                                                # NLP Bazen Argo Kelimeleri Kaçırabilir, Eğer Küfür Listesinde Eşleşme Varsa Skoru Artır
                                                if len(detected_swears) > 0:
                                                    # En az %45 toksik olarak işaretle (Orta Toksisite Başlangıcı)
                                                    # Her küfür için +0.05 ekle
                                                    base_swear_score = 0.45 + (len(detected_swears) * 0.05)
                                                    negative_score = max(negative_score, min(0.99, base_swear_score))
                                                
                                                    if negative_score > 0.7:
                                                        toxicity_label = "Yüksek Toksik (Küfür + NLP)"
                                                    elif negative_score > 0.4:
                                                        toxicity_label = "Orta Toksik (Küfür + NLP)"
                                            else:
                                                # NLP yüklenemedi - küfür sayısına göre hesapla (FALLBACK)
                                                st.info("NLP modeli yüklenemedi. Küfür tespitine dayalı analiz yapılıyor...")
                                            
                                                # Tespit edilen küfür sayısına göre toksisite hesapla
                                                swear_count = len(detected_swears)
                                                word_count = len(full_transcription.split())
                                            
                                                if word_count > 0 and swear_count > 0:
                                                    # Küfür oranı: (küfür sayısı / toplam kelime) * ağırlık faktörü
                                                    raw_ratio = swear_count / word_count
                                                    # Her küfür için %10-15 baz toksisite ekle, max %95
                                                    negative_score = min(0.95, swear_count * 0.12 + raw_ratio * 2)
                                                
                                                    if negative_score > 0.7:
                                                        toxicity_label = "Yüksek Toksik (Küfür Bazlı)"
                                                    elif negative_score > 0.4:
                                                        toxicity_label = "Orta Toksik (Küfür Bazlı)"
                                                    else:
                                                        toxicity_label = "Düşük Toksik (Küfür Bazlı)"
                                                else:
                                                    negative_score = 0.0
                                                    toxicity_label = "Temiz (Küfür Yok)"
                                
                                    # Her durumda sonucu göster
                                    score_percent = negative_score * 100
                                
                                    # Toksisite sonuç kartı
                                    st.markdown("---")
                                    st.markdown("### Toksisite Analiz Sonucu")
                                
                                    # Küfür sayısı bilgisi
                                    if len(detected_swears) > 0:
                                        st.warning(f"**{len(detected_swears)} adet uygunsuz ifade tespit edildi.**")
                                
                                    tox_col1, tox_col2 = st.columns(2)
                                
                                    with tox_col1:
                                        if negative_score > 0.7:
                                            st.error("**Yüksek Toksisite**")
                                        elif negative_score > 0.4:
                                            st.warning("**Orta Toksisite**")
                                        elif negative_score > 0.1:
                                            st.info("**Düşük Toksisite**")
                                        else:
                                            st.success("**Temiz İçerik**")
                                
                                    with tox_col2:
                                        render_quality_metric_cards(
                                            [
                                                ("Toksisite Oranı", f"%{score_percent:.1f}", toxicity_label),
                                            ]
                                        )
                                
                                    # Detaylı açıklama
                                    if negative_score > 0.7:
                                        st.error(f"Bu ses kaydı **%{score_percent:.1f}** oranında toksik içerik barındırıyor. ({len(detected_swears)} küfür)")
                                    elif negative_score > 0.4:
                                        st.warning(f"Bu ses kaydı **%{score_percent:.1f}** oranında olumsuz ifade içeriyor. ({len(detected_swears)} küfür)")
                                    elif negative_score > 0.1:
                                        st.info(f"Bu ses kaydı **%{score_percent:.1f}** oranında hafif olumsuz ifade içeriyor.")
                                    else:
                                        st.success("Bu ses kaydı temiz görünüyor. Küfür tespit edilmedi.")

                                st.session_state["last_single_result"] = {
                                    "audio_path": audio_path,
                                    "uploaded_name": uploaded_file.name,
                                    "formatted_text": formatted_text,
                                    "detected_swears": detected_swears,
                                    "full_transcription": full_transcription,
                                    "segments_data": segments_data,
                                    "run_metrics": run_metrics,
                                    "toxicity_label": toxicity_label,
                                    "negative_score": negative_score,
                                }

                                # --- ANAHTAR KELİME & KONU TESPİTİ ---
                                keyword_result = None
                                if KEYWORD_DETECTION_ENABLED:
                                    with st.spinner("Anahtar kelime ve konu analizi yapılıyor..."):
                                        keyword_result = run_keyword_analysis(
                                            segments_data,
                                            full_transcription,
                                            sector=domain_key,
                                            audio_path=audio_path,
                                            uploaded_name=uploaded_file.name,
                                            asr_confidence=run_metrics.get("confidence", 0.0),
                                            quality_gate_passed=run_metrics.get("quality_gate_met", True),
                                        )
                                        st.session_state["last_single_result"]["keyword_result"] = keyword_result
                                    if keyword_result and not keyword_result.get("error"):
                                        st.markdown("---")
                                        st.markdown("### 🏷️ Tespit Edilen Konular")
                                        topics_list = keyword_result.get("topics", [])
                                        if topics_list:
                                            for t in topics_list:
                                                st.markdown(f"- {t}")
                                            
                                            st.markdown("### ⚠️ Trend & Erken Uyarı Tespiti")
                                            current_trend_data = get_trend_data(14)
                                            anomalies = detect_anomalies(current_trend_data)
                                            anomaly_topics = {a.topic: a for a in anomalies}
                                            
                                            found_anomaly = False
                                            for topic in topics_list:
                                                if topic in anomaly_topics:
                                                    a = anomaly_topics[topic]
                                                    found_anomaly = True
                                                    if a.severity == "CRITICAL":
                                                        st.error(f"🚨 **DİKKAT! '{topic}'** konusu bu görüşmede tespit edildi. Sistemde bu konu son günlerde **%{a.increase_percentage} artış** ile KRİZ seviyesinde!")
                                                    else:
                                                        st.warning(f"⚠️ **Uyarı! '{topic}'** konusu bu görüşmede tespit edildi. Sistemde bu konu **%{a.increase_percentage} artış** ile YÜKSELİŞTE!")
                                            
                                            if not found_anomaly:
                                                st.success("✅ Bu görüşmedeki konular mevcut trend anomalileri (kriz) ile eşleşmedi. Rutin işlem.")
                                        else:
                                            st.info("Bu görüşmede bilinen bir kriz veya trend konusu tespit edilmedi.")
                                    elif keyword_result and keyword_result.get("error"):
                                        st.info(f"Anahtar kelime modülü: {keyword_result['error']}")

                                # --- DIARIZATION (BETA) ---
                                diarization_result = None
                                if diarization_mode and hf_token:
                                    with st.spinner("Konuşmacı ayrımı yapılıyor..."):
                                        diarization_result = diarize_audio(audio_path, hf_token)
                                        if isinstance(diarization_result, str) and ("Hatası" in diarization_result or "Eksik" in diarization_result): # Hata döndü
                                            st.warning(f"Diarization Hatası: {diarization_result}")
                                        else:
                                            st.success("Konuşmacı ayrımı tamamlandı.")
                                            with st.expander("Konuşmacı Ayrımı (A/B) Sonuçları", expanded=True):
                                                st.text(diarization_result)

                                # --- OTOMATİK ÖZETLEME (CRM AUTO-NOTE) ---
                                if enable_autonote:
                                    st.markdown("---")
                                    st.markdown("### 📝 CRM Kapanış Notu (Auto-Note)")
                                    
                                    with st.spinner("Çağrı özetleniyor..."):
                                        if ollama_model != "Kapalı (Sadece Yerel Motor)":
                                            crm_summary = generate_ollama_summary(full_transcription, ollama_model, nlp_classifier if enable_nlp else None)
                                            badge = f"🔒 <b>%100 Gizli Lokal LLM ({ollama_model})</b>"
                                        else:
                                            crm_summary = generate_crm_summary(full_transcription, nlp_classifier if enable_nlp else None)
                                            badge = "🤖 <b>Yerel AI ile Özetlendi</b>"
                                        
                                    # Şık bir CRM bileti UI'ı
                                    st.markdown(f"""
                                    <div style="background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                                        <h4 style="margin-top: 0; border-bottom: 1px solid #444; padding-bottom: 5px; color: #f0f2f6;">Yapay Zeka Çağrı Özeti <span style="float:right; font-size: 0.6em; color: #00ffaa; margin-top:5px;">{badge}</span></h4>
                                        <ul style="list-style-type: none; padding-left: 0; color: #d0d0d0; line-height: 1.8;">
                                            <li>👤 <b>Müşteri:</b> {crm_summary.intent}</li>
                                            <li>⚠️ <b>Sorun:</b> {crm_summary.issue}</li>
                                            <li>⚙️ <b>İşlem:</b> {crm_summary.action}</li>
                                            <li>✅ <b>Sonuç:</b> {crm_summary.resolution}</li>
                                        </ul>
                                        <div style="background-color: #2b2b2b; padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 3px solid #007aff;">
                                            <p style="margin: 0; font-size: 0.9em; color: #a0a0a0;"><b>Yönetici Özeti:</b><br/><i>"{crm_summary.executive_summary}"</i></p>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Kopyalama metni (Pano için)
                                    copy_text = f"Müşteri: {crm_summary.intent}\\nSorun: {crm_summary.issue}\\nİşlem: {crm_summary.action}\\nSonuç: {crm_summary.resolution}\\n\\nÖzet: {crm_summary.executive_summary}"
                                    st.code(copy_text, language="text")
                                    st.caption("Üstteki kutunun sağ köşesindeki kopyalama ikonuna tıklayarak CRM ekranınıza yapıştırabilirsiniz.")

                                # --- GÖRSELLEŞTİRME ---
                                st.markdown("---")
                            
                                # KELİME BULUTU
                                if enable_wordcloud:
                                    with st.expander("Kavram Haritası (Word Cloud)", expanded=True):
                                         wc_fig = create_wordcloud(full_transcription)
                                         if wc_fig:
                                             st.pyplot(wc_fig)

                                # --- ARAMA MOTORU (YENİ) ---
                                st.markdown("#### Ses İçi Arama")
                                search_query = st.text_input("Ses kaydı içinde kelime ara...", placeholder="Örn: bütçe, toplantı, rakam...")
                            
                                if search_query:
                                    found_count = 0
                                    st.write(f"**'{search_query}' için sonuçlar:**")
                                    for segment in segments_data:
                                        if search_query.lower() in segment.text.lower():
                                            found_count += 1
                                            # Zamanı formatla
                                            m, s = divmod(segment.start, 60)
                                            time_fmt = f"{int(m):02d}:{s:04.1f}"
                                            st.markdown(f"**{time_fmt}**: ...{segment.text.strip()}...")
                                
                                    if found_count == 0:
                                        st.info("Kelime bulunamadı.")
                                    else:
                                        st.caption(f"Toplam {found_count} eşleşme bulundu.")
                            
                                st.markdown("---")

                                # --- TRANSKRİPSİYON GÖRÜNÜMÜ ---
                                st.markdown("#### Deşifre Çıktısı")

                                if keyword_result and keyword_result.get("hits") and KEYWORD_DETECTION_ENABLED:
                                    st.markdown(
                                        highlight_transcript_html(segments_data, keyword_result["hits"]),
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    html_output = '<div class="transcript-container">'
                                    for segment in segments_data:
                                        m, s = divmod(segment.start, 60)
                                        time_formatted = f"{int(m):02d}:{s:04.1f}"
                                        text_content = html.escape(segment.text.strip())
                                        
                                        # Apply search highlight if there is an active search
                                        if search_query and search_query.lower() in segment.text.lower():
                                            pattern = re.compile(re.escape(search_query), re.IGNORECASE)
                                            text_content = pattern.sub(lambda m: f'<span class="search-highlight">{m.group(0)}</span>', text_content)
                                            
                                        html_output += f"""
                                        <div class="transcript-row">
                                            <div class="transcript-time">{time_formatted}</div>
                                            <div class="transcript-text">{text_content}</div>
                                        </div>"""
                                    html_output += '</div>'
                                    
                                    st.markdown(html_output, unsafe_allow_html=True)

                                # --- İNDİRME SEÇENEKLERİ ---
                                col_d1, col_d2, col_d3 = st.columns(3)
                                with col_d1:
                                    st.download_button(
                                        label="📥 RAPOR (TXT)",
                                        data=formatted_text,
                                        file_name=f"ASR_Rapor_{os.path.splitext(uploaded_file.name)[0]}.txt",
                                        mime="text/plain"
                                    )
                                with col_d2:
                                    try:
                                        srt_content = create_srt(segments_data)
                                        st.download_button(
                                            label="🎬 ALTYAZI (SRT)",
                                            data=srt_content,
                                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.srt",
                                            mime="text/plain"
                                        )
                                    except Exception as report_error:
                                        st.warning(f"SRT hazırlanamadı: {report_error}")
                                with col_d3:
                                    tox_info = f"{toxicity_label} (%{negative_score*100:.1f})" if 'toxicity_label' in locals() else "Analiz Edilmedi"
                                    try:
                                        pdf_bytes = create_pdf_report(uploaded_file.name, formatted_text, detected_swears, tox_info)
                                        st.download_button(
                                            label="📑 RAPOR (PDF)",
                                            data=pdf_bytes,
                                            file_name=f"ASR_Rapor_{os.path.splitext(uploaded_file.name)[0]}.pdf",
                                            mime="application/pdf"
                                        )
                                    except Exception as report_error:
                                        st.warning(f"PDF hazırlanamadı: {report_error}")

                            except Exception as e:
                                st.error(f"Sistem hatası: {e}")
                                st.exception(e)
                    else:
                        cached_result = st.session_state.get("last_single_result")
                        if cached_result and cached_result.get("audio_path") == audio_path:
                            render_cached_analysis_result(cached_result, reference_text, target_latency_s, enable_wordcloud)
                else:
                     output_placeholder.markdown(
                        """
                        <div class="output-frame waiting" class="empty-state">
                            <div>
                                <div class="output-kicker" style="color:var(--asr-accent); font-weight:800; text-transform:uppercase; font-size:0.8rem; margin-bottom:0.5rem;">Çıktı bekleniyor</div>
                                <div class="output-title" style="color:var(--asr-text); font-size:1.1rem; font-weight:600; margin-bottom:0.5rem;">Önce bir ses dosyası yükleyin.</div>
                                <div class="output-copy" style="color:var(--asr-muted); font-size:0.9rem;">Dosya yüklendiğinde analiz butonu açılır; sonuçlar bu bölümde oluşur.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                     )
                


        # --------------------------
        # --- CANLI DİNLEME (MİKROFON) ---
        # --------------------------
        elif mode == "Canlı Dinleme (Mikrofon)":
            render_panel(
                "Canlı Ses Analizi",
                "Mikrofondan alınan kısa kayıtlar tek dosya akışıyla aynı kalite kapısı, sektör sözlüğü ve raporlama mantığından geçer.",
                "Mikrofon"
            )
        
            # st.audio_input (Streamlit 1.39+)
            audio_value = st.audio_input("Mikrofonu etkinleştirmek için tıklayın")
        
            if audio_value:
                st.audio(audio_value)
            
                # Geçici kaydet
                mic_path = os.path.join(TEMP_AUDIO_DIR, "microphone_input.wav")
                with open(mic_path, "wb") as f:
                    f.write(audio_value.read())
                
                if st.button("Kaydı Analiz Et", type="primary"):
                     with st.spinner("Canlı kayıt işleniyor..."):
                        try:
                            model = load_whisper_model(model_size, st.session_state.get("hardware_engine", "Windows"))
                            task_type = "translate" if translate_mode else "transcribe"
                            formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = transcribe_audio_file(
                                model,
                                mic_path,
                                LANGUAGE,
                                TURKISH_SWEAR_WORDS,
                                task=task_type,
                                profile_key=profile_key,
                                domain_key=domain_key,
                                hotwords=hotwords_input,
                                target_latency_s=target_latency_s,
                            )
                        
                            st.success("Tüm kuyruk tamamlandı.")


                            st.caption(f"İşlem: {run_metrics['elapsed_s']:.1f}s • ASR Güveni: %{run_metrics['confidence']:.1f}")
                            st.text_area("Sonuç:", full_transcription, height=200)
                            render_reference_gate_hint(reference_text)
                            if reference_text.strip():
                                wer_stats = calculate_word_accuracy(reference_text, full_transcription)
                                render_quality_metric_cards(
                                    [
                                        ("Kelime Doğruluğu", f"%{wer_stats['accuracy']:.1f}", f"WER %{wer_stats['wer'] * 100:.1f}"),
                                    ]
                                )
                        
                            if enable_wordcloud:
                                wc_fig = create_wordcloud(full_transcription)
                                if wc_fig:
                                    st.pyplot(wc_fig)
                            
                        except Exception as e:
                            st.error(f"Hata: {e}")



        # --------------------------
        # --- TOPLU KLASÖR İŞLEME ---
        # --------------------------
        elif mode == "Toplu İşlem Merkezi":
        
            render_panel(
                "Toplu İşlem Merkezi",
                f"Kaynak klasör: {BATCH_DIR}. Bu mod, klasördeki tüm sesleri sırayla işler ve tek bir denetim raporu üretir.",
                "Batch"
            )

            st.markdown("### 📥 Dosyaları Yükleyin")
            uploaded_batch = st.file_uploader(
                "Toplu analiz için birden fazla ses dosyası seçin", 
                type=["mp3", "wav", "m4a", "flac"], 
                accept_multiple_files=True
            )
            
            audio_files = []
            if uploaded_batch:
                for uf in uploaded_batch:
                    file_path = os.path.join(BATCH_DIR, uf.name)
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())
                        
            audio_files = [os.path.join(BATCH_DIR, f) for f in os.listdir(BATCH_DIR) if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]

            if not audio_files:
                st.warning("Klasörde veya yüklenenler arasında ses dosyası bulunamadı.")
            else:
                st.success(f"{len(audio_files)} dosya kuyrukta bekliyor.")
            
                with st.expander("Dosya Listesini Gör"):
                    st.write([os.path.basename(f) for f in audio_files])

                OUTPUT_BATCH_FILENAME = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

                if st.button("Toplu Analizi Başlat", type="primary"):
                    with st.spinner("YZ modelleri hazırlanıyor..."):
                        model = load_whisper_model(model_size, st.session_state.get("hardware_engine", "Windows"))
                        nlp_classifier = load_toxicity_classifier() if enable_nlp else None
                    total_detected_swears = 0
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                    with open(OUTPUT_BATCH_FILENAME, "w", encoding="utf-8") as f:
                        f.write(f"--- ASR PRO RAPORU ---\nTarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                        for i, file_path in enumerate(audio_files):
                            file_name = os.path.basename(file_path)
                            status_text.text(f"İşleniyor ({i+1}/{len(audio_files)}): {file_name}")
                        
                            try:
                                formatted_text, detected_swears, full_transcription, segments_data, info, run_metrics = transcribe_audio_file(
                                    model,
                                    file_path,
                                    LANGUAGE,
                                    TURKISH_SWEAR_WORDS,
                                    profile_key=profile_key,
                                    domain_key=domain_key,
                                    hotwords=hotwords_input,
                                    target_latency_s=target_latency_s,
                                )
                                # Toplu sonuçlara ekleme
                                total_detected_swears += len(detected_swears)
                                toxicity_label, negative_score = analyze_toxicity(full_transcription, nlp_classifier) if enable_nlp else ("Atlandı", 0.0)
                            
                                f.write(f"\n[{file_name}]\n")
                                f.write(
                                    f"İşlem Süresi: {run_metrics['elapsed_s']:.1f}s | "
                                    f"ASR Güveni: %{run_metrics['confidence']:.1f} | "
                                    f"Ses Kalitesi: %{run_metrics.get('audio_quality_score', 0):.0f} | "
                                    f"Kalite Kapısı: {'Geçti' if run_metrics.get('quality_gate_met') else 'İnceleme'} | "
                                    f"Filtre: {run_metrics['filtered_segments']}\n"
                                )
                                if run_metrics.get("quality_retry"):
                                    f.write(f"Otomatik Yeniden Deneme: {', '.join(run_metrics.get('retry_profiles', []))}\n")
                                if reference_text.strip():
                                    wer_stats = calculate_word_accuracy(reference_text, full_transcription)
                                    f.write(f"Kelime Doğruluğu: %{wer_stats['accuracy']:.1f} | WER: %{wer_stats['wer'] * 100:.1f}\n")
                                f.write(f"Toksisite: {toxicity_label} (%{negative_score*100:.1f})\n")
                                if detected_swears:
                                    f.write(f"Uygunsuz İfadeler: {len(detected_swears)} adet\n")
                                f.write("-" * 30 + "\n")
                                f.write(formatted_text + "\n")
                            
                            except Exception as e:
                                f.write(f"[{file_name}] HATA: {e}\n")
                        
                            progress_bar.progress((i + 1) / len(audio_files))

                    status_text.text("Tamamlandı!")
                    st.success(f"Tüm işlemler bitti. Toplam {total_detected_swears} uygunsuz ifade yakalandı.")
                
                    with open(OUTPUT_BATCH_FILENAME, "r", encoding="utf-8") as f:
                        st.download_button(
                            label="⬇️ Toplu Raporu İndir",
                            data=f.read(),
                            file_name=OUTPUT_BATCH_FILENAME,
                            mime="text/plain"
                        )

        # --------------------------
        # --- TREND VE ERKEN UYARI RADARI ---
        # --------------------------
        elif mode == "📈 Trend ve Erken Uyarı Radarı":
            from asr_pro.core.trend_engine import forecast_tomorrow
            render_panel(
                "Müşteri Yolculuğu ve Erken Uyarı Radarı",
                "Yapay zeka, geçmiş binlerce çağrıyı tarayarak ürün ekipleri için kritik anomali ve trendleri yakalar.",
                "Radar"
            )
            
            st.markdown("### 📥 Canlı Veri / Simülasyon Besleme")
            st.info("Trendleri anlık test etmek için buraya çoklu ses dosyası yükleyebilirsiniz. Dosyalar anında deşifre edilip konuları Trend radarına yansıtılacaktır.")
            uploaded_trend = st.file_uploader("Trendi test etmek için ses dosyaları yükleyin", type=["mp3", "wav", "m4a", "flac"], accept_multiple_files=True, key="trend_uploader")
            
            if uploaded_trend:
                if st.button("Yüklenenleri Trende İşle", type="primary"):
                    with st.spinner("Dosyalar işleniyor ve Trend radarına aktarılıyor..."):
                        model = load_whisper_model(model_size, st.session_state.get("hardware_engine", "Windows"))
                        for uf in uploaded_trend:
                            file_path = os.path.join(TEMP_AUDIO_DIR, uf.name)
                            with open(file_path, "wb") as f:
                                f.write(uf.getbuffer())
                            
                            try:
                                _, _, full_transcription, _, _, _ = transcribe_audio_file(
                                    model, file_path, LANGUAGE, TURKISH_SWEAR_WORDS,
                                    profile_key=profile_key, domain_key=domain_key
                                )
                                run_keyword_analysis([], full_transcription)
                            except Exception as e:
                                st.error(f"{uf.name} işlenirken hata: {e}")
                                
                        st.success("Tüm dosyalar işlendi ve Trend radarına yansıtıldı!")
                        time.sleep(1)
                        st.rerun()
                        
            st.markdown("### 🚨 Ürün Ekipleri İçin Erken Uyarı (Early Warning) Sistemi")
            
            with st.spinner("Geçmiş 14 günün veritabanı analiz ediliyor..."):
                trend_data = get_trend_data(days=14)
                alerts = detect_anomalies(trend_data)
                forecasts = forecast_tomorrow(trend_data)
                
            if alerts:
                for alert in alerts:
                    if alert.severity == "CRITICAL":
                        st.error(f"**{alert.severity} ALARM:** `{alert.topic}` şikayetlerinde son günlerde **%{alert.increase_percentage}** artış var! (Normal Seviye: {alert.baseline_avg}/gün ➡️ Şu an: {alert.recent_count}/gün)")
                    else:
                        st.warning(f"**{alert.severity} UYARI:** `{alert.topic}` şikayetleri yükseliş trendinde (**%{alert.increase_percentage}**).")
            else:
                st.success("✅ Son 48 saatte herhangi bir şikayet anomalisi / spike tespit edilmedi.")
                
            if forecasts:
                st.markdown("### 🔮 Yapay Zeka Gelecek Tahmini (Forecasting)")
                cols = st.columns(min(3, len(forecasts)))
                for i, fcast in enumerate(forecasts):
                    with cols[i % 3]:
                        color = "#ff4b4b" if fcast.confidence_level == "HIGH" else "#ffb200"
                        st.markdown(f"""
                        <div style="background-color: rgba(255, 0, 0, 0.05); border-left: 5px solid {color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                            <h4 style="margin:0; color: {color};">Yarınki Beklenti</h4>
                            <p style="margin:5px 0 0 0; color: white;"><b>{fcast.topic}</b></p>
                            <h2 style="margin:5px 0 0 0; color: {color};">{fcast.predicted_volume} Çağrı</h2>
                            <p style="margin:5px 0 0 0; color: gray; font-size: 0.8em;">İvme Hızı: {fcast.trend_slope} / Gün ({fcast.confidence_level})</p>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📈 Son 14 Günlük Şikayet / Konu Hacmi")
            
            import pandas as pd
            if trend_data:
                df = pd.DataFrame.from_dict(trend_data, orient='index')
                if not df.empty and not df.columns.empty and df.sum().sum() > 0:
                    st.line_chart(df, height=400, use_container_width=True)
                else:
                    st.info("Henüz görüntülenecek trend verisi bulunmuyor.")
                
                with st.expander("Ham Veriyi Görüntüle"):
                    st.dataframe(df)
