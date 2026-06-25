import streamlit as st

# ============================================
# PERFORMANS OPTİMİZASYONU: Page config ilk sırada
# ============================================
st.set_page_config(
    page_title="ASR Analiz Paneli",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎙️",
)

# --- TEMEL IMPORTLAR (Hafif) ---
import os
import sys
import warnings
from pathlib import Path

# Anahtar kelime & konu tespiti entegrasyonu
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from asr_pro.core.churn_engine import analyze_churn_risk
    from asr_pro.core.compliance_engine import analyze_compliance_risk
    from asr_pro.core.empathy_engine import analyze_soft_skills
    from asr_pro.core.keyword_engine import SegmentInput
    from asr_pro.core.sentiment_engine import analyze_sentiment
    from asr_pro.core.summary_engine import generate_crm_summary, generate_ollama_summary
    from asr_pro.core.trend_engine import detect_anomalies, get_trend_data, log_call_trend

    # Lightweight Topic Extractor to replace the missing asr_bridge.py
    def run_keyword_analysis(segments_data, full_transcription, sector=None, audio_path=None, uploaded_name=None, asr_confidence=0.0, quality_gate_passed=True):
        topics = [
            "Mobil Uygulama Çökmesi", "Ödeme Ekranı Hatası", "Kargo Gecikmesi",
            "Şifre Yenileme", "Üyelik İptali", "Ürün İadesi", "Müşteri Hizmetleri Şikayeti",
            "Fatura İtirazı", "İnternet Bağlantı Sorunu"
        ]

        detected_topics = []
        text_lower = full_transcription.lower()

        if "uygulama" in text_lower and ("çök" in text_lower or "açılmıyor" in text_lower or "hata" in text_lower):
            detected_topics.append("Mobil Uygulama Çökmesi")
        if "kargo" in text_lower and ("gelmedi" in text_lower or "gecik" in text_lower or "nerede" in text_lower):
            detected_topics.append("Kargo Gecikmesi")
        if "ödeme" in text_lower or "kart" in text_lower or "para" in text_lower:
            detected_topics.append("Ödeme Ekranı Hatası")
        if "iptal" in text_lower or "kapatmak" in text_lower:
            detected_topics.append("Üyelik İptali")
        if "fatura" in text_lower or "ücret" in text_lower:
            detected_topics.append("Fatura İtirazı")

        # Log detected topics to the trend database directly!
        for t in detected_topics:
            try:
                log_call_trend(topic=t)
            except Exception:
                pass # Silently fail if DB is locked or not initialized yet

        # Return a mock format expected by render_keyword_results if needed
        return {
            "topics": detected_topics,
            "raw_text": full_transcription
        }

    KEYWORD_DETECTION_ENABLED = True
except ImportError:
    # Do not print error to console to prevent continuous Streamlit terminal spam
    pass
    KEYWORD_DETECTION_ENABLED = False

# --- LAZY IMPORT DEĞİŞKENLERİ ---
# Ağır kütüphaneler ihtiyaç duyulduğunda yüklenecek
_torch = None
_plt = None


# --- DEFERRED IMPORTS (Sayfa açılınca yüklenecek) ---
import shutil

# --- FFmpeg AYARLARI (SESSION STATE İLE OPTİMİZE) ---
if "ffmpeg_ready" not in st.session_state:
    import imageio_ffmpeg
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dst = os.path.join(os.getcwd(), "ffmpeg.exe")
    if not os.path.exists(ffmpeg_dst):
        try:
            shutil.copy2(ffmpeg_src, ffmpeg_dst)
        except Exception:
            pass
    os.environ["PATH"] += os.pathsep + os.getcwd()
    st.session_state["ffmpeg_path"] = ffmpeg_src
    st.session_state["ffmpeg_ready"] = True

# --- YAPILANDIRMA VE KÜFÜR LİSTESİ ---
from config import *

# ------------------------------------------------

# PyTorch ve Whisper'dan gelebilecek uyarıları gizle
warnings.simplefilter("ignore")

# Gerekli klasörleri oluştur
if not os.path.exists(TEMP_AUDIO_DIR):
    os.makedirs(TEMP_AUDIO_DIR)
if not os.path.exists(BATCH_DIR):
    os.makedirs(BATCH_DIR)

# --- MODEL YÜKLEME (CACHE + LAZY + MAXIMUM SPEED) ---
# Standart faster-whisper modelleri (yerel cache - hızlı yükleme)




from logic_handlers import *
from ui_components import *

render_app()

