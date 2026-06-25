import os
import re
import ssl
import warnings
from datetime import datetime

import streamlit as st
import whisper
from pydub import AudioSegment
from transformers import pipeline

# --- YAPILANDIRMA ---
MODEL_SIZE = "medium"  # Doğruluk için 'medium' önerilir.
LANGUAGE = "tr"
TEMP_AUDIO_DIR = "temp_audio_uploads"
BATCH_DIR = "batch_audio_files"
# NLP için Türkçe Toksisite/Duygu Analizi Modeli (Büyük boyutlu olabilir!)
TOXICITY_CLASSIFIER_MODEL = "savasy/bert-base-turkish-sentiment-cased"
# --------------------

# --- KÜFÜR LİSTESİ (Genişletmeyi unutmayın!) ---
TURKISH_SWEAR_WORDS = [
    "lan", "aptal", "salak", "gerizekalı", "şerefsiz", "piç", "siktir","hassiktir", "pezevenk", "orospu", "mal", "dangalak", "ayı", "eşek","amına","koyayım","ahlaksız",
    "hanzo","bacaksız", "çapsız", "düzenbaz","zibidi", "meymenetsiz", "uyuz", "lavuk", "dallama", "nankör",
    "açgözlü", "basitsin", "korkak", "serseri", "cahil", "gavur", "hain", "kukla", "soytarı", "şebek", "yalancı",
    "şerefsiz", "haysiyetsiz", "namussuz","orospu","pezevenk","ibne","gavat","dümbük","deyyus","dallama","yavşak","Gerizekalı","aptal","salak","beyinsiz","embesil",
    "ahmak","beyinli","bunak","Hırsız","dolandırıcı","sahtekar","rüşvetçi","tecavüzcü","sapık","ayyaş","hapçı","haini","terörist","şarlatan","soytarı","müfteri",
    "üçkağıtçı","şaklaban","Kaşar","sürtük","eskort","pavyon","boynuzlu","Köpek","it","eşek","domuz","hayvan","çakal","sırtlan","akbaba","kene","kuduz",
    "maymun","gergedan","dana","şaşı","kel","kambur","cüce","şişko","dombili","Münafık","dinsiz","imansız","godoş","mason","ataput","soytarısı",
    "deccal","nemrut","pis gavur","yobaz","pis","sikerim","sikcek"
]
# ------------------------------------------------

# PyTorch ve Whisper'dan gelebilecek uyarıları gizle
warnings.simplefilter("ignore")

# Gerekli klasörleri oluştur
if not os.path.exists(TEMP_AUDIO_DIR):
    os.makedirs(TEMP_AUDIO_DIR)
if not os.path.exists(BATCH_DIR):
    os.makedirs(BATCH_DIR)

# --- MODEL YÜKLEME (CACHE) ---
@st.cache_resource
def load_whisper_model(model_name: str):
    """Whisper modelini yükler (SSL sorununu çözerek)."""
    try:
        # SSL ÇÖZÜMÜ
        ssl._create_default_https_context = ssl._create_unverified_context
        model = whisper.load_model(model_name)
        st.success(f"✅ Whisper '{model_name}' modeli başarıyla yüklendi.")
        return model
    except Exception as e:
        st.error(f"❌ Whisper modeli yüklenirken hata: {e}")
        st.stop()

@st.cache_resource
def load_toxicity_classifier():
    """NLP sınıflandırıcısını yükler (Sadece bir kere)."""
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model=TOXICITY_CLASSIFIER_MODEL,
            tokenizer=TOXICITY_CLASSIFIER_MODEL,
            return_all_scores=True
        )
        st.sidebar.success("✅ NLP Sınıflandırıcısı yüklendi.")
        return classifier
    except Exception as e:
        st.sidebar.warning(f"❌ NLP Sınıflandırıcısı yüklenemedi: {e}")
        return None

# Modelleri yükle
model = load_whisper_model(MODEL_SIZE)
nlp_classifier = load_toxicity_classifier()

# --- ZAMAN DAMGALI TRANSKRİPSİYON & KÜFÜR TESPİTİ FONKSİYONU ---
def transcribe_audio_file(model, file_path: str, lang: str, swear_list: list) -> tuple:
    """Ses dosyasını tanır ve sonuçları döndürür."""

    result = model.transcribe(
        file_path,
        language=lang,
        word_timestamps=True,
        # Ses kalitesi için deneme parametreleri
        no_speech_threshold=0.35,
        logprob_threshold=-0.9
    )

    formatted_text = ""
    detected_swears = []

    # 1. Segment Bazlı Çıktı Oluşturma
    formatted_text += "--- 1. Segment Bazlı Çıktı ---\n"
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        formatted_text += f"[{start_time:.2f}s - {end_time:.2f}s]: {text.strip()}\n"

    formatted_text += "\n--- 2. Kelime Bazlı Çıktı ---\n"
    word_list = []
    for segment in result["segments"]:
        if "words" in segment:
            word_list.extend(segment["words"])

    current_line = ""
    LINE_LENGTH_LIMIT = 80

    for word_data in word_list:
        start_time = word_data["start"]
        word = word_data["word"].strip()

        # Sadece harf ve sayıları bırakarak kelimeyi temizle
        clean_word = re.sub(r'[^\w]', '', word.lower())

        # *** KÜFÜR TESPİTİ BÖLÜMÜ ***
        if clean_word in swear_list:
            detected_swears.append({
                "word": word,
                "time": start_time
            })
            # Tespit edilen kelimeyi çıktıda **vurgula**
            word_with_timestamp = f"({start_time:.2f}s) **{word}** "
        else:
            word_with_timestamp = f"({start_time:.2f}s){word} "

        # Çıktı formatlama
        if len(current_line) + len(word_with_timestamp) > LINE_LENGTH_LIMIT:
            formatted_text += current_line.strip() + "\n"
            current_line = ""

        current_line += word_with_timestamp

    formatted_text += current_line.strip()

    # NLP analizi için tüm metin
    full_transcription = result["text"]

    return formatted_text, detected_swears, full_transcription

# --- NLP TOKSİSİTE ANALİZ FONKSİYONU ---
def analyze_toxicity(text: str, classifier):
    """Metnin saldırganlık/toksisite skorunu hesaplar."""
    if not classifier or not text.strip():
        return "Analiz Yapılamadı", 0.0

    # Metni model için uygun hale getir (Çok uzun metinler için kesme)
    max_len = 512
    if len(text) > max_len * 2:
        input_text = text[:max_len] + " [SEP] " + text[-max_len:]
    else:
        input_text = text

    try:
        results = classifier(input_text)
        negative_score = 0
        positive_score = 0

        for item in results[0]:
            if 'negative' in item['label'].lower():
                negative_score = item['score']
            elif 'positive' in item['label'].lower():
                positive_score = item['score']

        if negative_score > 0.7 and negative_score > positive_score:
            toxicity_label = "Yüksek Negatif/Toksik"
        elif negative_score > 0.5:
            toxicity_label = "Orta Negatif"
        else:
            toxicity_label = "Düşük Negatif/Nötr"

        return toxicity_label, negative_score

    except Exception:
        return "NLP Hata", 0.0

# --- YARDIMCI GÖRÜNTÜLEME FONKSİYONU ---
def display_detection_results(detected_swears, col_display):
    if detected_swears:
        col_display.error(f"🚨 **UYARI: {len(detected_swears)} TANE UYGUNSUZ İFADE TESPİT EDİLDİ!**")
        swear_report = [f" - **'{item['word']}'** anı: {item['time']:.2f}s" for item in detected_swears]
        col_display.markdown("\n".join(swear_report))
    else:
        col_display.success("🎉 Dosyada uygunsuz ifade tespit edilmemiştir.")

# --- STREAMLIT ARAYÜZ ---
st.set_page_config(layout="wide")
st.title("🎙️ Lokal Whisper ASR - Gelişmiş Analiz Aracı")
st.markdown("**Gizlilik Uyarısı:** Tüm işlemler yerel sunucunuzda gerçekleşmektedir.")

mode = st.sidebar.radio(
    "İşlem Modunu Seçin:",
    ("Tekli Dosya Yükleme", "Toplu Klasör İşleme (Batch)")
)

# --------------------------
# --- TEKLİ DOSYA İŞLEME ---
# --------------------------
if mode == "Tekli Dosya Yükleme":

    st.header("1. Tek Dosya İşleme")
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Bir ses dosyası seçin (.mp3, .wav, vb.)", type=["mp3", "wav", "m4a", "flac"])
        audio_path = None
        if uploaded_file is not None:
            audio_path = os.path.join(TEMP_AUDIO_DIR, uploaded_file.name)
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"'{uploaded_file.name}' yüklendi.")
            st.audio(uploaded_file, format=uploaded_file.type, start_time=0)
            try:
                audio = AudioSegment.from_file(audio_path)
                duration_seconds = len(audio) / 1000
                st.info(f"Süre: {int(duration_seconds // 60)} dk {int(duration_seconds % 60)} sn")
            except Exception:
                st.warning("Ses bilgileri alınamadı.")

    with col2:
        st.header("2. Sonuçlar (Zaman Damgası, Küfür & NLP)")
        swear_display_placeholder = st.empty()
        nlp_display_placeholder = st.empty()

        if audio_path is not None:
            if st.button("Tek Dosya Tanımayı Başlat"):
                with st.spinner("⏳ Ses dosyası tanınıyor..."):
                    try:
                        formatted_text, detected_swears, full_transcription = transcribe_audio_file(model, audio_path, LANGUAGE, TURKISH_SWEAR_WORDS)

                        # Küfür Tespiti Sonucu
                        display_detection_results(detected_swears, swear_display_placeholder)

                        # NLP Analizi Sonucu
                        toxicity_label, negative_score = analyze_toxicity(full_transcription, nlp_classifier)
                        if toxicity_label != "Analiz Yapılamadı" and toxicity_label != "NLP Hata":
                            if "Negatif" in toxicity_label:
                                nlp_display_placeholder.error(f"**NLP Analizi:** {toxicity_label} | Negatif Skor: %{negative_score * 100:.2f}")
                            else:
                                nlp_display_placeholder.success(f"**NLP Analizi:** {toxicity_label} | Negatif Skor: %{negative_score * 100:.2f}")

                        st.success("✅ Tanıma Tamamlandı! (Metinde vurgulandı)")
                        st.markdown("**Zaman Damgalı Sonuçlar:**")
                        st.code(formatted_text, language='markdown')

                        st.download_button(
                            label="📄 Metni İndir (.txt)",
                            data=formatted_text,
                            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_timestamped_analiz.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"❌ Tanıma sırasında bir hata oluştu: {e}")
        else:
            swear_display_placeholder.info("Lütfen soldan bir ses dosyası yükleyin.")


# --------------------------
# --- TOPLU KLASÖR İŞLEME ---
# --------------------------
elif mode == "Toplu Klasör İşleme (Batch)":

    st.header("1. Toplu İşleme")
    st.warning(f"⚠️ **DİKKAT:** Dosyalarınız burada olmalıdır: `{BATCH_DIR}`")

    audio_files = [os.path.join(BATCH_DIR, f) for f in os.listdir(BATCH_DIR) if f.endswith(('.mp3', '.wav', '.m4a', '.flac'))]

    if not audio_files:
        st.warning(f"'{BATCH_DIR}' klasöründe işlenecek ses dosyası bulunamadı.")
    else:
        st.success(f"📂 İşlenmeye hazır **{len(audio_files)}** dosya bulundu.")
        st.text(", ".join([os.path.basename(f) for f in audio_files]))

        OUTPUT_BATCH_FILENAME = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        if st.button("Toplu İşlemi Başlat (Tüm Analizlerle)"):
            if not audio_files:
                 st.error("İşlenecek dosya yok.")
            else:
                total_detected_swears = 0
                with st.spinner("🚀 Tüm dosyalar arka planda işleniyor..."):
                    try:
                        with open(OUTPUT_BATCH_FILENAME, "w", encoding="utf-8") as f:
                            f.write(f"--- Toplu Whisper ASR ve Gelişmiş Analiz Sonuçları ---\nModel: {MODEL_SIZE}, NLP Model: {TOXICITY_CLASSIFIER_MODEL}\nBaşlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                            progress_bar = st.progress(0)

                            for i, file_path in enumerate(audio_files):
                                file_name = os.path.basename(file_path)
                                st.info(f"({i+1}/{len(audio_files)}) İşleniyor: {file_name}...")

                                formatted_text, detected_swears, full_transcription = transcribe_audio_file(model, file_path, LANGUAGE, TURKISH_SWEAR_WORDS)
                                total_detected_swears += len(detected_swears)

                                # NLP Analizi
                                toxicity_label, negative_score = analyze_toxicity(full_transcription, nlp_classifier)

                                # Dosyaya Yazma Başlıkları
                                f.write("\n========================================\n")
                                f.write(f"### DOSYA: {file_name} ###\n")
                                f.write("========================================\n")

                                # Küfür Raporu
                                if detected_swears:
                                    f.write(f"🚨 TESPİT EDİLEN UYGUNSUZ İFADE SAYISI: {len(detected_swears)}\n")
                                    for item in detected_swears:
                                        f.write(f"  - '{item['word']}' anı: {item['time']:.2f}s\n")
                                else:
                                    f.write("🎉 Uygunsuz ifade tespit edilmedi.\n")

                                # NLP Raporu
                                f.write("\n--- NLP Toksisite Analizi ---\n")
                                if toxicity_label != "Analiz Yapılamadı" and toxicity_label != "NLP Hata":
                                    f.write(f"Sınıflandırma: {toxicity_label}\n")
                                    f.write(f"Negatif Skor: %{negative_score * 100:.2f}\n")
                                    if "Negatif" in toxicity_label:
                                        f.write("*** YÜKSEK SALDIRGANLIK/NEGATİFLİK TESPİT EDİLDİ ***\n")
                                else:
                                     f.write("NLP Analizi yapılamadı (Model Yükleme Hatası).\n")

                                f.write("\n--- Zaman Damgalı Metin ---\n")
                                f.write(formatted_text + "\n")

                                progress = (i + 1) / len(audio_files)
                                progress_bar.progress(progress)

                        st.success(f"✅ Toplu işlem tamamlandı! Toplam {total_detected_swears} uygunsuz ifade bulundu.")

                        with open(OUTPUT_BATCH_FILENAME, encoding="utf-8") as f:
                            batch_data = f.read()
                            st.download_button(
                                label="⬇️ Toplu Sonuçları İndir (Tüm Analizler)",
                                data=batch_data,
                                file_name=OUTPUT_BATCH_FILENAME,
                                mime="text/plain"
                            )

                    except Exception as e:
                        st.error(f"❌ Toplu işleme sırasında kritik bir hata oluştu: {e}")
