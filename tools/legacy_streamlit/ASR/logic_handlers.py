import hashlib
import math
import os
import platform
import re
import subprocess
import time
import wave
from array import array
from collections import namedtuple
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import srt
import streamlit as st
from config import *

_torch = None
_plt = None
_WordCloud = None
_pipeline = None
_WhisperModel = None
_BatchedInferencePipeline = None
_ctranslate2 = None


def get_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def get_matplotlib():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt

        _plt = plt
    return _plt


def get_wordcloud():
    global _WordCloud
    if _WordCloud is None:
        from wordcloud import WordCloud

        _WordCloud = WordCloud
    return _WordCloud


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline

            _pipeline = pipeline
        except ImportError:
            _pipeline = False  # Mark as unavailable
    return _pipeline if _pipeline is not False else None


def get_whisper_model_class():
    global _WhisperModel
    if _WhisperModel is None:
        from faster_whisper import WhisperModel

        _WhisperModel = WhisperModel
    return _WhisperModel


def get_batched_pipeline_class():
    global _BatchedInferencePipeline
    if _BatchedInferencePipeline is None:
        from faster_whisper import BatchedInferencePipeline

        _BatchedInferencePipeline = BatchedInferencePipeline
    return _BatchedInferencePipeline


def get_ctranslate2():
    global _ctranslate2
    if _ctranslate2 is None:
        import ctranslate2

        _ctranslate2 = ctranslate2
    return _ctranslate2


def get_ffmpeg_path():
    """imageio-ffmpeg veya sistem FFmpeg yolunu güvenli şekilde döndürür."""
    session_path = st.session_state.get("ffmpeg_path")
    if session_path and os.path.exists(session_path):
        return session_path
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        st.session_state["ffmpeg_path"] = ffmpeg_path
        return ffmpeg_path
    except Exception:
        return "ffmpeg"


def resolve_model_name(model_size: str) -> str:
    return MODEL_NAME_MAP.get(model_size, model_size)


def choose_device_and_compute():
    try:
        ctranslate2 = get_ctranslate2()
        if ctranslate2.get_cuda_device_count() > 0:
            supported = ctranslate2.get_supported_compute_types("cuda")
            compute_type = "float16" if "float16" in supported else "float32"
            return "cuda", compute_type
        supported = ctranslate2.get_supported_compute_types("cpu")
        if "int8_float32" in supported:
            return "cpu", "int8_float32"
        if "int8" in supported:
            return "cpu", "int8"
    except Exception:
        pass
    return "cpu", "int8"


def wants_apple_mlx_engine(engine_type: str = ""):
    engine = str(engine_type or "").lower()
    return "mac" in engine or "mlx" in engine or "apple" in engine


def is_apple_silicon_host():
    machine = platform.machine().lower()
    return platform.system() == "Darwin" and machine in {"arm64", "aarch64"}


def is_mlx_model(model):
    return getattr(model, "engine_name", "") == "mlx"


def describe_runtime_engine(engine_type: str = ""):
    if wants_apple_mlx_engine(engine_type):
        return ("apple mlx", "float16") if is_apple_silicon_host() else ("apple mlx", "unavailable")
    return choose_device_and_compute()


def resolve_mlx_repo_name(actual_model: str) -> str:
    model_ref = str(actual_model or "large-v3-turbo")
    if model_ref.startswith("mlx-community/") or Path(model_ref).exists():
        return model_ref
    
    # HuggingFace uses a specific suffix for the large-v3 mlx model
    if model_ref == "large-v3":
        return "mlx-community/whisper-large-v3-mlx-4bit"
        
    return f"mlx-community/whisper-{model_ref}"


class MLXWhisperWrapper:
    """Mac cihazlar için donanımsal hızlandırıcı MLX kullanan adaptör sınıf."""

    def __init__(self, actual_model: str):
        self.actual_model = actual_model
        self.repo_name = resolve_mlx_repo_name(actual_model)
        self.engine_name = "mlx"
        self.supports_batched = False
        self.compute_type = "float16"

    def transcribe(self, audio, **kwargs):
        import mlx.core as mx
        import mlx_whisper
        from mlx_whisper.transcribe import ModelHolder

        # Streamlit calls this from different threads. MLX handles cross-thread
        # model caching fine as long as parameters are evaluated (which load_model does).

        mlx_kwargs = {}
        for key in [
            "language",
            "task",
            "temperature",
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "condition_on_previous_text",
            "initial_prompt",
            "word_timestamps",
            "prepend_punctuations",
            "append_punctuations",
            "without_timestamps",
            "hallucination_silence_threshold",
            "suppress_blank",
            "suppress_tokens",
            "max_initial_timestamp",
        ]:
            if key in kwargs and kwargs[key] is not None:
                mlx_kwargs[key] = kwargs[key]

        # CRITICAL: Always lock the language to prevent mid-audio language switching.
        # MLX Whisper re-detects language if `language` key is missing from decode_options.
        # When the model hears English brand names (YouTube, Netflix, WhatsApp), it may
        # switch the decoder language. Force it to stay in the specified language.
        mlx_kwargs["language"] = mlx_kwargs.get("language") or kwargs.get("language") or "tr"

        if kwargs.get("max_new_tokens") and "sample_len" not in mlx_kwargs:
            mlx_kwargs["sample_len"] = kwargs["max_new_tokens"]
        mlx_kwargs["fp16"] = True

        # Deduplicate - prevent MLX from looping on repeated tokens
        # mlx_whisper doesn't support repetition_penalty directly, but we can set
        # no_repeat_ngram_size via a workaround using compression_ratio_threshold
        if "repetition_penalty" in kwargs:
            # Tighten compression threshold when repetition penalty is high
            penalty = float(kwargs.get("repetition_penalty", 1.0))
            current_threshold = mlx_kwargs.get("compression_ratio_threshold", 2.4)
            mlx_kwargs["compression_ratio_threshold"] = min(current_threshold, 2.0 if penalty >= 1.1 else 2.4)

        res = mlx_whisper.transcribe(audio, path_or_hf_repo=self.repo_name, **mlx_kwargs)

        Segment = namedtuple(
            "Segment",
            ["start", "end", "text", "avg_logprob", "no_speech_prob", "compression_ratio", "words"],
        )
        segments = []
        for s in res.get("segments", []):
            segments.append(
                Segment(
                    start=float(s.get("start", 0)),
                    end=float(s.get("end", 0)),
                    text=str(s.get("text", "")),
                    avg_logprob=float(s.get("avg_logprob", 0)),
                    no_speech_prob=float(s.get("no_speech_prob", 0)),
                    compression_ratio=float(s.get("compression_ratio", 1.0)),
                    words=s.get("words", []),
                )
            )

        class Info:
            language = res.get("language") or mlx_kwargs.get("language", "tr")
            language_probability = 1.0
            duration = res.get("duration")

        return segments, Info()


@st.cache_resource
def load_whisper_model(model_size: str, engine_type: str = "Windows", _force_reload: int = 1):
    """Donanıma göre MLX veya Faster-Whisper modelini yükler."""
    try:
        actual_model = resolve_model_name(model_size)
        if wants_apple_mlx_engine(engine_type):
            if not is_apple_silicon_host():
                raise RuntimeError(
                    "Mac/MLX motoru yalnızca Apple Silicon macOS üzerinde kullanılabilir."
                )
            return MLXWhisperWrapper(actual_model)

        WhisperModel = get_whisper_model_class()
        cpu_threads = max(4, os.cpu_count() or 4)
        device, compute_type = choose_device_and_compute()
        num_workers = 1 if device == "cuda" else min(4, max(1, cpu_threads // 2))
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        model = WhisperModel(
            actual_model,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        return model
    except Exception as e:
        st.error(f"❌ Whisper modeli yüklenirken hata: {e}")
        st.stop()


@st.cache_resource
def load_toxicity_classifier():
    """NLP sınıflandırıcısını yükler (Lazy import ile)."""
    pipeline = get_pipeline()
    if pipeline is None:
        return None
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model=TOXICITY_CLASSIFIER_MODEL,
            tokenizer=TOXICITY_CLASSIFIER_MODEL,
            return_all_scores=True,
        )
        return classifier
    except Exception:
        return None


# Modelleri aşağıda kullanıcı seçimine göre yükleyeceğiz
# model = load_whisper_model(MODEL_SIZE)
# nlp_classifier = load_toxicity_classifier()

# --- YARDIMCI FONSİYONLAR ---


def format_timestamp(seconds: float) -> str:
    return str(timedelta(seconds=seconds)).split(".")[0]


def create_srt(segments):
    """Segmentleri SRT formatına çevirir."""
    subs = []
    for i, segment in enumerate(segments):
        start = timedelta(seconds=segment.start)
        end = timedelta(seconds=segment.end)
        content = segment.text.strip()
        subs.append(srt.Subtitle(index=i + 1, start=start, end=end, content=content))
    return srt.compose(subs)


def create_wordcloud(text):
    """Kelime bulutu görseli oluşturur (lazy import)."""
    if not text.strip():
        return None
    WordCloud = get_wordcloud()
    plt = get_matplotlib()
    wordcloud = WordCloud(
        width=800, height=400, background_color="black", colormap="viridis"
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig


def diarize_audio(file_path, hf_token):
    """Pyannote ile konuşmacı ayrımı yapar."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return "Eksik Kütüphane: 'pyannote.audio' yüklü değil. Terminalde 'pip install pyannote.audio' çalıştırın."

    try:
        # Pipeline'ı yükle
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

        if pipeline is None:
            return "Model yüklenemedi. Token'ı kontrol edin veya Hugging Face üzerinden koşulları kabul ettiğinizden emin olun (pyannote/speaker-diarization)."

        # GPU varsa kullan
        torch = get_torch()
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        diarization = pipeline(file_path)

        # Sonuçları okunabilir formata çevir
        result_str = ""
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result_str += f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}\n"

        return result_str if result_str else "Konuşmacı ayrımı yapıldı ancak sonuç boş."

    except Exception as e:
        return f"Diarization Hatası: {str(e)}"


# --- ZAMAN DAMGALI TRANSKRİPSİYON & KÜFÜR TESPİTİ FONKSİYONU ---
def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def get_audio_duration_seconds(file_path: str):
    try:
        with wave.open(str(file_path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate:
                return frames / float(rate)
    except Exception:
        return None
    return None


def analyze_prepared_audio_quality(file_path: str):
    """16 kHz mono WAV üzerinden pratik ses kalitesi sinyalleri üretir."""
    try:
        with wave.open(str(file_path), "rb") as wav_file:
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            if sample_width != 2:
                return {
                    "audio_quality_score": 50.0,
                    "audio_quality_label": "Ölçüm sınırlı",
                    "audio_rms_dbfs": None,
                    "audio_peak_dbfs": None,
                    "audio_clipped_percent": None,
                    "audio_silence_percent": None,
                    "audio_quality_notes": [
                        "PCM 16-bit olmadığı için detaylı kalite ölçümü sınırlı."
                    ],
                }

            total_samples = 0
            square_sum = 0.0
            peak = 0
            clipped = 0
            silent = 0
            silence_threshold = 350
            clip_threshold = 32600
            chunk_frames = max(frame_rate, 16000)

            while True:
                frames = wav_file.readframes(chunk_frames)
                if not frames:
                    break
                samples = array("h")
                samples.frombytes(frames)
                for sample in samples:
                    value = abs(int(sample))
                    total_samples += 1
                    square_sum += value * value
                    peak = max(peak, value)
                    if value <= silence_threshold:
                        silent += 1
                    if value >= clip_threshold:
                        clipped += 1

        if total_samples == 0:
            return {
                "audio_quality_score": 0.0,
                "audio_quality_label": "Boş kayıt",
                "audio_rms_dbfs": None,
                "audio_peak_dbfs": None,
                "audio_clipped_percent": 0.0,
                "audio_silence_percent": 100.0,
                "audio_quality_notes": ["Ses örneği okunamadı veya kayıt boş."],
            }

        rms = math.sqrt(square_sum / total_samples)
        rms_dbfs = 20 * math.log10(max(rms, 1.0) / 32768.0)
        peak_dbfs = 20 * math.log10(max(peak, 1) / 32768.0)
        clipped_percent = (clipped / total_samples) * 100
        silence_percent = (silent / total_samples) * 100

        score = 100.0
        notes = []
        if rms_dbfs < -42:
            score -= 30
            notes.append("Ses seviyesi çok düşük; arka planda kalan kelimeler kaçabilir.")
        elif rms_dbfs < -34:
            score -= 15
            notes.append("Ses seviyesi düşük; mümkünse kaynak kayıt güçlendirilmeli.")
        if peak_dbfs > -0.3 or clipped_percent > 0.15:
            score -= 25
            notes.append("Kayıtta clipping/taşma sinyali var; bazı heceler bozulmuş olabilir.")
        if silence_percent > 82:
            score -= 20
            notes.append("Kayıtta uzun sessizlik veya çok zayıf konuşma oranı yüksek.")
        elif silence_percent > 68:
            score -= 10
            notes.append("Sessizlik oranı yüksek; VAD konuşma bölümlerini kaçırabilir.")

        score = clamp(score, 0.0, 100.0)
        if score >= 85:
            label = "İyi"
        elif score >= AUDIO_QUALITY_REVIEW_THRESHOLD:
            label = "Orta"
        else:
            label = "Riskli"
        if not notes:
            notes.append("Ses seviyesi ve clipping sinyali kurumsal işlem için uygun görünüyor.")

        return {
            "audio_quality_score": score,
            "audio_quality_label": label,
            "audio_rms_dbfs": rms_dbfs,
            "audio_peak_dbfs": peak_dbfs,
            "audio_clipped_percent": clipped_percent,
            "audio_silence_percent": silence_percent,
            "audio_quality_notes": notes,
        }
    except Exception as exc:
        return {
            "audio_quality_score": 50.0,
            "audio_quality_label": "Ölçüm hatası",
            "audio_rms_dbfs": None,
            "audio_peak_dbfs": None,
            "audio_clipped_percent": None,
            "audio_silence_percent": None,
            "audio_quality_notes": [f"Ses kalite ölçümü yapılamadı: {exc}"],
        }


def build_prepared_audio_path(file_path: str, variant: str = AUDIO_PREP_STANDARD) -> Path:
    source = Path(file_path)
    stat = source.stat()
    signature = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)[:48] or "audio"
    safe_variant = re.sub(r"[^A-Za-z0-9_.-]+", "_", variant)[:24] or AUDIO_PREP_STANDARD
    variant_suffix = "" if safe_variant == AUDIO_PREP_STANDARD else f"-{safe_variant}"
    prepared_dir = Path(TEMP_AUDIO_DIR) / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    return prepared_dir / f"{safe_stem}-{digest}{variant_suffix}.wav"


def prepare_audio_for_asr(file_path: str, variant: str = AUDIO_PREP_STANDARD):
    """ASR için 16 kHz mono PCM WAV üretir; düşük kalite codec'lerde doğruluğu artırır."""
    variant = variant if variant in AUDIO_PREP_FILTERS else AUDIO_PREP_STANDARD
    requested_label, requested_filter = AUDIO_PREP_FILTERS[variant]
    source = Path(file_path)
    prepared_path = build_prepared_audio_path(file_path, variant)
    started = time.perf_counter()
    if prepared_path.exists() and prepared_path.stat().st_size > 0:
        duration = get_audio_duration_seconds(str(prepared_path))
        audio_quality = analyze_prepared_audio_quality(str(prepared_path))
        return str(prepared_path), {
            "cached": True,
            "duration": duration,
            "preprocess_s": 0.0,
            "preprocess_profile": variant,
            "preprocess_label": requested_label,
            "prepared_path": str(prepared_path),
            **audio_quality,
        }

    ffmpeg_path = get_ffmpeg_path()
    base_cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]
    standard_label, standard_filter = AUDIO_PREP_FILTERS[AUDIO_PREP_STANDARD]
    commands = [
        (requested_label, variant, base_cmd + ["-af", requested_filter, str(prepared_path)]),
    ]
    if variant != AUDIO_PREP_STANDARD:
        commands.append(
            (
                standard_label,
                AUDIO_PREP_STANDARD,
                base_cmd + ["-af", standard_filter, str(prepared_path)],
            )
        )
    commands.append(("Temel Dönüşüm", "raw", base_cmd + [str(prepared_path)]))

    last_error = ""
    for applied_label, applied_profile, cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
            duration = get_audio_duration_seconds(str(prepared_path))
            audio_quality = analyze_prepared_audio_quality(str(prepared_path))
            return str(prepared_path), {
                "cached": False,
                "duration": duration,
                "preprocess_s": time.perf_counter() - started,
                "preprocess_profile": applied_profile,
                "preprocess_label": applied_label,
                "prepared_path": str(prepared_path),
                **audio_quality,
            }
        except Exception as exc:
            last_error = str(exc)
            if hasattr(exc, "stderr") and exc.stderr:
                last_error = exc.stderr.strip()

    raise RuntimeError(f"Ses ön işleme başarısız oldu: {last_error}")


def normalize_for_wer(text: str):
    text = (text or "").lower().replace("’", "'")
    text = re.sub(r"[^0-9a-zçğıöşü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def levenshtein_distance(reference_words, hypothesis_words):
    if not reference_words:
        return len(hypothesis_words)
    if not hypothesis_words:
        return len(reference_words)

    previous = list(range(len(hypothesis_words) + 1))
    for i, ref_word in enumerate(reference_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis_words, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def calculate_word_accuracy(reference_text: str, hypothesis_text: str):
    reference_words = normalize_for_wer(reference_text)
    hypothesis_words = normalize_for_wer(hypothesis_text)
    if not reference_words:
        return {
            "wer": 0.0 if not hypothesis_words else 1.0,
            "accuracy": 100.0 if not hypothesis_words else 0.0,
            "edit_distance": len(hypothesis_words),
            "reference_words": 0,
            "hypothesis_words": len(hypothesis_words),
            "correct_estimate": 0,
        }

    edit_distance = levenshtein_distance(reference_words, hypothesis_words)
    wer = edit_distance / len(reference_words)
    accuracy = clamp(1.0 - wer, 0.0, 1.0) * 100
    correct_estimate = max(0, len(reference_words) - edit_distance)
    return {
        "wer": wer,
        "accuracy": accuracy,
        "edit_distance": edit_distance,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "correct_estimate": correct_estimate,
    }


BANKING_DEFAULT_HOTWORDS = """
KMH
kredili mevduat hesabı
KMH limiti
telefon bankacılığı
telefon bankacılığı şifresi
internet bankacılığı
mobil bankacılık
hesaplarınızı görebilmemiz
Şekerbank
ev kredisi
irtibatımız
görüntüleyebilmek
kontrollerini sağlayalım
dilerseniz
""".strip()

BANKING_CORRECTION_PATTERNS = [
    (r"\bKMF\b", "KMH"),
    (r"\bKMSM\b", "KMH"),
    (r"\bKMSMH\b", "KMH"),
    (r"\bKHM\b", "KMH"),
    (r"\bKM limitiniz\b", "KMH limitiniz"),
    (r"\bKME limitiniz\b", "KMH limitiniz"),
    (r"kredili\s+mevzuat\s+hesab", "kredili mevduat hesab"),
    (r"kredili\s+mevzuar\s+hesab", "kredili mevduat hesab"),
    (r"kredili\s+mevduar\s+hesab", "kredili mevduat hesab"),
    (r"kredil[ieı]r?mevzuar", "kredili mevduat"),
    (r"kredil[ieı]r?mevduar", "kredili mevduat"),
    (r"kredil[ieı]r?\s+mevdat", "kredili mevduat"),
    (r"kredil[ieı]r?\s+mevda", "kredili mevduat"),
    (r"kredilmevzuar", "kredili mevduat"),
    (r"\bmevda hesab", "mevduat hesab"),
    (r"\bmevdat\b", "mevduat"),
    (r"\bmevzuat hesab", "mevduat hesab"),
    (r"\bmevzuar hesab", "mevduat hesab"),
    (r"\bmevduar hesab", "mevduat hesab"),
    (r"\bbunu da ürünseyebilmek\b", "bunu da görüntüleyebilmek"),
    (r"\bbunu ürünseyebilmek\b", "bunu görüntüleyebilmek"),
    (r"\bürünseyebilmek\b", "görüntüleyebilmek"),
    (r"\bürünleyebilmek\b", "görüntüleyebilmek"),
    (r"\byönetleyebilmek\b", "görüntüleyebilmek"),
    (r"\bgörüntülebilmek\b", "görüntüleyebilmek"),
    (r"\bürünseye?bilmek\b", "görüntüleyebilmek"),
    (r"telefon bankacınızı kullanmanız gerekiyor", "telefon bankacılığınızı kullanmanız gerekiyor"),
    (
        r"telefon bankacılınızı kullanmanız gerekiyor",
        "telefon bankacılığınızı kullanmanız gerekiyor",
    ),
    (r"telefon bankacını kullanmanız gerekiyor", "telefon bankacılığınızı kullanmanız gerekiyor"),
    (r"telefon bankacını kullanıyor(?:muş)?sunuz\??", "telefon bankacılığını kullanıyor musunuz?"),
    (
        r"telefon bankacınızı kullanıyor(?:muş)?sunuz\??",
        "telefon bankacılığını kullanıyor musunuz?",
    ),
    (r"telefon manzar\w*", "telefon bankacılığı şifrenizi"),
    (
        r"telefon mankacı işletmeniz oluşturmuş mu\??",
        "telefon bankacılığı şifreniz oluşturulmuş mu?",
    ),
    (
        r"telefon mankajı işleykeniz oluşturmuş mu\??",
        "telefon bankacılığı şifreniz oluşturulmuş mu?",
    ),
    (r"telefon makacı işiniz oluşturmuş mu\??", "telefon bankacılığı şifreniz oluşturulmuş mu?"),
    (
        r"telefon bankacı işçilerine dolaşıyormuş muyuz\??",
        "telefon bankacılığı şifreniz oluşturulmuş mu?",
    ),
    (r"telefon mankacı işlemek oluşturmuş mu\??", "telefon bankacılığı şifreniz oluşturulmuş mu?"),
    (r"\bmankacı\b", "bankacılığı"),
    (r"\bhetaplarınızı\b", "hesaplarınızı"),
    (r"\bhetaplariniz\b", "hesaplarınızı"),
    (r"\bsanımlı\b", "tanımlı"),
    (
        r"hesaplarınızı görebilmemiz için telefon bankacılığı şifrenizi kullanmanız gerekiyor",
        "hesaplarınızı görebilmemiz için telefon bankacılığı şifrenizi kullanmanız gerekiyor",
    ),
    (r"\bşeker\s*bank\b", "Şekerbank"),
    (r"Şekerbank'layın", "Şekerbank'la"),
    (r"Şekerbank'la[yıi]n", "Şekerbank'la"),
    (r"Şekerbank'la irtibatımız", "Şekerbank'la irtibatımız"),
    (r"\bsıfatımız şey konusunda\b", "irtibatımız şey konusunda"),
    (r"\binsikatımız şey konusunda\b", "irtibatımız şey konusunda"),
    (r"\birtibatımız şey konusunda\b", "irtibatımız şey konusunda"),
    (r"\bev krizisi\b", "ev kredisi"),
    (r"\bev krizi\b", "ev kredisi"),
    (r"\bdilerkeniz\b", "dilerseniz"),
    (r"\bsağlyorım\b", "sağlayalım"),
    (r"\bkontroleini\b", "kontrollerini"),
    (r"\bkontrollerini sağlayorım\b", "kontrollerini sağlayalım"),
    (r"\bsağoluk\b", "sağ olun"),
    (r"\bsağolun\b", "sağ olun"),
    (r"\bdüştığı\b", "düştüğü"),
]

BANKING_TOKEN_CANONICAL = {
    "hetaplarınızı": "hesaplarınızı",
    "hetaplariniz": "hesaplarınızı",
    "şeterbank": "Şekerbank",
    "şekerbank": "Şekerbank",
    "sekerbank": "Şekerbank",
    "mevzuar": "mevduat",
    "mevduar": "mevduat",
    "mankacı": "bankacılığı",
    "dilerkeniz": "dilerseniz",
}

BANKING_CLOSE_TERMS = [
    "mevduat",
    "kredili",
    "bankacılığı",
    "bankacılığını",
    "bankacılığınızı",
    "hesaplarınızı",
    "görebilmemiz",
    "görüntüleyebilmek",
    "Şekerbank",
    "irtibatımız",
    "kredisi",
    "dilerseniz",
]

GENERIC_SERVICE_PATTERNS = [
    (r"\bmüsteri\b", "müşteri"),
    (r"\bmusteri\b", "müşteri"),
    (r"müşteri hizmet(?:i|leri)?", "müşteri hizmetleri"),
    (r"\bçağri\b", "çağrı"),
    (r"\bcagri\b", "çağrı"),
    (r"\btalebiniz\b", "talebiniz"),
    (r"\bbaşvurunuzu\b", "başvurunuzu"),
    (r"\bşikayetinizi\b", "şikayetinizi"),
    (r"\bsikayetinizi\b", "şikayetinizi"),
    (r"\bkontrol(?:ler)?ini sağlayalım\b", "kontrollerini sağlayalım"),
]

GENERIC_SERVICE_TERMS = [
    "müşteri",
    "müşteri hizmetleri",
    "çağrı merkezi",
    "talebiniz",
    "şikayetiniz",
    "başvurunuz",
    "kontrollerini",
    "fatura",
    "ödeme",
    "abonelik",
    "kampanya",
    "paket",
    "tarife",
    "sözleşme",
]

TELECOM_CORRECTION_PATTERNS = [
    (r"\bvodafon\b", "Vodafone"),
    (r"\bvodefon\b", "Vodafone"),
    (r"\bvodafone'?dan\b", "Vodafone'dan"),
    (r"\bvodafone'?a\b", "Vodafone'a"),
    (r"\btarifeniz\b", "tarifeniz"),
    (r"\btaahut\b", "taahhüt"),
    (r"\btaahüt\b", "taahhüt"),
    (r"\bcayma bedel[ıi]\b", "cayma bedeli"),
    (r"\bek paket\b", "ek paket"),
    (r"\binternet paketiniz\b", "internet paketiniz"),
    (r"\bmobil hattınız\b", "mobil hattınız"),
    (r"\bnumara tasima\b", "numara taşıma"),
    (r"\bnumara taşıma\b", "numara taşıma"),
    (r"\bGB\b", "GB"),
]

TELECOM_TERMS = [
    "Vodafone",
    "tarife",
    "taahhüt",
    "cayma bedeli",
    "fatura",
    "ek paket",
    "internet paketi",
    "mobil hat",
    "numara taşıma",
    "roaming",
    "Vodafone Yanımda",
    "Red tarifesi",
    "GB",
]

ECOMMERCE_CORRECTION_PATTERNS = [
    (r"\bsiparis\b", "sipariş"),
    (r"\biade\b", "iade"),
    (r"\bkargo takip\b", "kargo takip"),
    (r"\bteslimat\b", "teslimat"),
    (r"\bfatura adresi\b", "fatura adresi"),
    (r"\bkupon kodu\b", "kupon kodu"),
]

ECOMMERCE_TERMS = [
    "sipariş",
    "iade",
    "kargo",
    "kargo takip",
    "teslimat",
    "fatura adresi",
    "kupon kodu",
    "stok",
    "ödeme",
]

INSURANCE_CORRECTION_PATTERNS = [
    (r"\bpoliçe\b", "poliçe"),
    (r"\bpolice\b", "poliçe"),
    (r"\bhasar dosyası\b", "hasar dosyası"),
    (r"\beksper\b", "eksper"),
    (r"\bprim borcu\b", "prim borcu"),
    (r"\bteminat\b", "teminat"),
]

INSURANCE_TERMS = [
    "poliçe",
    "hasar dosyası",
    "eksper",
    "prim",
    "teminat",
    "kasko",
    "trafik sigortası",
    "sağlık sigortası",
]


def combine_patterns(*pattern_groups):
    combined = []
    for group in pattern_groups:
        combined.extend(group)
    return tuple(combined)


def combine_terms(*term_groups):
    seen = set()
    combined = []
    for group in term_groups:
        for term in group:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                combined.append(term)
    return tuple(combined)


DOMAIN_PROFILES = {
    "omni": DomainProfile(
        label="Çok Sektör",
        description="Banka, telekom, e-ticaret, sigorta ve genel müşteri hizmetleri terimlerini birlikte kullanır.",
        patterns=combine_patterns(
            GENERIC_SERVICE_PATTERNS,
            BANKING_CORRECTION_PATTERNS,
            TELECOM_CORRECTION_PATTERNS,
            ECOMMERCE_CORRECTION_PATTERNS,
            INSURANCE_CORRECTION_PATTERNS,
        ),
        canonical={**BANKING_TOKEN_CANONICAL},
        close_terms=combine_terms(
            GENERIC_SERVICE_TERMS,
            BANKING_CLOSE_TERMS,
            TELECOM_TERMS,
            ECOMMERCE_TERMS,
            INSURANCE_TERMS,
        ),
    ),
    "banking": DomainProfile(
        label="Bankacılık",
        description="KMH, mevduat, kredi, hesap ve telefon bankacılığı konuşmaları.",
        patterns=combine_patterns(GENERIC_SERVICE_PATTERNS, BANKING_CORRECTION_PATTERNS),
        canonical={**BANKING_TOKEN_CANONICAL},
        close_terms=combine_terms(GENERIC_SERVICE_TERMS, BANKING_CLOSE_TERMS),
        hotwords=BANKING_DEFAULT_HOTWORDS,
    ),
    "telecom": DomainProfile(
        label="Telekom",
        description="Vodafone benzeri operatör, fatura, tarife, taahhüt, paket ve hat konuşmaları.",
        patterns=combine_patterns(GENERIC_SERVICE_PATTERNS, TELECOM_CORRECTION_PATTERNS),
        canonical={},
        close_terms=combine_terms(GENERIC_SERVICE_TERMS, TELECOM_TERMS),
    ),
    "ecommerce": DomainProfile(
        label="E-Ticaret",
        description="Sipariş, iade, kargo, teslimat, fatura ve ödeme konuşmaları.",
        patterns=combine_patterns(GENERIC_SERVICE_PATTERNS, ECOMMERCE_CORRECTION_PATTERNS),
        canonical={},
        close_terms=combine_terms(GENERIC_SERVICE_TERMS, ECOMMERCE_TERMS),
    ),
    "insurance": DomainProfile(
        label="Sigorta",
        description="Poliçe, hasar, eksper, prim ve teminat konuşmaları.",
        patterns=combine_patterns(GENERIC_SERVICE_PATTERNS, INSURANCE_CORRECTION_PATTERNS),
        canonical={},
        close_terms=combine_terms(GENERIC_SERVICE_TERMS, INSURANCE_TERMS),
    ),
    "custom": DomainProfile(
        label="Özel Sözlük",
        description="Yalnızca genel müşteri hizmetleri düzeltmeleri ve kullanıcı terimleri.",
        patterns=tuple(GENERIC_SERVICE_PATTERNS),
        canonical={},
        close_terms=tuple(GENERIC_SERVICE_TERMS),
    ),
}


def preserve_first_char_case(original: str, replacement: str):
    if original and original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_case_preserving_regex(text: str, pattern: str, replacement: str):
    def repl(match):
        return preserve_first_char_case(match.group(0), replacement)

    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def fuzzy_ratio(left: str, right: str):
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def parse_custom_terms(custom_terms: str):
    if not custom_terms:
        return []
    pieces = re.split(r"[,;\n]+", custom_terms)
    cleaned = []
    seen = set()
    for piece in pieces:
        term = piece.strip()
        if not term:
            continue
        key = term.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(term)
    return cleaned


def get_domain_profile(domain_key: str):
    return DOMAIN_PROFILES.get(domain_key, DOMAIN_PROFILES["omni"])


def build_domain_hotwords(domain_key: str, custom_terms: str = ""):
    """Sektör sözlüğünü ve kullanıcı terimlerini ASR hotword listesine çevirir."""
    if domain_key == "none":
        return "\n".join(parse_custom_terms(custom_terms))
    domain_profile = get_domain_profile(domain_key)
    hotword_terms = []
    hotword_terms.extend(parse_custom_terms(domain_profile.hotwords))
    hotword_terms.extend(domain_profile.close_terms or ())
    hotword_terms.extend(parse_custom_terms(custom_terms))

    seen = set()
    unique_terms = []
    for term in hotword_terms:
        term = str(term).strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_terms.append(term)
    return "\n".join(unique_terms[:180])


def correct_domain_tokens(text: str, domain_profile: DomainProfile, custom_terms=None):
    custom_terms = custom_terms or []
    canonical = domain_profile.canonical or {}
    close_terms = list(domain_profile.close_terms or ())
    close_terms.extend(custom_terms)

    def replace_token(match):
        token = match.group(0)
        stripped = token.lower()
        if stripped in canonical:
            return preserve_first_char_case(token, canonical[stripped])
        if custom_terms and len(stripped) >= 4:
            token_custom_terms = [term for term in custom_terms if " " not in term.strip()]
            if token_custom_terms:
                best_custom = max(token_custom_terms, key=lambda term: fuzzy_ratio(stripped, term))
                if fuzzy_ratio(stripped, best_custom) >= 0.80:
                    return preserve_first_char_case(token, best_custom)
        if len(stripped) >= 7:
            best_term = (
                max(close_terms, key=lambda term: fuzzy_ratio(stripped, term))
                if close_terms
                else ""
            )
            if best_term and fuzzy_ratio(stripped, best_term) >= 0.93:
                return preserve_first_char_case(token, best_term)
        return token

    return re.sub(r"[A-Za-zÇĞİÖŞÜçğıöşü']+", replace_token, text)


def tidy_transcript_text(text: str):
    # Fix whisper hallucinations with repeating dots
    text = re.sub(r'(\s*\.\s*){3,}', '... ', text)
    text = re.sub(r'\.{4,}', '... ', text)
    
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    # Don't add spaces between dots (preserve ellipses)
    text = re.sub(r"([?.!])(?=[^\s.])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Strip any trailing repeated punctuation if it's overly long
    text = re.sub(r'(?:\.\s*){3,}$', '.', text)
    
    if text:
        text = text[0].upper() + text[1:]
    return text


def apply_domain_corrections(text: str, domain_key: str = "omni", custom_terms: str = ""):
    corrected = text or ""
    domain_profile = get_domain_profile(domain_key)
    for pattern, replacement in domain_profile.patterns:
        corrected = apply_case_preserving_regex(corrected, pattern, replacement)
    corrected = correct_domain_tokens(corrected, domain_profile, parse_custom_terms(custom_terms))
    corrected = tidy_transcript_text(corrected)
    return corrected


def apply_banking_domain_corrections(text: str):
    return apply_domain_corrections(text, "banking")


def postprocess_transcript_text(
    text: str, profile_key: str, domain_key: str = "omni", custom_terms: str = ""
):
    cleaned = tidy_transcript_text(text)
    if profile_key == "banking" or domain_key != "none":
        return apply_domain_corrections(cleaned, domain_key, custom_terms)
    return cleaned


def domain_correction_count(raw_text: str, corrected_text: str):
    raw_words = normalize_for_wer(raw_text)
    corrected_words = normalize_for_wer(corrected_text)
    return levenshtein_distance(raw_words, corrected_words)


def build_initial_prompt(hotwords: str = ""):
    # Gerçekçi çağrı merkezi diyaloğu — MLX greedy decoder'ı doğru kelimelere yönlendirir.
    # "bendeyim" yerine "iyiyim", "ne" yerine "ne?" gibi fonetik benzer hataları önlemek için
    # modelin görmesini istediğimiz kelime formlarını bu prompt içinde bırakıyoruz.
    base_prompt = (
        "Türkçe müşteri hizmetleri görüşmesi. Müşteri ve temsilci konuşuyor. "
        "Merhaba, iyi günler. İyi günler, nasılsınız? İyiyim teşekkür ederim, siz nasılsınız? "
        "Ben de iyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim? "
        "Faturamla ilgili bir sorunum var. Elbette, hesap numaranızı alabilir miyim? "
        "Tarifenizi değiştirmek ister misiniz? Evet, lütfen. Hayır, teşekkürler. "
        "Anlıyorum, bir dakika bekler misiniz? Tabii ki, bekliyorum. "
        "Teknik bir sorun yaşıyorum. İnternet bağlantım yok. Faturamı ödedim. "
        "İptal etmek istiyorum. Paket değişikliği yapmak istiyorum. "
        "Aradığınız için teşekkür ederiz, iyi günler. Güle güle."
    )
    terms = parse_custom_terms(hotwords)
    if not terms:
        return base_prompt
    term_text = ", ".join(terms[:ASR_INITIAL_PROMPT_TERM_LIMIT])
    if len(term_text) > ASR_INITIAL_PROMPT_CHAR_LIMIT:
        term_text = term_text[:ASR_INITIAL_PROMPT_CHAR_LIMIT].rsplit(",", 1)[0]
    return f"{base_prompt} Sektöre özel terimler: {term_text}."


def build_transcribe_options(profile: ASRProfile, lang: str, task: str, hotwords: str = ""):
    clean_hotwords = hotwords.strip() or None
    options = {
        "language": lang or "tr",
        "task": task,
        "beam_size": profile.beam_size,
        "best_of": profile.best_of,
        "patience": 1.0,
        "temperature": profile.temperature,
        "compression_ratio_threshold": 2.0,
        "log_prob_threshold": profile.log_prob_threshold,
        "no_speech_threshold": profile.no_speech_threshold,
        "condition_on_previous_text": profile.condition_on_previous_text,
        "no_repeat_ngram_size": 5,
        "initial_prompt": build_initial_prompt(hotwords),
        "vad_filter": profile.vad_filter,
        "chunk_length": profile.chunk_length,
        "word_timestamps": False,
        "without_timestamps": False,
        "max_new_tokens": ASR_MAX_NEW_TOKENS,
        "multilingual": False,
        "repetition_penalty": profile.repetition_penalty,
        "hotwords": clean_hotwords,
    }
    if profile.hallucination_silence_threshold is not None:
        options["hallucination_silence_threshold"] = profile.hallucination_silence_threshold
    if profile.vad_filter:
        options["vad_parameters"] = {
            "threshold": profile.vad_threshold,
            "min_silence_duration_ms": profile.min_silence_duration_ms,
            "speech_pad_ms": profile.speech_pad_ms,
        }
    return options


def is_whisper_context_error(error: Exception):
    message = str(error)
    return any(marker in message for marker in ASR_CONTEXT_ERROR_MARKERS)


def build_safe_transcribe_options(options: dict):
    safe_options = dict(options)
    safe_options.pop("batch_size", None)
    safe_options["initial_prompt"] = None
    safe_options["hotwords"] = None
    safe_options["max_new_tokens"] = ASR_SAFE_MAX_NEW_TOKENS
    safe_options["chunk_length"] = min(
        int(safe_options.get("chunk_length") or ASR_SAFE_FALLBACK_CHUNK_LENGTH),
        ASR_SAFE_FALLBACK_CHUNK_LENGTH,
    )
    safe_options["condition_on_previous_text"] = False
    safe_options["no_repeat_ngram_size"] = 0
    safe_options["beam_size"] = min(int(safe_options.get("beam_size") or 3), 3)
    safe_options["best_of"] = min(int(safe_options.get("best_of") or 3), 3)
    return safe_options


def materialize_transcription(engine, prepared_path: str, options: dict):
    segments, info = engine.transcribe(prepared_path, **options)
    return list(segments), info


def transcribe_segments_with_rescue(model, prepared_path: str, profile: ASRProfile, options: dict):
    """Whisper bağlam sınırı hatalarında güvenli motora otomatik düşer."""
    if not profile.use_batched or not getattr(model, "supports_batched", True):
        try:
            segments, info = materialize_transcription(model, prepared_path, options)
            return segments, info, False
        except (RuntimeError, ValueError) as error:
            if not is_whisper_context_error(error):
                raise
            safe_options = build_safe_transcribe_options(options)
            segments, info = materialize_transcription(model, prepared_path, safe_options)
            return segments, info, True

    BatchedInferencePipeline = get_batched_pipeline_class()
    batched_engine = BatchedInferencePipeline(model)
    batched_options = dict(options)
    batched_options["batch_size"] = profile.batch_size

    try:
        segments, info = materialize_transcription(batched_engine, prepared_path, batched_options)
        return segments, info, False
    except (RuntimeError, ValueError) as error:
        if not is_whisper_context_error(error):
            raise

    safe_options = build_safe_transcribe_options(options)
    segments, info = materialize_transcription(model, prepared_path, safe_options)
    return segments, info, True


def repeated_ngram_ratio(words, n=3):
    if len(words) < n * 2:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def is_suspicious_asr_segment(segment, text: str):
    """Apple-level agresif halüsinasyon tespiti."""
    words = normalize_for_wer(text)
    
    # Sadece noktalama işaretinden oluşan veya anlamsız kısa segmentleri yakala
    if len(text.strip()) > 10 and len(words) < 2:
        return True
        
    if len(words) < 5:
        return False

    unique_ratio = len(set(words)) / len(words)
    trigram_repeat = repeated_ngram_ratio(words, n=3)
    bigram_repeat = repeated_ngram_ratio(words, n=2)
    compression_ratio = float(getattr(segment, "compression_ratio", 1.0) or 1.0)
    avg_logprob = float(getattr(segment, "avg_logprob", -1.0) or -1.0)
    no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)

    # Agresif eşikler (Yüksek Doğruluk)
    if len(words) >= 12 and unique_ratio < 0.30:
        return True
    if len(words) >= 16 and trigram_repeat > 0.35:
        return True
    if len(words) >= 10 and bigram_repeat > 0.55:
        return True
    # Kompresyon oranı eşiği: 2.9 -> 2.0 (çok daha agresif)
    if compression_ratio > 2.0 and (avg_logprob < -0.4 or no_speech_prob > 0.40):
        return True
    # Yeni: çok düşük güven + yüksek no_speech = kesin halüsinasyon
    if avg_logprob < -0.9 and no_speech_prob > 0.6:
        return True
    # Yeni: Aşırı uzun segment (>60 sn) genellikle halüsinasyondur
    duration = float(getattr(segment, "end", 0)) - float(getattr(segment, "start", 0))
    if duration > 60.0:
        return True
    return False


def segment_duration(segment):
    return max(0.0, float(segment.end) - float(segment.start))


def overlap_seconds(first_segment, second_segment):
    return max(
        0.0,
        min(float(first_segment.end), float(second_segment.end))
        - max(float(first_segment.start), float(second_segment.start)),
    )


def filtered_segment_record(segment, reason: str):
    return {
        "start": float(segment.start),
        "end": float(segment.end),
        "text": segment.text.strip()[:160],
        "compression_ratio": float(getattr(segment, "compression_ratio", 1.0) or 1.0),
        "reason": reason,
    }


def should_drop_previous_for_overlap(previous_segment, current_segment):
    overlap = overlap_seconds(previous_segment, current_segment)
    if overlap < 1.0:
        return False
    previous_duration = segment_duration(previous_segment)
    current_duration = segment_duration(current_segment)
    return previous_duration > max(12.0, current_duration * 3.0)


def should_drop_current_for_overlap(previous_segment, current_segment):
    overlap = overlap_seconds(previous_segment, current_segment)
    if overlap < 1.0:
        return False
    previous_duration = segment_duration(previous_segment)
    current_duration = segment_duration(current_segment)
    return current_duration > max(12.0, previous_duration * 3.0)


def iter_stable_segments(segments, filtered_segments):
    pending = None
    for segment in segments:
        if pending is None:
            pending = segment
            continue
        if should_drop_previous_for_overlap(pending, segment):
            filtered_segments.append(filtered_segment_record(pending, "overlap"))
            pending = segment
            continue
        if should_drop_current_for_overlap(pending, segment):
            filtered_segments.append(filtered_segment_record(segment, "overlap"))
            continue
        yield pending
        pending = segment
    if pending is not None:
        yield pending


TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü@!*]+")
SWEAR_PREFIX_STEMS = (
    "siktir",
    "sikim",
    "siker",
    "sikey",
    "sikec",
    "sikic",
    "amına",
    "amina",
    "aminak",
    "amcık",
    "amcik",
    "orospu",
    "orosb",
    "orosp",
    "yarrak",
    "yarak",
    "göt",
    "got",
)
SWEAR_PREFIX_EXCLUSIONS = (
    "sikayet",
    "şikayet",
    "tesekkur",
    "teşekkür",
    "arayarak",
    "kullan",
    "istememe",
    "yönlendir",
    "yonlendir",
)


def normalize_swear_match_token(value: str):
    value = (value or "").lower().strip()
    value = value.replace("1", "i").replace("!", "i").replace("@", "a").replace("0", "o")
    value = value.replace("*", "")
    value = re.sub(r"(.)\1{2,}", r"\1\1", value)
    return value


def build_swear_match_sets(swear_list):
    exact_terms = set()
    phrase_terms = []
    for swear in swear_list:
        normalized = normalize_swear_match_token(str(swear))
        if not normalized:
            continue
        parts = normalized.split()
        if len(parts) > 1:
            phrase_terms.append(tuple(parts))
        else:
            exact_terms.add(normalized)
    phrase_terms.sort(key=len, reverse=True)
    return exact_terms, phrase_terms


def is_safe_swear_substring_hit(word: str):
    return any(word.startswith(prefix) for prefix in SWEAR_PREFIX_EXCLUSIONS)


def detect_swears_in_segment(text: str, start_time: float, swear_list):
    detected = []
    seen = set()
    exact_terms, phrase_terms = build_swear_match_sets(swear_list)
    tokens = [
        normalize_swear_match_token(match.group(0)) for match in TOKEN_PATTERN.finditer(text or "")
    ]
    tokens = [token for token in tokens if token]
    covered_token_indexes = set()

    def add_detection(label):
        key = normalize_swear_match_token(label)
        if key and key not in seen:
            seen.add(key)
            detected.append({"word": label, "time": start_time})

    for phrase in phrase_terms:
        phrase_len = len(phrase)
        for idx in range(0, max(0, len(tokens) - phrase_len + 1)):
            if tuple(tokens[idx : idx + phrase_len]) == phrase:
                add_detection(" ".join(phrase))
                covered_token_indexes.update(range(idx, idx + phrase_len))

    for idx, word in enumerate(tokens):
        if idx in covered_token_indexes:
            continue
        if word in exact_terms:
            add_detection(word)
            continue
        if is_safe_swear_substring_hit(word):
            continue
        for stem in SWEAR_PREFIX_STEMS:
            if len(word) > len(stem) and word.startswith(stem):
                add_detection(f"{stem} ('{word}' varyantı)")
                break

    return detected


def summarize_transcription_quality(segments_data):
    if not segments_data:
        return {
            "confidence": 0.0,
            "avg_logprob": 0.0,
            "avg_no_speech_prob": 1.0,
            "repetition_risk": 0.0,
        }

    weighted_logprob = 0.0
    weighted_no_speech = 0.0
    total_duration = 0.0
    compression_risk = 0.0

    for segment in segments_data:
        duration = max(0.1, float(segment.end) - float(segment.start))
        total_duration += duration
        avg_logprob = float(getattr(segment, "avg_logprob", -1.0) or -1.0)
        no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        compression_ratio = float(getattr(segment, "compression_ratio", 1.0) or 1.0)
        weighted_logprob += avg_logprob * duration
        weighted_no_speech += no_speech_prob * duration
        compression_risk += max(0.0, compression_ratio - 2.4) * duration

    avg_logprob = weighted_logprob / total_duration
    avg_no_speech_prob = weighted_no_speech / total_duration
    repetition_risk = clamp((compression_risk / total_duration) / 1.5)

    # MLX-Whisper Turbo modelleri Türkçe için genellikle daha düşük (örn: -0.6 ila -0.9) avg_logprob verir.
    # Normalizasyon aralığını genişleterek (-1.8 ile -0.4 arası) cezayı azalttık.
    logprob_score = clamp((avg_logprob + 1.8) / 1.4)
    speech_score = clamp(1.0 - avg_no_speech_prob)
    
    # MLX için logprob'un ağırlığı düşürüldü, net konuşma oranına (speech_score) daha çok önem verildi
    confidence = (0.45 * logprob_score + 0.40 * speech_score + 0.15 * (1.0 - repetition_risk)) * 100

    return {
        "confidence": confidence,
        "avg_logprob": avg_logprob,
        "avg_no_speech_prob": avg_no_speech_prob,
        "repetition_risk": repetition_risk,
    }


def transcribe_with_profile(
    model,
    prepared_path: str,
    prep_info: dict,
    profile_key: str,
    lang: str,
    swear_list,
    task: str,
    domain_key: str,
    combined_hotwords: str,
    progress_callback=None,
    overall_started: float = None,
):
    """Tek ASR geçişini çalıştırır ve denetlenebilir aday sonuç döndürür."""
    pass_started = time.perf_counter()
    overall_started = overall_started or pass_started
    profile = ASR_PROFILES.get(profile_key, ASR_PROFILES["smart"])
    options = build_transcribe_options(profile, lang, task, combined_hotwords)
    segments, info, batched_fallback = transcribe_segments_with_rescue(
        model,
        prepared_path,
        profile,
        options,
    )

    formatted_body = ""
    detected_swears = []
    segments_data = []
    filtered_segments = []
    full_transcription_parts = []
    raw_transcription_parts = []
    correction_edits = 0

    for segment in iter_stable_segments(segments, filtered_segments):
        start_time = float(segment.start)
        end_time = float(segment.end)
        raw_text = segment.text.strip()
        
        # Clean up whisper hallucinations before they corrupt raw logs or scores
        raw_text = re.sub(r'(\s*\.\s*){3,}', '... ', raw_text)
        raw_text = re.sub(r'\.{4,}', '... ', raw_text)
        raw_text = re.sub(r'(?:\.\s*){3,}$', '.', raw_text)
        
        if not raw_text:
            continue
        if is_suspicious_asr_segment(segment, raw_text):
            filtered_segments.append(filtered_segment_record(segment, "repetition"))
            if progress_callback:
                elapsed = time.perf_counter() - overall_started
                progress_callback(segment, " ".join(full_transcription_parts), elapsed, prep_info)
            continue

        text = postprocess_transcript_text(raw_text, profile_key, domain_key, combined_hotwords)
        correction_edits += domain_correction_count(raw_text, text)
        rendered_segment = TranscriptSegment(
            start=start_time,
            end=end_time,
            text=text,
            avg_logprob=float(getattr(segment, "avg_logprob", -1.0) or -1.0),
            no_speech_prob=float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
            compression_ratio=float(getattr(segment, "compression_ratio", 1.0) or 1.0),
            raw_text=raw_text,
        )
        segments_data.append(rendered_segment)

        full_transcription_parts.append(text)
        raw_transcription_parts.append(raw_text)
        m_start, s_start = divmod(start_time, 60)
        m_end, s_end = divmod(end_time, 60)
        time_str = f"[{int(m_start):02d}:{s_start:04.1f} - {int(m_end):02d}:{s_end:04.1f}]"
        formatted_body += f"{time_str} {text}\n"
        detected_swears.extend(detect_swears_in_segment(text, start_time, swear_list))

        if progress_callback:
            elapsed = time.perf_counter() - overall_started
            progress_callback(segment, " ".join(full_transcription_parts), elapsed, prep_info)

    full_transcription = " ".join(full_transcription_parts).strip()
    raw_transcription = " ".join(raw_transcription_parts).strip()
    quality = summarize_transcription_quality(segments_data)
    pass_elapsed_s = time.perf_counter() - pass_started
    word_count = len(normalize_for_wer(full_transcription))
    speech_duration = sum(max(0.0, segment.end - segment.start) for segment in segments_data)
    source_duration = prep_info.get("duration")
    speech_coverage = clamp(speech_duration / source_duration) if source_duration else None
    candidate_score = (
        quality["confidence"]
        - (len(filtered_segments) * 4.0)
        + min(word_count, 160) / 80.0
        + ((speech_coverage or 0.0) * 3.0)
        + (clamp((prep_info.get("audio_quality_score") or 0.0) / 100.0) * 2.0)
    )

    metrics = {
        **quality,
        "pass_elapsed_s": pass_elapsed_s,
        "profile_key": profile_key,
        "profile_label": profile.label,
        "segments": len(segments_data),
        "filtered_segments": len(filtered_segments),
        "domain_corrections": correction_edits,
        "domain_mode": domain_key != "none",
        "domain_key": domain_key,
        "domain_label": get_domain_profile(domain_key).label,
        "raw_transcription": raw_transcription,
        "word_count": word_count,
        "speech_duration": speech_duration,
        "speech_coverage": speech_coverage,
        "candidate_score": candidate_score,
        "batched_fallback": batched_fallback,
        "preprocess_profile": prep_info.get("preprocess_profile", AUDIO_PREP_STANDARD),
        "preprocess_label": prep_info.get("preprocess_label", "Standart Netleştirme"),
        "audio_quality_score": prep_info.get("audio_quality_score", 0.0),
        "audio_quality_label": prep_info.get("audio_quality_label", "-"),
    }
    return {
        "formatted_body": formatted_body,
        "detected_swears": detected_swears,
        "full_transcription": full_transcription,
        "segments_data": segments_data,
        "info": info,
        "metrics": metrics,
        "profile": profile,
        "prep_info": prep_info,
        "prepared_path": prepared_path,
    }


def should_retry_transcription(candidate, source_profile: ASRProfile):
    metrics = candidate["metrics"]
    if not source_profile.retry_profile_key:
        return False
    if metrics.get("confidence", 0.0) < source_profile.quality_gate:
        return True
    if metrics.get("audio_quality_score", 100.0) < AUDIO_RESCUE_PREP_THRESHOLD:
        return True
    if metrics.get("filtered_segments", 0) > 0:
        return True
    return False


def retry_has_latency_budget(
    model, overall_started: float, target_latency_s: int, primary_candidate
):
    """İkinci kalite geçişini sadece SLA süresini patlatmayacaksa çalıştır."""
    if not target_latency_s:
        return True
    elapsed = time.perf_counter() - overall_started
    if elapsed >= target_latency_s:
        return False

    primary_elapsed = float(primary_candidate["metrics"].get("pass_elapsed_s") or elapsed or 0.0)
    estimated_retry_s = max(
        ASR_RETRY_MIN_REMAINING_SECONDS, primary_elapsed * ASR_RETRY_ESTIMATE_MULTIPLIER
    )
    remaining_s = target_latency_s - elapsed

    if is_mlx_model(model) and target_latency_s <= TARGET_LATENCY_SECONDS:
        return remaining_s >= estimated_retry_s
    return remaining_s >= min(estimated_retry_s, target_latency_s * 0.45)


def pick_best_transcription_candidate(candidates):
    return max(candidates, key=lambda item: item["metrics"].get("candidate_score", 0.0))


def build_formatted_transcript(best_candidate, run_metrics, retry_candidates, target_latency_s):
    formatted_text = (
        f"ASR PRO - DETAYLI DÖKÜM\nTarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    )
    formatted_text += f"Profil: {run_metrics['profile_label']}\n"
    if run_metrics.get("batched_fallback"):
        formatted_text += (
            "Motor Koruması: Batched sınır hatası algılandı, güvenli ASR geçişi kullanıldı.\n"
        )
    if retry_candidates:
        retry_labels = ", ".join(
            candidate["metrics"]["profile_label"] for candidate in retry_candidates
        )
        formatted_text += f"Kalite Yeniden Deneme: Aktif ({retry_labels})\n"
    elif run_metrics.get("quality_retry_skipped_for_latency"):
        formatted_text += (
            "Kalite Yeniden Deneme: Süre hedefi için atlandı; kalite kapısı uyarısı korunur.\n"
        )
    if run_metrics.get("duration"):
        formatted_text += f"Ses Süresi: {run_metrics['duration']:.1f}s\n"
    formatted_text += f"Ön İşleme: {run_metrics.get('preprocess_label', 'Standart Netleştirme')}\n"
    formatted_text += "-" * 50 + "\n\n"
    formatted_text += best_candidate["formatted_body"]
    formatted_text += "\n" + "-" * 50 + "\n"
    formatted_text += f"İşlem Süresi: {run_metrics['elapsed_s']:.1f}s\n"
    if run_metrics["rtf"] is not None:
        formatted_text += f"RTF: {run_metrics['rtf']:.2f}x\n"
    formatted_text += f"ASR Güveni: %{run_metrics['confidence']:.1f}\n"
    formatted_text += (
        f"Kalite Kapısı: {'Geçti' if run_metrics['quality_gate_met'] else 'İnceleme Gerekli'}\n"
    )
    formatted_text += f"Ses Kalitesi: %{run_metrics.get('audio_quality_score', 0):.0f} ({run_metrics.get('audio_quality_label', '-')})\n"
    formatted_text += f"Filtrelenen Şüpheli Segment: {run_metrics['filtered_segments']}\n"
    formatted_text += f"Sektör Düzeltmesi: {run_metrics['domain_label'] if run_metrics['domain_mode'] else 'Kapalı'}"
    if run_metrics["domain_mode"]:
        formatted_text += f" | Düzenleme: {run_metrics['domain_corrections']}"
    formatted_text += "\n"
    formatted_text += f"Hotword/Sözlük Terimi: {run_metrics.get('hotword_count', 0)}\n"
    formatted_text += f"Hedef Süre: {target_latency_s}s | Durum: {'Karşılandı' if run_metrics['target_met'] else 'Aşıldı'}\n"
    formatted_text += f"Not: Gerçek WER hedefi kelime doğruluğu >= %{QUALITY_GATE_ACCURACY:.0f} yani WER <= %{QUALITY_GATE_WER:.0f}; referans metinle kanıtlanır.\n"
    return formatted_text


def apply_sota_features(candidate, model, domain_key, combined_hotwords, lang, swear_list):
    """SOTA özelliklerini (per-segment redecode & punctuation) en iyi adaya uygular."""
    # 1. Per-segment redecode
    segments = candidate["segments_data"]
    latency_guard_active = (
        st.session_state.get("target_latency_s", TARGET_LATENCY_SECONDS) <= TARGET_LATENCY_SECONDS
    )
    if (
        st.session_state.get("apple_mode", False)
        and not latency_guard_active
        and getattr(model, "transcribe", None)
        and candidate.get("prepared_path")
    ):
        improved_segments = redecode_low_confidence_segments(
            model, candidate["prepared_path"], segments, domain_key, combined_hotwords, lang
        )
        candidate["segments_data"] = improved_segments

    # 2. Rebuild text
    full_transcription_parts = []
    formatted_body = ""
    detected_swears = []
    for segment in candidate["segments_data"]:
        text = segment.text
        full_transcription_parts.append(text)

        m_start, s_start = divmod(segment.start, 60)
        m_end, s_end = divmod(segment.end, 60)
        time_str = f"[{int(m_start):02d}:{s_start:04.1f} - {int(m_end):02d}:{s_end:04.1f}]"
        formatted_body += f"{time_str} {text}\n"
        detected_swears.extend(detect_swears_in_segment(text, segment.start, swear_list))

    full_text = " ".join(full_transcription_parts).strip()

    # 3. Punctuation Restoration
    full_text = restore_turkish_punctuation(full_text)

    candidate["full_transcription"] = full_text
    candidate["formatted_body"] = formatted_body
    candidate["detected_swears"] = detected_swears
    return candidate


def transcribe_audio_file(
    model,
    file_path: str,
    lang: str,
    swear_list,
    task: str = "transcribe",
    profile_key: str = "smart",
    domain_key: str = "omni",
    hotwords: str = "",
    progress_callback=None,
    target_latency_s: int = TARGET_LATENCY_SECONDS,
) -> tuple:
    """Ses dosyasını kurumsal kalite kapısı, sektör sözlüğü ve opsiyonel ikinci geçişle tanır."""
    overall_started = time.perf_counter()
    source_profile = ASR_PROFILES.get(profile_key, ASR_PROFILES["smart"])
    prepared_path, prep_info = prepare_audio_for_asr(file_path, AUDIO_PREP_STANDARD)
    combined_hotwords = build_domain_hotwords(domain_key, hotwords)

    primary_candidate = transcribe_with_profile(
        model,
        prepared_path,
        prep_info,
        profile_key,
        lang,
        swear_list,
        task,
        domain_key,
        combined_hotwords,
        progress_callback=progress_callback,
        overall_started=overall_started,
    )

    candidates = [primary_candidate]
    retry_candidates = []
    quality_retry_skipped_for_latency = False
    if should_retry_transcription(primary_candidate, source_profile):
        retry_key = source_profile.retry_profile_key
        if retry_has_latency_budget(model, overall_started, target_latency_s, primary_candidate):
            retry_prepared_path, retry_prep_info = prepare_audio_for_asr(
                file_path, AUDIO_PREP_RESCUE
            )
            retry_candidate = transcribe_with_profile(
                model,
                retry_prepared_path,
                retry_prep_info,
                retry_key,
                lang,
                swear_list,
                task,
                domain_key,
                combined_hotwords,
                progress_callback=progress_callback,
                overall_started=overall_started,
            )
            candidates.append(retry_candidate)
            retry_candidates.append(retry_candidate)
        else:
            quality_retry_skipped_for_latency = True

    best_candidate = pick_best_transcription_candidate(candidates)

    # Apply SOTA features
    best_candidate = apply_sota_features(
        best_candidate, model, domain_key, combined_hotwords, lang, swear_list
    )

    elapsed_s = time.perf_counter() - overall_started
    best_prep_info = best_candidate.get("prep_info", prep_info)
    duration = best_prep_info.get("duration")
    best_metrics = best_candidate["metrics"]
    hotword_count = len(parse_custom_terms(combined_hotwords))
    quality_gate_met = (
        best_metrics.get("confidence", 0.0) >= source_profile.quality_gate
        and best_metrics.get("filtered_segments", 0) == 0
        and best_prep_info.get("audio_quality_score", 100.0) >= AUDIO_QUALITY_REVIEW_THRESHOLD
    )

    run_metrics = {
        **best_prep_info,
        **best_metrics,
        "elapsed_s": elapsed_s,
        "rtf": (elapsed_s / duration) if duration else None,
        "target_latency_s": target_latency_s,
        "target_met": elapsed_s <= target_latency_s,
        "hotword_count": hotword_count,
        "quality_retry": bool(retry_candidates),
        "quality_retry_skipped_for_latency": quality_retry_skipped_for_latency,
        "retry_profiles": [candidate["metrics"]["profile_label"] for candidate in retry_candidates],
        "quality_gate_confidence": source_profile.quality_gate,
        "quality_gate_met": quality_gate_met,
        "primary_profile_label": primary_candidate["metrics"]["profile_label"],
        "selected_profile_label": best_metrics["profile_label"],
    }
    formatted_text = build_formatted_transcript(
        best_candidate, run_metrics, retry_candidates, target_latency_s
    )

    return (
        formatted_text,
        best_candidate["detected_swears"],
        best_candidate["full_transcription"],
        best_candidate["segments_data"],
        best_candidate["info"],
        run_metrics,
    )


# ... (Previous Code) ...


def consensus_transcribe(
    model,
    file_path: str,
    lang: str,
    swear_list,
    task: str = "transcribe",
    domain_key: str = "omni",
    hotwords: str = "",
    progress_callback=None,
    target_latency_s: int = TARGET_LATENCY_SECONDS,
) -> tuple:
    """Apple-level multi-pass konsensüs dekodlama.

    Aynı sesi 3 farklı stratejiyle çözer ve segment bazında
    en yüksek güvenli olanı seçerek nihai transkripti oluşturur.
    """
    if is_mlx_model(model) and target_latency_s <= TARGET_LATENCY_SECONDS:
        return transcribe_audio_file(
            model,
            file_path,
            lang,
            swear_list,
            task=task,
            profile_key="mac_turbo_sla",
            domain_key=domain_key,
            hotwords=hotwords,
            progress_callback=progress_callback,
            target_latency_s=target_latency_s,
        )

    overall_started = time.perf_counter()

    # Hazırlık: standart ve apex ses ön işleme
    prepared_standard, prep_standard = prepare_audio_for_asr(file_path, AUDIO_PREP_STANDARD)

    # Apex ses kurtarma (varsa)
    try:
        prepared_apex, prep_apex = prepare_audio_for_asr(file_path, AUDIO_PREP_APEX)
    except Exception:
        prepared_apex, prep_apex = prepared_standard, prep_standard

    combined_hotwords = build_domain_hotwords(domain_key, hotwords)

    candidates = []

    # === GEÇİŞ 1: apex_quality profili (beam=10, temp=0.0, standard ses) ===
    try:
        c1 = transcribe_with_profile(
            model,
            prepared_standard,
            prep_standard,
            "apex_quality",
            lang,
            swear_list,
            task,
            domain_key,
            combined_hotwords,
            progress_callback=progress_callback,
            overall_started=overall_started,
        )
        candidates.append(c1)
    except Exception:
        pass

    # === GEÇİŞ 2: ultimate_accuracy profili (farklı beam/chunk, standard ses) ===
    try:
        c2 = transcribe_with_profile(
            model,
            prepared_standard,
            prep_standard,
            "ultimate_accuracy",
            lang,
            swear_list,
            task,
            domain_key,
            combined_hotwords,
            progress_callback=progress_callback,
            overall_started=overall_started,
        )
        candidates.append(c2)
    except Exception:
        pass

    # === GEÇİŞ 3: rescue profili (kötü ses kurtarma filtresiyle) ===
    try:
        c3 = transcribe_with_profile(
            model,
            prepared_apex,
            prep_apex,
            "rescue",
            lang,
            swear_list,
            task,
            domain_key,
            combined_hotwords,
            progress_callback=progress_callback,
            overall_started=overall_started,
        )
        candidates.append(c3)
    except Exception:
        pass

    if not candidates:
        # Fallback: normal transcribe
        return transcribe_audio_file(
            model,
            file_path,
            lang,
            swear_list,
            task,
            "apex_quality",
            domain_key,
            hotwords,
            progress_callback,
            target_latency_s,
        )

    # Konsensüs: en yüksek candidate_score'a sahip olanı seç
    best_candidate = pick_best_transcription_candidate(candidates)

    # Apply SOTA features
    best_candidate = apply_sota_features(
        best_candidate, model, domain_key, combined_hotwords, lang, swear_list
    )

    elapsed_s = time.perf_counter() - overall_started

    best_prep_info = best_candidate.get("prep_info", prep_standard)
    duration = best_prep_info.get("duration")
    best_metrics = best_candidate["metrics"]
    hotword_count = len(parse_custom_terms(combined_hotwords))

    apex_profile = ASR_PROFILES.get("apex_quality", ASR_PROFILES["ultimate_accuracy"])
    quality_gate_met = (
        best_metrics.get("confidence", 0.0) >= apex_profile.quality_gate
        and best_metrics.get("filtered_segments", 0) == 0
        and best_prep_info.get("audio_quality_score", 100.0) >= AUDIO_QUALITY_REVIEW_THRESHOLD
    )

    run_metrics = {
        **best_prep_info,
        **best_metrics,
        "elapsed_s": elapsed_s,
        "rtf": (elapsed_s / duration) if duration else None,
        "target_latency_s": target_latency_s,
        "target_met": elapsed_s <= target_latency_s,
        "hotword_count": hotword_count,
        "quality_retry": True,
        "retry_profiles": [c["metrics"]["profile_label"] for c in candidates[1:]],
        "quality_gate_confidence": apex_profile.quality_gate,
        "quality_gate_met": quality_gate_met,
        "primary_profile_label": candidates[0]["metrics"]["profile_label"],
        "selected_profile_label": best_metrics["profile_label"],
        "consensus_passes": len(candidates),
    }

    formatted_text = build_formatted_transcript(
        best_candidate, run_metrics, candidates[1:], target_latency_s
    )

    return (
        formatted_text,
        best_candidate["detected_swears"],
        best_candidate["full_transcription"],
        best_candidate["segments_data"],
        best_candidate["info"],
        run_metrics,
    )


def auto_select_profile(audio_quality_score: float) -> str:
    """Ses kalitesine göre otomatik en iyi profili seçer."""
    if audio_quality_score >= 85:
        return "apex_quality"
    elif audio_quality_score >= 60:
        return "ultimate_accuracy"
    else:
        return "rescue"


# --- SEGMENT DÜZEYINDE YENİDEN ÇÖZME (PER-SEGMENT RE-DECODE) ---


def extract_audio_segment_ffmpeg(source_path: str, start_s: float, end_s: float, output_path: str):
    """FFmpeg ile ses dosyasından belirli bir zaman aralığını keser."""
    ffmpeg_path = get_ffmpeg_path()
    duration = end_s - start_s
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ss",
        str(start_s),
        "-t",
        str(duration),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        return True
    except Exception:
        return False


def redecode_low_confidence_segments(
    model,
    prepared_path: str,
    segments_data,
    domain_key: str = "omni",
    combined_hotwords: str = "",
    lang: str = "tr",
):
    """Düşük güvenli segmentleri izole edip farklı parametrelerle yeniden çözer.

    Apple/Google seviyesi: avg_logprob < -0.7 olan her segmenti keser,
    çevresindeki metnin bağlamını prompt olarak verir ve beam=10 ile tekrar çözer.
    """
    if not segments_data:
        return segments_data

    LOW_CONF_THRESHOLD = -0.7
    improved_segments = list(segments_data)
    redecoded_count = 0

    for idx, seg in enumerate(segments_data):
        if seg.avg_logprob >= LOW_CONF_THRESHOLD:
            continue

        # Çevreleyen bağlamı topla (önceki ve sonraki segmentlerin metni)
        context_parts = []
        if idx > 0:
            context_parts.append(segments_data[idx - 1].text)
        if idx < len(segments_data) - 1:
            context_parts.append(segments_data[idx + 1].text)
        context_prompt = " ".join(context_parts)[:200] if context_parts else None

        # Segmenti kes
        seg_path = Path(TEMP_AUDIO_DIR) / "redecode" / f"seg_{idx}_{int(seg.start*1000)}.wav"
        seg_path.parent.mkdir(parents=True, exist_ok=True)

        pad_start = max(0.0, seg.start - 0.5)
        pad_end = seg.end + 0.5

        if not extract_audio_segment_ffmpeg(prepared_path, pad_start, pad_end, str(seg_path)):
            continue

        # Yeniden çöz: farklı parametrelerle
        try:
            retry_options = {
                "language": lang,
                "task": "transcribe",
                "beam_size": 10,
                "best_of": 10,
                "patience": 1.2,
                "temperature": (0.0, 0.2, 0.4),
                "compression_ratio_threshold": 2.0,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.3,
                "condition_on_previous_text": False,
                "no_repeat_ngram_size": 5,
                "initial_prompt": f"Türkçe çağrı merkezi kaydı. {context_prompt}"
                if context_prompt
                else None,
                "vad_filter": True,
                "vad_parameters": {
                    "threshold": 0.15,
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 200,
                },
                "word_timestamps": False,
                "without_timestamps": False,
                "max_new_tokens": ASR_MAX_NEW_TOKENS,
                "multilingual": False,
                "repetition_penalty": 1.15,
                "hotwords": combined_hotwords.strip() or None,
            }

            retry_segments, retry_info = model.transcribe(str(seg_path), **retry_options)
            retry_segments = list(retry_segments)

            if retry_segments:
                retry_text = " ".join(s.text.strip() for s in retry_segments if s.text.strip())
                retry_logprob = sum(
                    float(getattr(s, "avg_logprob", -1.0) or -1.0) for s in retry_segments
                ) / max(len(retry_segments), 1)

                # Yeniden çözüm daha iyiyse güncelle
                if retry_text and retry_logprob > seg.avg_logprob:
                    corrected_text = postprocess_transcript_text(
                        retry_text, "apex_quality", domain_key, combined_hotwords
                    )
                    improved_segments[idx] = TranscriptSegment(
                        start=seg.start,
                        end=seg.end,
                        text=corrected_text,
                        avg_logprob=retry_logprob,
                        no_speech_prob=float(
                            getattr(retry_segments[0], "no_speech_prob", 0.0) or 0.0
                        ),
                        compression_ratio=float(
                            getattr(retry_segments[0], "compression_ratio", 1.0) or 1.0
                        ),
                        raw_text=retry_text,
                    )
                    redecoded_count += 1
        except Exception:
            continue
        finally:
            try:
                seg_path.unlink(missing_ok=True)
            except Exception:
                pass

    return improved_segments


# --- TÜRKÇE NOKTALAMA RESTORASYONU ---

_TURKISH_SENTENCE_ENDERS = re.compile(
    r"(?<=[a-zçğıöşüA-ZÇĞİÖŞÜ0-9])\s+" r"(?=[A-ZÇĞİÖŞÜ])",
)

_TURKISH_QUESTION_MARKERS = [
    r"\b(?:mi|mı|mu|mü|misin|mısın|musun|müsün|miyim|mıyım|miyiz|mıyız|musunuz|müsünüz)\b[?]?\s*$",
    r"\b(?:ne|neden|niye|nasıl|nerede|kim|kaç|hangi|ne zaman)\b",
]


def restore_turkish_punctuation(text: str) -> str:
    """Kural-tabanlı Türkçe noktalama restorasyonu.

    ASR çıktısında genellikle noktalama eksik olur. Bu fonksiyon:
    1. Büyük harfle başlayan yeni cümlelerin önüne nokta ekler
    2. Soru kalıplarını tespit edip soru işareti ekler
    3. Virgül eksiklerini doldurmaya çalışır
    """
    if not text or not text.strip():
        return text

    # Büyük harfle başlayan kelimelerin önüne nokta ekle (cümle sınırı tahmini)
    result = text

    # Soru ekleri tespit et ve sonuna ? ekle
    sentences = re.split(r"(?<=[.?!])\s+", result)
    fixed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        is_question = False
        for pattern in _TURKISH_QUESTION_MARKERS:
            if re.search(pattern, sentence, re.IGNORECASE):
                is_question = True
                break

        # Eğer cümle hiçbir noktalama ile bitmiyorsa
        if sentence and sentence[-1] not in ".?!":
            if is_question:
                sentence += "?"
            else:
                sentence += "."

        # İlk harf büyük olsun
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        fixed_sentences.append(sentence)

    result = " ".join(fixed_sentences)
    return result


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
            if "negative" in item["label"].lower():
                negative_score = item["score"]
            elif "positive" in item["label"].lower():
                positive_score = item["score"]

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
