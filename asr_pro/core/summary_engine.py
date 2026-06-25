"""Top-Tier CRM Auto-Note (Structured Summarization) Engine."""

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

# NLP Classifier nesnesini ASR'den almak için interface uyumu
try:
    from asr_pro.core.sentiment_engine import SentimentClassifier
except ImportError:
    SentimentClassifier = None


@dataclass
class CallSummary:
    intent: str  # Müşteri Ana Konu
    issue: str  # Sorun / Kök Neden
    action: str  # İşlem / Aksiyon
    resolution: str  # Sonuç Durumu
    executive_summary: str  # 1-2 Cümlelik Yönetici Özeti


def _extract_executive_summary(text: str, num_sentences: int = 2) -> str:
    """CPU dostu (Lightweight) Cümle Skorlama ve Özetleme."""
    # Basit cümle bölücü
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    if len(sentences) <= num_sentences:
        return text.strip()

    # Kelime frekanslarını hesapla (Stopword'leri kaba taslak elemek için 3 harften büyükleri al)
    words = re.findall(r"\b\w{4,}\b", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Cümleleri skorla
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        s_words = re.findall(r"\b\w{4,}\b", sentence.lower())
        score = sum(freq.get(w, 0) for w in s_words)
        # Uzun cümleleri hafif penalize et (normalize)
        if len(s_words) > 0:
            score = score / (len(s_words) ** 0.5)
        scored_sentences.append((score, i, sentence))

    # En yüksek skorlu cümleleri al ve orijinal sırasına göre diz
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    top_sentences = sorted(scored_sentences[:num_sentences], key=lambda x: x[1])

    return " ".join([s[2] for s in top_sentences])


def generate_crm_summary(full_text: str, classifier: Any = None) -> CallSummary:
    """Zero-Shot NLP ile Yapılandırılmış CRM Kapanış Notu Üretir."""

    if not full_text or len(full_text.strip()) < 10:
        return CallSummary(
            "Belirsiz", "Belirsiz", "Belirsiz", "Belirsiz", "Yetersiz konuşma verisi."
        )

    # Her zaman SentimentClassifier (Zero-Shot) kullanacağız, yanlışlıkla UI'dan toksisite modeli gelmiş olabilir.
    executive_summary = _extract_executive_summary(full_text)

    if classifier:
        zero_shot_classifier = classifier
    elif SentimentClassifier:
        zero_shot_classifier = SentimentClassifier.get_instance()
    else:
        zero_shot_classifier = None

    if not zero_shot_classifier:
        return CallSummary(
            "Bilinmiyor", "Bilinmiyor", "Bilinmiyor", "Bilinmiyor", executive_summary
        )

    # SADECE EN ANLAMLI KISMI (Örn son 500 kelime) ANALİZ ETKİ HIZLI ÇALIŞSIN
    words = full_text.split()
    if len(words) > 200:
        # Özet için başı ve sonu daha değerlidir. İlk 100 ve son 100 kelimeyi birleştir
        analysis_text = " ".join(words[:100] + words[-100:])
    else:
        analysis_text = full_text

    # 1. Müşteri (Intent)
    intent_labels = [
        "Fatura İtirazı",
        "Teknik Destek",
        "Üyelik İptali",
        "Bilgi Alma",
        "Şikayet ve Öneri",
    ]
    intent_res = zero_shot_classifier.predict(
        analysis_text, labels=intent_labels, hypothesis="Müşterinin ana talebi {} ile ilgilidir."
    )
    intent = intent_res["labels"][0]

    # 2. Sorun (Issue)
    issue_labels = [
        "Yanlış Ücretlendirme",
        "Sistem Hatası",
        "Kargo Gecikmesi",
        "Kullanıcı Hatası",
        "Kusurlu Ürün",
        "İletişim Eksikliği",
    ]
    issue_res = zero_shot_classifier.predict(
        analysis_text, labels=issue_labels, hypothesis="Bu çağrıdaki kök neden {} kaynaklıdır."
    )
    issue = issue_res["labels"][0]

    # 3. İşlem (Action)
    action_labels = [
        "Kredi Tanımlandı",
        "İade Başlatıldı",
        "Talep Oluşturuldu",
        "Bilgi Verildi",
        "Şifre Sıfırlandı",
        "İşlem Yapılmadı",
    ]
    action_res = zero_shot_classifier.predict(
        analysis_text,
        labels=action_labels,
        hypothesis="Müşteri temsilcisinin yaptığı işlem {} olmuştur.",
    )
    action = action_res["labels"][0]

    # 4. Sonuç Durumu (Resolution)
    res_labels = ["Çözüldü", "Çözülemedi", "Beklemede", "Üst Birime Aktarıldı"]
    res_hyp = zero_shot_classifier.predict(
        analysis_text,
        labels=res_labels,
        hypothesis="Çağrının final durumu {} olarak sonuçlanmıştır.",
    )
    resolution = res_hyp["labels"][0]

    return CallSummary(
        intent=intent,
        issue=issue,
        action=action,
        resolution=resolution,
        executive_summary=executive_summary,
    )


def generate_ollama_summary(
    full_text: str, model_name: str = "llama3", classifier: Any = None
) -> CallSummary:
    """
    Yerel Ollama sunucusunu (http://localhost:11434) kullanarak %100 gizli ve internetsiz özet üretir.
    Hata durumunda veya Ollama kapalıysa anında zero-shot motora dönüş (Fallback) yapar.
    """
    if model_name == "Kapalı (Sadece Yerel Motor)":
        return generate_crm_summary(full_text, classifier)

    system_prompt = """
    Sen Apple seviyesinde kıdemli bir Müşteri Hizmetleri Yöneticisisin. Sana verilen müşteri temsilcisi çağrı deşifresini analiz edip sadece aşağıdaki JSON formatında çıktı vereceksin.
    Lütfen executive_summary kısmına olayı çok iyi özetleyen, harika akıcılıkta, 2 cümlelik profesyonel bir edebi yönetici özeti yaz.

    Format:
    {
        "intent": "Fatura, Teknik Destek vs.",
        "issue": "Kök neden (Örn: Sistem Hatası)",
        "action": "Temsilcinin yaptığı işlem",
        "resolution": "Çözüldü / Çözülemedi / Beklemede",
        "executive_summary": "Kusurlu akıcılıkta paragraf..."
    }
    """

    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model_name,
        "prompt": system_prompt + "\n\nÇağrı Metni:\n" + full_text[:3000],
        "stream": False,
        "format": "json",
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()

            content = result.get("response", "{}")
            parsed = json.loads(content)

            return CallSummary(
                intent=parsed.get("intent", "Bilinmiyor"),
                issue=parsed.get("issue", "Bilinmiyor"),
                action=parsed.get("action", "Bilinmiyor"),
                resolution=parsed.get("resolution", "Bilinmiyor"),
                executive_summary=parsed.get("executive_summary", "Özet çıkarılamadı."),
            )
    except Exception as e:
        logger.warning(f"Ollama Bağlantı Hatası: {e}. Standart (Zero-Shot) motora geçiliyor...")
        return generate_crm_summary(full_text, classifier)
