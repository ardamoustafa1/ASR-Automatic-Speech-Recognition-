from unittest.mock import MagicMock, patch

from asr_pro.core.summary_engine import (
    _extract_executive_summary,
    generate_crm_summary,
    generate_ollama_summary,
)


class MockClassifier:
    def predict(self, text, labels, hypothesis):
        return {"labels": [labels[0]], "scores": [0.99]}


def test_executive_summary_short_text():
    text = "Merhaba efendim. Ben size faturanızı düzelttiğimi bildirmek isterim. İyi günler."
    summary = _extract_executive_summary(text)
    assert "faturanızı düzelttiğimi bildirmek isterim" in summary


def test_executive_summary_long_text():
    text = """
    Müşteri hizmetlerine hoş geldiniz. İsmim Ayşe, size nasıl yardımcı olabilirim?
    Evet anlıyorum, faturanızda beklentinizin üzerinde bir rakam gelmiş.
    Bunun sebebi yurt dışı dolaşım paketini açık bırakmış olmanızdır.
    Durumu inceledim ve mağduriyetinizi gidermek adına faturanızdan 500 TL iade işlemini başlattım.
    Başka bir sorunuz var mıydı efendim? Peki, iyi günler dileriz.
    """
    summary = _extract_executive_summary(text, num_sentences=2)
    assert "iade" in summary.lower() or "dolaşım" in summary.lower()


def test_generate_crm_summary_empty():
    res = generate_crm_summary("")
    assert res.intent == "Belirsiz"
    assert res.executive_summary == "Yetersiz konuşma verisi."


@patch("asr_pro.core.summary_engine.SentimentClassifier")
def test_generate_crm_summary_no_classifier(mock_sentiment_classifier):
    mock_sentiment_classifier.get_instance.return_value = None
    text = "Faturamda yanlışlık var. Hemen düzelttik."
    summary = generate_crm_summary(text, classifier=None)
    assert summary.intent == "Bilinmiyor"
    assert summary.issue == "Bilinmiyor"
    assert summary.action == "Bilinmiyor"
    assert summary.resolution == "Bilinmiyor"
    assert "Faturamda" in summary.executive_summary


def test_generate_crm_summary_with_classifier():
    text = "Faturamda fazla ücret var, lütfen düzeltin. İşlem yapıldı ve para iadesi onaylandı."
    classifier = MockClassifier()
    res = generate_crm_summary(text, classifier=classifier)
    assert res.intent == "Fatura İtirazı"  # the first label in the code's list
    assert res.issue == "Fiyat veya Tarife Anlaşmazlığı"
    assert res.action == "Tarife veya Paket Değişikliği Yapıldı"
    assert res.resolution == "Çözüldü"


@patch("httpx.Client.post")
def test_generate_ollama_summary_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": '{"intent": "Ollama Intent", "issue": "Ollama Issue", "action": "Ollama Action", "resolution": "Ollama Resolution", "executive_summary": "Ollama summary"}'
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    res = generate_ollama_summary("Test conversation with Ollama.", model_name="llama3")
    assert res.intent == "Ollama Intent"
    assert res.issue == "Ollama Issue"
    assert res.resolution == "Ollama Resolution"
    assert res.action == "Ollama Action"
    assert res.executive_summary == "Ollama summary"


@patch("httpx.Client.post")
def test_generate_ollama_summary_fallback(mock_post):
    # Mock Ollama failure
    mock_post.side_effect = Exception("Connection Refused")

    classifier = MockClassifier()
    res = generate_ollama_summary(
        "Müşteri aradı ve iade istedi.", model_name="llama3", classifier=classifier
    )
    # Fallbacks to standard CRM summary which uses the classifier returning the first label
    assert res.intent == "Fatura İtirazı"
    assert "iade" in res.executive_summary


def test_generate_ollama_summary_local_only():
    classifier = MockClassifier()
    res = generate_ollama_summary(
        "Müşteri aradı", model_name="Kapalı (Sadece Yerel Motor)", classifier=classifier
    )
    assert res.intent == "Fatura İtirazı"
