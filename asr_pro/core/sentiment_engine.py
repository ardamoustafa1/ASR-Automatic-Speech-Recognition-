from __future__ import annotations
"""Advanced AI-Based Sentiment & Emotion Analysis Engine using Transformers."""


import threading
from dataclasses import dataclass
from typing import Optional, Literal

from loguru import logger

from asr_pro.core.keyword_engine import SegmentInput

EmotionCategory = Literal["Öfke", "Hayal Kırıklığı", "Memnuniyet", "Endişe", "Nötr İletişim"]
StressLevel = Literal["Düşük", "Normal", "Yüksek"]

# Valid candidate labels for the Zero-Shot Classifier
# Using English labels for mDeBERTa as it often aligns better with its pre-training
EMOTION_LABELS = ["angry", "frustrated", "satisfied", "anxious", "neutral"]


@dataclass(frozen=True)
class SentimentResult:
    sentiment_score: float  # -1.0 (Most Negative) to +1.0 (Most Positive)
    emotion_category: EmotionCategory
    stress_level: StressLevel
    confidence: float
    segment_index: int
    speaker: Optional[str]



class SentimentClassifier:
    """Singleton for Lazy Loading the AI Model and Managing Hardware Acceleration."""
    _instance: Optional[SentimentClassifier] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._pipeline = None
        self._device_str = "cpu"
        self._setup_device()

    @classmethod
    def get_instance(cls) -> SentimentClassifier:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _setup_device(self):
        """Determine the best available hardware accelerator (Apple Silicon MPS, CUDA, or CPU)."""
        import torch
        if torch.backends.mps.is_available():
            self._device_str = "mps"
            logger.info("MPS (Apple Silicon Neural Engine) detected. Hardware acceleration enabled.")
        elif torch.cuda.is_available():
            self._device_str = "cuda"
            logger.info("CUDA GPU detected. Hardware acceleration enabled.")
        else:
            self._device_str = "cpu"
            logger.info("No hardware acceleration detected. Falling back to CPU.")

    def _load_model(self):
        """Lazy load the transformer model."""
        if self._pipeline is None:
            logger.info("Loading Zero-Shot NLP Model (mDeBERTa) into memory...")
            from transformers import pipeline
            # We use a robust multilingual zero-shot model for accurate Turkish emotion detection
            model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=self._device_str
            )
            logger.info("Model loaded successfully.")

    def predict(self, text: str, labels: list[str] = None, hypothesis: str = None) -> dict:
        """Run inference on the text."""
        labels = labels or EMOTION_LABELS
        hypothesis = hypothesis or "The emotion expressed in this text is {}."
        
        if not text or not text.strip():
            return {"labels": labels, "scores": [1.0] + [0.0]*(len(labels)-1)}
            
        self._load_model()
        return self._pipeline(text, labels, hypothesis_template=hypothesis, multi_label=False)


def _map_to_category(label: str) -> EmotionCategory:
    mapping: dict[str, EmotionCategory] = {
        "angry": "Öfke",
        "frustrated": "Hayal Kırıklığı",
        "satisfied": "Memnuniyet",
        "anxious": "Endişe",
        "neutral": "Nötr İletişim"
    }
    return mapping.get(label.lower(), "Nötr İletişim")


def _calculate_scores(labels: list[str], scores: list[float]) -> tuple[float, StressLevel]:
    """Derive sentiment (-1 to 1) and stress level based on model probabilities."""
    score_map = dict(zip(labels, scores))
    
    # Calculate a weighted sentiment score based on emotion probabilities
    positive_score = score_map.get("satisfied", 0.0)
    negative_score = score_map.get("angry", 0.0) + score_map.get("frustrated", 0.0) + score_map.get("anxious", 0.0)
    
    # Range is roughly -1.0 to 1.0
    sentiment = float(positive_score - negative_score)
    
    # Calculate stress
    high_stress_prob = score_map.get("angry", 0.0) + score_map.get("anxious", 0.0) + (score_map.get("frustrated", 0.0) * 0.5)
    
    if high_stress_prob > 0.45:
        stress = "Yüksek"
    elif score_map.get("satisfied", 0.0) > 0.4 or score_map.get("neutral", 0.0) > 0.5:
        stress = "Düşük"
    else:
        stress = "Normal"
        
    return round(sentiment, 3), stress


def analyze_sentiment(segment: SegmentInput) -> SentimentResult:
    """Analyze the emotional tone of a segment using the AI Transformer Model."""
    text = segment.text or ""
    if not text.strip():
        return SentimentResult(
            sentiment_score=0.0,
            emotion_category="Nötr İletişim",
            stress_level="Normal",
            confidence=1.0,
            segment_index=segment.segment_index,
            speaker=segment.speaker
        )

    # 1. Run inference using the Singleton classifier
    classifier = SentimentClassifier.get_instance()
    result = classifier.predict(text)

    # 2. Extract best match
    best_label = result["labels"][0]
    best_score = result["scores"][0]
    
    # 3. Derive secondary metrics
    sentiment_score, stress_level = _calculate_scores(result["labels"], result["scores"])
    category = _map_to_category(best_label)

    return SentimentResult(
        sentiment_score=sentiment_score,
        emotion_category=category,
        stress_level=stress_level,
        confidence=round(float(best_score), 3),
        segment_index=segment.segment_index,
        speaker=segment.speaker
    )
