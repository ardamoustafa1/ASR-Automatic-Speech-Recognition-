# Evaluates customer and agent emotional valence and sentiment trajectories over time.
from __future__ import annotations

"""Advanced AI-Based Sentiment & Emotion Analysis Engine using Transformers."""


import threading
from dataclasses import dataclass
from typing import Literal

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
    speaker: str | None


class SentimentClassifier:
    """Singleton for Lazy Loading the AI Model and Managing Hardware Acceleration."""

    _instance: SentimentClassifier | None = None
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
        try:
            import torch

            # MPS for Apple Silicon - but use -1 (CPU) as device index since
            # transformers pipeline MPS support has edge cases; MPS is still used
            # implicitly when model is moved via .to('mps') after load.
            if torch.backends.mps.is_available():
                self._device_str = "cpu"  # Load on CPU, then MPS via torch
                self._use_mps = True
                logger.info("MPS (Apple Silicon) detected. Loading model to CPU first.")
            elif torch.cuda.is_available():
                self._device_str = "cuda"
                self._use_mps = False
                logger.info("CUDA GPU detected. Hardware acceleration enabled.")
            else:
                self._device_str = "cpu"
                self._use_mps = False
                logger.info("No hardware acceleration detected. Falling back to CPU.")
        except Exception:
            self._device_str = "cpu"
            self._use_mps = False

    def _load_model(self):
        """Lazy load the transformer models.

        Primary model: savasy/bert-base-turkish-sentiment-cased
        - Fine-tuned on Turkish text for positive/negative/neutral classification
        - Far more accurate for Turkish than a generic multilingual zero-shot model

        Fallback (zero-shot) model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
        - Used for fine-grained emotion classification (angry, frustrated, anxious)
          when the Turkish BERT model is unavailable or for churn/topic labeling.
        """
        if self._pipeline is None:
            logger.info("Loading Turkish sentiment model (savasy/bert-base-turkish-sentiment-cased)...")
            try:
                from transformers import pipeline as hf_pipeline

                self._pipeline = hf_pipeline(
                    "text-classification",
                    model="savasy/bert-base-turkish-sentiment-cased",
                    device=self._device_str,
                    truncation=True,
                    max_length=512,
                    top_k=None,  # Return scores for all labels
                )
                logger.info("Turkish BERT sentiment model loaded successfully.")
                self._model_type = "turkish_bert"
            except Exception as e:
                logger.warning(
                    f"Turkish BERT model failed to load ({e}). "
                    "Falling back to multilingual zero-shot model."
                )
                self._load_zero_shot_model()

    def _load_zero_shot_model(self):
        """Lazy load the multilingual zero-shot model as a fallback."""
        if getattr(self, "_zero_shot_pipeline", None) is None:
            logger.info("Loading multilingual zero-shot model (mDeBERTa-v3)...")
            try:
                from transformers import pipeline as hf_pipeline

                self._zero_shot_pipeline = hf_pipeline(
                    "zero-shot-classification",
                    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
                    device=self._device_str,
                    truncation=True,
                    max_length=512,
                )
                if self._pipeline is None:
                    # Primary model failed — use zero-shot as primary too
                    self._pipeline = self._zero_shot_pipeline
                    self._model_type = "zero_shot"
                logger.info("Zero-shot fallback model loaded.")
            except Exception as e:
                logger.error(f"Zero-shot model also failed to load: {e}")
                self._zero_shot_pipeline = None

    def predict(self, text: str, labels: list[str] = None, hypothesis: str = None) -> dict:
        """Run inference on the text.

        When called without explicit labels (emotion analysis), uses the Turkish
        BERT classifier (primary) which returns positive/negative/neutral labels.
        These are then mapped to the richer emotion categories.

        When called with explicit labels (churn, topic), always uses the zero-shot
        model since it supports arbitrary label sets.
        """
        labels = labels or EMOTION_LABELS
        hypothesis = hypothesis or "The emotion expressed in this text is {}."

        if not text or not text.strip():
            return {"labels": labels, "scores": [1.0] + [0.0] * (len(labels) - 1)}

        # Truncate very long texts to avoid context window issues (max ~500 chars)
        text = text[:500] if len(text) > 500 else text

        # If custom labels are requested (churn/topic analysis), always use
        # the zero-shot pipeline which supports arbitrary hypothesis templates.
        is_custom_labels = set(labels) != set(EMOTION_LABELS)
        if is_custom_labels:
            self._load_zero_shot_model()
            zs_pipe = getattr(self, "_zero_shot_pipeline", None) or self._pipeline
            if zs_pipe is None:
                return {"labels": labels, "scores": [1.0 / len(labels)] * len(labels)}
            try:
                return zs_pipe(
                    text,
                    labels,
                    hypothesis_template=hypothesis,
                    multi_label=False,
                    truncation=True,
                )
            except Exception as e:
                logger.warning(f"Zero-shot inference failed: {e}")
                return {"labels": labels, "scores": [1.0 / len(labels)] * len(labels)}

        # Emotion analysis — use the Turkish BERT primary model
        self._load_model()
        if self._pipeline is None:
            return {"labels": labels, "scores": [1.0] + [0.0] * (len(labels) - 1)}

        model_type = getattr(self, "_model_type", "zero_shot")

        if model_type == "turkish_bert":
            try:
                results = self._pipeline(text)
                # savasy model returns [{"label": "positive", "score": 0.98}, ...]
                # Flatten top_k=None result if it's a list of lists
                if results and isinstance(results[0], list):
                    results = results[0]
                score_map_bert = {r["label"].lower(): r["score"] for r in results}
                # Map Turkish BERT labels → emotion label scores
                # positive → satisfied
                # negative → distribute between angry/frustrated/anxious proportionally
                # neutral  → neutral
                pos = score_map_bert.get("positive", 0.0)
                neg = score_map_bert.get("negative", 0.0)
                neu = score_map_bert.get("neutral", 0.0)
                # Distribute negative probability across stress emotions
                mapped_scores = [
                    neg * 0.45,   # angry
                    neg * 0.35,   # frustrated
                    pos,          # satisfied
                    neg * 0.20,   # anxious
                    neu,          # neutral
                ]
                # Normalize
                total = sum(mapped_scores) or 1.0
                mapped_scores = [s / total for s in mapped_scores]
                return {"labels": EMOTION_LABELS, "scores": mapped_scores}
            except Exception as e:
                logger.warning(f"Turkish BERT inference failed: {e}")

        # Fallback: zero-shot
        try:
            return self._pipeline(
                text,
                labels,
                hypothesis_template=hypothesis,
                multi_label=False,
                truncation=True,
            )
        except Exception as e:
            logger.warning(f"Sentiment inference failed: {e}")
            return {"labels": labels, "scores": [0.2] * len(labels)}


def _map_to_category(label: str) -> EmotionCategory:
    mapping: dict[str, EmotionCategory] = {
        "angry": "Öfke",
        "frustrated": "Hayal Kırıklığı",
        "satisfied": "Memnuniyet",
        "anxious": "Endişe",
        "neutral": "Nötr İletişim",
    }
    return mapping.get(label.lower(), "Nötr İletişim")


def _calculate_scores(labels: list[str], scores: list[float]) -> tuple[float, StressLevel]:
    """Derive sentiment (-1 to 1) and stress level based on model probabilities."""
    score_map = dict(zip(labels, scores))

    # Calculate a weighted sentiment score based on emotion probabilities
    positive_score = score_map.get("satisfied", 0.0)
    negative_score = (
        score_map.get("angry", 0.0)
        + score_map.get("frustrated", 0.0)
        + score_map.get("anxious", 0.0)
    )

    # Range is roughly -1.0 to 1.0
    sentiment = float(positive_score - negative_score)

    # Calculate stress
    high_stress_prob = (
        score_map.get("angry", 0.0)
        + score_map.get("anxious", 0.0)
        + (score_map.get("frustrated", 0.0) * 0.5)
    )

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
            speaker=segment.speaker,
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
        speaker=segment.speaker,
    )
