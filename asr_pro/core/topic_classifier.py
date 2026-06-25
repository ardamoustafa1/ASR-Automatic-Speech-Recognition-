from __future__ import annotations

"""Lightweight topic classification from keyword hits and seed synonyms."""


from collections.abc import Sequence
from dataclasses import dataclass

from asr_pro.core.keyword_engine import SEMANTIC_SYNONYMS, KeywordHitResult, normalize_token


@dataclass(frozen=True)
class TopicInput:
    id: str
    slug: str
    label_tr: str
    seed_keywords: tuple[str, ...]
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class TopicMatch:
    topic_id: str
    slug: str
    label_tr: str
    confidence: float
    matched_via: str


def classify_topics_from_hits(
    hits: Sequence[KeywordHitResult],
    topics: Sequence[TopicInput],
) -> list[TopicMatch]:
    if not hits or not topics:
        return []

    results: list[TopicMatch] = []
    seen: set[str] = set()

    keyword_to_topic: dict[str, TopicInput] = {}
    for topic in topics:
        for kw in topic.seed_keywords:
            keyword_to_topic[normalize_token(kw)] = topic
        for syn in topic.synonyms:
            keyword_to_topic[normalize_token(syn)] = topic
        for key, syns in SEMANTIC_SYNONYMS.items():
            if normalize_token(key) in {normalize_token(k) for k in topic.seed_keywords}:
                for syn in syns:
                    keyword_to_topic[normalize_token(syn)] = topic

    for hit in hits:
        norm = normalize_token(hit.keyword)
        topic = keyword_to_topic.get(norm) or keyword_to_topic.get(
            normalize_token(hit.matched_text)
        )
        if topic and topic.id not in seen:
            seen.add(topic.id)
            results.append(
                TopicMatch(
                    topic_id=topic.id,
                    slug=topic.slug,
                    label_tr=topic.label_tr,
                    confidence=hit.confidence,
                    matched_via=hit.matched_text,
                )
            )
        elif hit.topic_id and hit.topic_id not in seen:
            seen.add(hit.topic_id)
            for t in topics:
                if t.id == hit.topic_id:
                    results.append(
                        TopicMatch(
                            topic_id=t.id,
                            slug=t.slug,
                            label_tr=t.label_tr,
                            confidence=hit.confidence,
                            matched_via=hit.matched_text,
                        )
                    )
                    break

    return results
