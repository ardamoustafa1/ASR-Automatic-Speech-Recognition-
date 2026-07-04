from asr_pro.core.keyword_engine import KeywordHitResult
from asr_pro.core.topic_classifier import TopicInput, classify_topics_from_hits


def test_classify_topics_empty():
    res = classify_topics_from_hits([], [])
    assert len(res) == 0


def test_classify_topics_matches():
    hit = KeywordHitResult(
        "1",
        "rule",
        "app",
        "mobil uygulama çöküyor",
        "exact",
        0.9,
        0.0,
        0,
        "customer",
        "",
        "info",
        None,
    )
    topic = TopicInput(id="t1", slug="crash", label_tr="Çökme", seed_keywords=("app",))
    res = classify_topics_from_hits([hit], [topic])
    assert len(res) == 1
    assert res[0].topic_id == "t1"
