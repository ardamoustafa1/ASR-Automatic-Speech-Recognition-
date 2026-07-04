from asr_pro.core.topic_classifier import classify_topics_from_hits


def test_classify_topics_from_hits():
    # Empty hit list
    res = classify_topics_from_hits([], None)
    assert res == []
