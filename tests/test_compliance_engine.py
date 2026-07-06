from asr_pro.core.compliance_engine import analyze_compliance_risk
from asr_pro.core.keyword_engine import SegmentInput


def test_finance_red_flag_fuzzy():
    segments = [
        SegmentInput(start=0.0, end=2.0, text="Merhaba hoş geldiniz.", segment_index=0),
        SegmentInput(
            start=2.0,
            end=5.0,
            text="Bu yatırım size kesin kazandırır merak etmeyin.",
            segment_index=1,
        ),
        SegmentInput(
            start=5.0, end=8.0, text="Kredi kart şifrenizi bana okur musunuz?", segment_index=2
        ),
    ]

    violations = analyze_compliance_risk(segments, domain_key="finance", use_ai=False)

    assert len(violations) == 2
    assert violations[0].severity == "CRITICAL"
    assert "Yanıltıcı Yatırım Vaadi" in violations[0].category

    assert violations[1].severity == "CRITICAL"
    assert "Veri Gizliliği" in violations[1].category


def test_negation_false_positive_shield():
    segments = [
        SegmentInput(
            start=0.0,
            end=3.0,
            text="Bizim fonlarımızda sıfır risk diye bir şey yoktur efendim.",
            segment_index=0,
        ),
        SegmentInput(
            start=3.0, end=6.0, text="Yatırımda kesin kazandırır demek yasaktır.", segment_index=1
        ),
    ]

    violations = analyze_compliance_risk(segments, domain_key="finance", use_ai=False)

    assert len(violations) == 0


def test_telecom_red_flag():
    segments = [
        SegmentInput(start=0.0, end=2.0, text="Taahhüt bozamazsınız hanımefendi.", segment_index=0)
    ]

    violations = analyze_compliance_risk(segments, domain_key="telecom", use_ai=False)

    assert len(violations) == 1


def test_clean_conversation():
    segments = [
        SegmentInput(start=0.0, end=3.0, text="Fiyatlarımız aylık yüz liradır.", segment_index=0)
    ]

    violations = analyze_compliance_risk(segments, domain_key="finance", use_ai=False)
    assert len(violations) == 0


def test_negation_filter_complex():
    segments = [
        SegmentInput(
            start=0.0, end=3.0, text="Size bu konuda maalesef garanti vermiyoruz.", segment_index=0
        ),
    ]
    violations = analyze_compliance_risk(segments, domain_key="finance", use_ai=False)
    assert len(violations) == 0


def test_domain_switching():
    segments = [
        SegmentInput(start=0.0, end=3.0, text="Bu tedavi sizi kesin iyileştirir.", segment_index=0)
    ]
    finance_v = analyze_compliance_risk(segments, domain_key="finance", use_ai=False)
    assert len(finance_v) == 0

    health_v = analyze_compliance_risk(segments, domain_key="health", use_ai=False)
    assert len(health_v) == 1
    assert "Umut Tacirliği" in health_v[0].category


def test_ai_confidence_gate():
    segments = [
        SegmentInput(start=0.0, end=3.0, text="Bu hisse senedi kesin kazandırır.", segment_index=0)
    ]

    violations_no_ai = analyze_compliance_risk(segments, domain_key="finance", use_ai=False)
    assert len(violations_no_ai) > 0
    assert violations_no_ai[0].severity == "CRITICAL"


def test_customer_speech_excluded_when_agent_id_provided():
    """Regulatory obligations apply to the AGENT, not the customer - a customer
    quoting/asking about a red-flag phrase must not be scored as a violation
    when agent_speaker_id is supplied to isolate the check."""
    segments = [
        SegmentInput(
            start=0.0,
            end=3.0,
            text="Bana kredi kartı şifremi mi soracaksınız şimdi?",
            segment_index=0,
            speaker="SPEAKER_00",  # customer
        ),
        SegmentInput(
            start=3.0,
            end=6.0,
            text="Hayır efendim, asla şifrenizi sormayız.",
            segment_index=1,
            speaker="SPEAKER_01",  # agent
        ),
    ]

    violations = analyze_compliance_risk(
        segments, domain_key="finance", use_ai=False, agent_speaker_id="SPEAKER_01"
    )

    assert len(violations) == 0


def test_agent_violation_still_caught_with_speaker_filter():
    segments = [
        SegmentInput(
            start=0.0,
            end=3.0,
            text="Merhaba, nasıl yardımcı olabilirim?",
            segment_index=0,
            speaker="SPEAKER_01",  # agent
        ),
        SegmentInput(
            start=3.0,
            end=6.0,
            text="Kredi kart şifrenizi bana okur musunuz?",
            segment_index=1,
            speaker="SPEAKER_01",  # agent
        ),
    ]

    violations = analyze_compliance_risk(
        segments, domain_key="finance", use_ai=False, agent_speaker_id="SPEAKER_01"
    )

    assert len(violations) == 1
    assert "Veri Gizliliği" in violations[0].category
