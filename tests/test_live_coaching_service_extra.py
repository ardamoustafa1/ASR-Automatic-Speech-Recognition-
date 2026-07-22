"""Additional coverage for LiveCoachingService: interruption alerts, compliance
disclosure tracking, KVKK reminders, and empty-input/clear-session edge cases.
"""

from __future__ import annotations

from asr_pro.services.live_coaching_service import LiveCoachingService


def test_evaluate_chunk_returns_none_for_empty_text():
    assert LiveCoachingService.evaluate_chunk(session_id="s-empty", text="") is None


def test_clear_session_is_a_noop_for_unknown_session():
    LiveCoachingService.clear_session("never-existed")  # must not raise


def test_interruption_alert_fires_on_second_interruption():
    session_id = "s-interrupt"
    LiveCoachingService.clear_session(session_id)

    first = LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="tamam", is_interruption=True
    )
    assert first is None  # only 1 interruption so far, threshold is 2

    second = LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="anladım", is_interruption=True
    )
    assert second is not None
    assert second["type"] == "interruption"

    # Duplicate suppression: a third interruption must not re-fire the same alert.
    third = LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="peki", is_interruption=True
    )
    assert third is None


def test_compliance_disclosures_are_tracked_on_state():
    session_id = "s-compliance"
    LiveCoachingService.clear_session(session_id)

    LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="Hoş geldiniz Vodafone, iyi günler", session_elapsed=1.0
    )
    state = LiveCoachingService.get_session_state(session_id)
    assert state["has_greeted"] is True
    assert state["has_kvkk"] is False

    LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="Görüşme kayıt altına alınmaktadır", session_elapsed=2.0
    )
    state = LiveCoachingService.get_session_state(session_id)
    assert state["has_kvkk"] is True


def test_kvkk_reminder_fires_after_15_seconds_without_disclosure():
    session_id = "s-kvkk-reminder"
    LiveCoachingService.clear_session(session_id)

    alert = LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="evet tabii anladım", session_elapsed=16.0
    )
    assert alert is not None
    assert alert["type"] == "compliance"

    # Must not fire twice for the same session.
    alert2 = LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="peki tamam", session_elapsed=20.0
    )
    assert alert2 is None


def test_kvkk_reminder_does_not_fire_once_kvkk_disclosed():
    session_id = "s-kvkk-disclosed"
    LiveCoachingService.clear_session(session_id)

    LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="kayıt altına alınmaktadır", session_elapsed=1.0
    )
    alert = LiveCoachingService.evaluate_chunk(
        session_id=session_id, text="evet devam edelim", session_elapsed=20.0
    )
    assert alert is None
