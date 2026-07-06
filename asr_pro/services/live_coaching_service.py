"""Live Coaching Service for real-time WebSocket agent supervision and guidance.

Monitors incoming audio streaming chunks and partial hypotheses to trigger instant pop-up notifications
for interruptions, customer escalation/frustration, and compliance adherence reminders.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger("asr_pro.services.live_coaching_service")

# Escalation / Frustration trigger phrases
ESCALATION_PHRASES = [
    "iptal etmek",
    "şikayetçiyim",
    "faturam yüksek",
    "yetkiliyle",
    "mahkemeye",
    "avukata",
    "bağlayamadınız",
    "rezalet",
    "soygun",
]

# Compliance triggers
COMPLIANCE_GREETINGS = ["hoş geldiniz", "vodafone", "nasıl yardımcı", "iyi günler"]
COMPLIANCE_KVKK = ["kayıt altına", "güvenlik amacıyla", "kvkk"]


class LiveCoachingService:
    """Real-time streaming agent supervision and coaching engine."""

    # In-memory session state tracking: { session_id: { "has_greeted": bool, "has_kvkk": bool, "last_speaker": str, "interruption_count": int } }
    _session_states: dict[str, dict[str, Any]] = {}

    @classmethod
    def get_session_state(cls, session_id: str) -> dict[str, Any]:
        if session_id not in cls._session_states:
            cls._session_states[session_id] = {
                "has_greeted": False,
                "has_kvkk": False,
                "last_speaker": None,
                "interruption_count": 0,
                "alerts_emitted": set(),
            }
        return cls._session_states[session_id]

    @classmethod
    def clear_session(cls, session_id: str) -> None:
        if session_id in cls._session_states:
            del cls._session_states[session_id]

    @classmethod
    def evaluate_chunk(
        cls,
        session_id: str,
        text: str,
        speaker: str | None = None,
        latency_ms: int = 0,
        session_elapsed: float = 0.0,
        is_interruption: bool = False,
    ) -> dict[str, Any] | None:
        """Evaluate a transcript chunk and return a live coaching alert if triggered."""
        if not text:
            return None

        state = cls.get_session_state(session_id)
        text_lower = text.lower().strip()

        # 1. Check Escalation / Churn Risk
        for phrase in ESCALATION_PHRASES:
            if phrase in text_lower and f"esc_{phrase}" not in state["alerts_emitted"]:
                state["alerts_emitted"].add(f"esc_{phrase}")
                logger.info(f"LiveCoaching [{session_id}]: Triggered escalation coaching for '{phrase}'")
                return {
                    "id": str(uuid.uuid4()),
                    "type": "escalation",
                    "title": "😡 Öfke & İtiraz Uyarı",
                    "message": f"Müşteri '{phrase}' talebinde bulunuyor. Lütfen 'Cayma Bedeli Sıfırlama ve Tutma Ekranı'nı açınız.",
                    "severity": "high",
                    "timestamp_sec": round(session_elapsed, 1),
                }

        # 2. Check Interruption
        if is_interruption:
            state["interruption_count"] += 1
            if state["interruption_count"] >= 2 and "int_warn" not in state["alerts_emitted"]:
                state["alerts_emitted"].add("int_warn")
                logger.info(f"LiveCoaching [{session_id}]: Triggered interruption warning.")
                return {
                    "id": str(uuid.uuid4()),
                    "type": "interruption",
                    "title": "⚡ Söz Kesme Uyarısı",
                    "message": "Müşterinin sözünü üst üste kestiniz. Lütfen aktif dinlemede kalın ve empati cümlesi kullanın.",
                    "severity": "warning",
                    "timestamp_sec": round(session_elapsed, 1),
                }

        # 3. Track Compliance disclosures
        if any(g in text_lower for g in COMPLIANCE_GREETINGS):
            state["has_greeted"] = True
        if any(k in text_lower for k in COMPLIANCE_KVKK):
            state["has_kvkk"] = True

        # 4. If elapsed > 15s and no greeting or KVKK yet, trigger compliance reminder
        if session_elapsed > 15.0 and not state["has_kvkk"] and "kvkk_remind" not in state["alerts_emitted"]:
            state["alerts_emitted"].add("kvkk_remind")
            logger.info(f"LiveCoaching [{session_id}]: Triggered KVKK reminder.")
            return {
                "id": str(uuid.uuid4()),
                "type": "compliance",
                "title": "💡 Kurumsal Uyum Hatırlatması",
                "message": "Görüşmenin 15. saniyesine ulaştınız. Zorunlu KVKK ses kaydı onay bildirimini yapmayı unutmayınız.",
                "severity": "info",
                "timestamp_sec": round(session_elapsed, 1),
            }

        return None
