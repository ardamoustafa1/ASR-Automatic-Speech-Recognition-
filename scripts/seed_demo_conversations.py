import datetime
import secrets

from sqlalchemy.orm import Session

from asr_pro.api.deps import get_db
from asr_pro.db.models import Conversation, TranscriptSegmentRow, new_uuid


def seed_demo_conversations():
    db: Session = next(get_db())

    convs = [
        {
            "sector": "omni",
            "duration_sec": 142.5,
            "full_transcript": "Merhaba, internet faturam çok yüksek gelmiş. Geçen ayki zam yanlış yansımış, iade talep ediyorum.",
            "asr_confidence": 0.96,
            "quality_gate_passed": True,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "Merhaba, internet faturam çok yüksek gelmiş.",
                    "speaker": "customer",
                },
                {
                    "start": 3.0,
                    "end": 6.5,
                    "text": "Geçen ayki zam yanlış yansımış, iade talep ediyorum.",
                    "speaker": "customer",
                },
            ],
        },
        {
            "sector": "telco",
            "duration_sec": 89.0,
            "full_transcript": "Aboneliğimi iptal etmek istiyorum. Rakip operatör daha iyi bir kampanya sundu.",
            "asr_confidence": 0.94,
            "quality_gate_passed": True,
            "segments": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": "Aboneliğimi iptal etmek istiyorum.",
                    "speaker": "customer",
                },
                {
                    "start": 4.5,
                    "end": 8.0,
                    "text": "Rakip operatör daha iyi bir kampanya sundu.",
                    "speaker": "customer",
                },
            ],
        },
        {
            "sector": "finance",
            "duration_sec": 310.2,
            "full_transcript": "Kredi kartı ekstremdeki bir işleme itiraz etmek istiyorum. Ben böyle bir harcama yapmadım, şikayetçiyim.",
            "asr_confidence": 0.98,
            "quality_gate_passed": True,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Kredi kartı ekstremdeki bir işleme itiraz etmek istiyorum.",
                    "speaker": "customer",
                },
                {
                    "start": 5.5,
                    "end": 9.5,
                    "text": "Ben böyle bir harcama yapmadım, şikayetçiyim.",
                    "speaker": "customer",
                },
            ],
        },
    ]

    for c in convs:
        conv_id = new_uuid()
        conversation = Conversation(
            id=conv_id,
            sector=c["sector"],
            duration_sec=c["duration_sec"],
            full_transcript=c["full_transcript"],
            asr_confidence=c["asr_confidence"],
            quality_gate_passed=c["quality_gate_passed"],
            created_at=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=secrets.randbelow(6)),
        )
        db.add(conversation)

        for s in c["segments"]:
            seg = TranscriptSegmentRow(
                id=new_uuid(),
                conversation_id=conv_id,
                start=s["start"],
                end=s["end"],
                text=s["text"],
                speaker=s["speaker"],
            )
            db.add(seg)

    db.commit()
    print("Seeded demo conversations successfully.")


if __name__ == "__main__":
    seed_demo_conversations()
