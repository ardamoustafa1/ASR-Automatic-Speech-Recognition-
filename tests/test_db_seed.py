from sqlalchemy.orm import Session

from asr_pro.db.models import KeywordRule, Topic, User
from asr_pro.db.seed import seed_defaults


def test_seed_defaults(db_session: Session):
    # Ensure it runs without error
    seed_defaults(db_session)

    # Check users
    users = db_session.query(User).all()
    assert len(users) >= 2

    # Check topics
    topics = db_session.query(Topic).all()
    assert len(topics) >= 5

    # Check keyword rules
    rules = db_session.query(KeywordRule).all()
    assert len(rules) >= 4
