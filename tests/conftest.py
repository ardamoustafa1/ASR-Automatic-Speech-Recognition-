import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Set environment variables for testing
os.environ["ASR_JWT_SECRET_KEY"] = "test_secret_key_at_least_32_bytes_long"
os.environ["ASR_ADMIN_PASSWORD"] = "password123"
os.environ["ASR_AGENT_PASSWORD"] = "password123"
TEST_DB_URL = "sqlite:///file:testdb?mode=memory&cache=shared&uri=true"
os.environ["ASR_DATABASE_URL"] = TEST_DB_URL

from asr_pro.api.deps import limiter
from asr_pro.api.main import app
from asr_pro.db.models import Base
from asr_pro.db.session import get_db

# Disable rate limiting for tests
limiter.enabled = False

engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

from asr_pro.services.seed_data import seed_defaults


@pytest.fixture(scope="session")
def setup_db():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    seed_defaults(session)
    session.commit()
    session.close()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_db):
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
