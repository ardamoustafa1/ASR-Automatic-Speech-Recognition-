import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    # ─── Core API Config ──────────────────────────────────────────────────────
    # Bind-all is intentional in containerized deployments behind Docker/Kubernetes ingress.
    api_host: str = "0.0.0.0"  # nosec B104
    api_port: int = 8000

    # ─── Security ─────────────────────────────────────────────────────────────
    # REQUIRED: Must be set via environment variable or .env file.
    # Do NOT use this default in production. Generate with:
    #   python -c "import secrets; print(secrets.token_hex(32))"
    jwt_secret_key: str = ""

    # ─── Database ─────────────────────────────────────────────────────────────
    database_url: str = f"sqlite:///{DATA_DIR / 'asr_pro.db'}"

    # ─── Admin ────────────────────────────────────────────────────────────────
    admin_password: str = ""

    # ─── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # ─── Cache / Queue ────────────────────────────────────────────────────────
    redis_url: str = ""

    # ─── Webhooks ─────────────────────────────────────────────────────────────
    webhook_url: str = ""

    model_config = SettingsConfigDict(env_prefix="ASR_", env_file=".env", extra="ignore")


settings = Settings()

# ─── Runtime Validation ───────────────────────────────────────────────────────
_is_testing = os.getenv("ASR_TEST_NO_MODEL") == "1" or "pytest" in sys.modules

if not settings.jwt_secret_key:
    if os.getenv("ASR_ENV") == "prod":
        raise ValueError("ASR_JWT_SECRET_KEY must be set in production environment!")
    else:
        import secrets

        print(
            "WARNING: ASR_JWT_SECRET_KEY not set. Generating a temporary secret for development.",
            file=sys.stderr,
        )
        settings = settings.model_copy(update={"jwt_secret_key": secrets.token_hex(32)})

if not settings.admin_password:
    pass  # Admin password must be provided externally


# ─── Aliases for backwards compatibility ─────────────────────────────────────
DATABASE_URL = settings.database_url
API_HOST = settings.api_host
API_PORT = settings.api_port
CORS_ORIGINS = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
JWT_SECRET_KEY = settings.jwt_secret_key
REDIS_URL = settings.redis_url
WEBHOOK_URL = settings.webhook_url
