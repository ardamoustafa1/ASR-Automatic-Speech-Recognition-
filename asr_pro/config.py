# Configuration settings and environment variable validation for ASR Pro.
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TEMP_AUDIO_DIR = ROOT_DIR / "temp_audio_uploads"

load_dotenv(ROOT_DIR / ".env")


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

    # ─── AI Models / Hugging Face ─────────────────────────────────────────────
    hf_token: str = ""

    # ─── Live Streaming ASR ───────────────────────────────────────────────────
    # Max concurrent /ws/live-asr connections before new ones are rejected (1013).
    max_ws_connections: int = 50
    # Max seconds of uncommitted audio buffered before a forced commit.
    streaming_max_pending_sec: float = 30.0
    # Trailing silence (seconds) after speech required to commit a segment as final.
    streaming_silence_commit_sec: float = 0.6
    # Minimum new audio (seconds) required before attempting another partial transcribe.
    streaming_min_partial_interval_sec: float = 0.9

    # ─── Churn Risk Engine (per-deployment tunable, see docs/CHURN_RISK_METHODOLOGY.md) ──
    # Comma-separated names of the DEPLOYING company (the client running this
    # system), case-insensitive. Excluded from churn_engine's competitor
    # detection - e.g. when deployed for Vodafone, an agent saying "Vodafone'dan
    # arıyorum" on an outbound sales call must not score as a competitor
    # mention. Set via ASR_CHURN_OWN_COMPANY_NAMES.
    churn_own_company_names: str = ""
    # WPM above which a chunk's risk is amplified (acoustic stress heuristic).
    churn_wpm_high_threshold: int = 190
    churn_wpm_high_multiplier: float = 1.4
    churn_wpm_mid_threshold: int = 160
    churn_wpm_mid_multiplier: float = 1.2
    # Blend weights for max(chunk risk) vs. temporal-weighted mean(chunk risk).
    # Must sum to 1.0 - validated below.
    churn_max_weight: float = 0.65
    churn_mean_weight: float = 0.35
    # Bounded bonus applied when competitor names are detected in-call.
    churn_competitor_bonus_per_mention: float = 0.1
    churn_competitor_bonus_cap: float = 0.2
    # Risk score (0-1) at or above which a call is flagged "high risk".
    churn_alarm_threshold: float = 0.75

    # ─── Empathy Engine: interruption detection (see docs/CHURN_RISK_METHODOLOGY.md) ──
    # Stereo calls have their left/right channels transcribed INDEPENDENTLY by
    # Whisper, so each channel's segment boundaries come from an unsynchronized
    # VAD pass. Measured against real call-center audio, this produces two
    # distinct failure modes, not just sub-second jitter: (1) short (<1s)
    # boundary noise, and (2) segments with grossly wrong end timestamps that
    # produce multi-second "overlaps" (observed up to 11s) - those are not
    # real interruptions, a person cannot audibly cut someone off for 11
    # seconds. Both a floor and a ceiling are needed.
    # Minimum overlap (seconds) before it counts as a real interruption at all.
    empathy_interruption_min_overlap_sec: float = 0.7
    # Maximum plausible overlap for a genuine interruption - beyond this the
    # "overlap" almost certainly reflects a corrupted/over-extended segment
    # timestamp rather than sustained simultaneous speech.
    empathy_interruption_max_overlap_sec: float = 2.5
    # The segment being "cut off" must itself be at least this long - filters
    # out one-word backchannel acks ("evet"/"tamam") that naturally overlap
    # due to boundary imprecision, which are not real interruptions.
    empathy_interruption_min_prior_segment_sec: float = 0.8

    # ─── ASR Quality Parameters ────────────────────────────────────────────────
    # Initial prompt fed to Whisper before each segment - primes the model with
    # domain vocabulary and formatting conventions for Turkish call-center audio.
    # An explicit Turkish prompt also steers the model away from language
    # confusion on accented speech or code-switching segments.
    asr_initial_prompt: str = (
        "Bu bir Türkçe müşteri hizmetleri görüşmesidir. "
        "Konuşma açık, noktalı ve standart Türkçe yazım kurallarına uygun şekilde yazılmaktadır."
    )
    # Probability above which a segment is classified as silence/no-speech.
    # Lower value = stricter silence filtering (fewer hallucinated segments).
    # 0.45 is more conservative than the previous 0.6 default.
    asr_no_speech_threshold: float = 0.45
    # Log-probability below which a segment is discarded as low-confidence.
    asr_log_prob_threshold: float = -0.8
    # VAD sensitivity (0-1). Lower = only very clear speech passes through.
    asr_vad_threshold: float = 0.40
    # Minimum speech duration (ms) the VAD will keep.
    asr_vad_min_speech_ms: int = 200
    # Minimum silence (ms) required to split speech segments.
    asr_vad_min_silence_ms: int = 400
    # Padding (ms) added around detected speech boundaries.
    asr_vad_speech_pad_ms: int = 150
    # Beam size for Whisper beam search (higher = more accurate but slower).
    asr_beam_size: int = 5
    # Temperature schedule for Whisper. Whisper retries with higher temperatures
    # when a segment fails the quality gate - this gives 3 chances.
    asr_temperature: str = "0.0,0.2,0.4"
    # Whether to compute word-level timestamps (required for precise diarization alignment).
    asr_word_timestamps: bool = True

    # ─── Diarization ──────────────────────────────────────────────────────────
    # Expected number of speakers per call, passed to pyannote as a hint.
    # Set to 0 to let pyannote auto-detect the speaker count.
    diarization_expected_speakers: int = 0
    # Hard upper bound on speaker count passed to pyannote's min/max_speakers.
    diarization_max_speakers: int = 4
    # Hard lower bound on speaker count.
    diarization_min_speakers: int = 1
    # Energy ratio margin required for one stereo channel to be considered the
    # dominant speaker in a window (e.g. 1.3 = one channel must be 30% louder).
    # Only used as a last-resort fallback when pyannote itself is unavailable
    # (no HF token, dependency missing, runtime error).
    diarization_energy_margin: float = 1.3
    # Whisper avg_logprob below this is flagged as low-confidence in the
    # transcript UI.
    transcript_low_confidence_logprob: float = -1.1

    model_config = SettingsConfigDict(env_prefix="ASR_", env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _validate_churn_weights(self) -> "Settings":
        total = self.churn_max_weight + self.churn_mean_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ASR_CHURN_MAX_WEIGHT + ASR_CHURN_MEAN_WEIGHT must sum to 1.0 (got {total})"
            )
        return self


settings = Settings()

# ─── Runtime Validation ───────────────────────────────────────────────────────
_is_testing = os.getenv("ASR_TEST_NO_MODEL") == "1" or "pytest" in sys.modules

# Sync Hugging Face Tokens for pyannote.audio and transformers
_hf_token = settings.hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if _hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    os.environ["HF_TOKEN"] = _hf_token

if not settings.jwt_secret_key:
    if os.getenv("ASR_ENV") == "prod":
        raise ValueError("ASR_JWT_SECRET_KEY must be set in production environment!")
    else:
        import secrets

        logger.warning(
            "ASR_JWT_SECRET_KEY not set. Generating ephemeral 256-bit secret for local development."
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
MAX_WS_CONNECTIONS = settings.max_ws_connections
STREAMING_MAX_PENDING_SEC = settings.streaming_max_pending_sec
STREAMING_SILENCE_COMMIT_SEC = settings.streaming_silence_commit_sec
STREAMING_MIN_PARTIAL_INTERVAL_SEC = settings.streaming_min_partial_interval_sec
