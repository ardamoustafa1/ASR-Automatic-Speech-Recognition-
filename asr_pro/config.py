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
    # Whisper model checkpoint. Default "large-v3". "large-v3-turbo" measured
    # faster AND more accurate on our clean benchmark (see
    # .benchmarks/results.md) but produced a real hallucination on one noisy
    # real call that large-v3 didn't - verify on your own real recordings
    # (scripts/evaluate_wer.py) before changing this default in production.
    asr_model_size: str = "large-v3"
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
    # Gap (seconds) under which adjacent VAD speech regions are merged into
    # one decode region (MLX path). Wider = fewer regions = less fixed
    # per-region decode overhead; the bridged silence is bounded by this
    # value and covered by the hallucination guards.
    asr_vad_region_merge_gap_sec: float = 3.0
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
    # Calls longer than this are treated as likely multi-party (conference /
    # supervisor transfer) rather than a plain two-party call, and get the
    # wider `diarization_max_speakers_conference` bound instead of
    # `diarization_max_speakers`. Only applies when diarization_expected_speakers
    # is unset (0), since an explicit hint always wins.
    diarization_conference_duration_sec: float = 900.0
    # Upper bound on speaker count for calls longer than
    # diarization_conference_duration_sec.
    diarization_max_speakers_conference: int = 8
    # Path to a segmentation model checkpoint fine-tuned on this deployment's
    # own annotated call recordings (see scripts/finetune_diarization.py). If
    # set and the file exists, DiarizationService swaps it in for the stock
    # pyannote/segmentation-3.0 segmentation model inside the pretrained
    # pyannote/speaker-diarization-3.1 pipeline (same embedding/clustering as
    # the stock pipeline, only segmentation is replaced). Empty = use the
    # stock pretrained pipeline unmodified.
    diarization_finetuned_segmentation_path: str = ""
    # Whisper avg_logprob below this is flagged as low-confidence in the
    # transcript UI.
    transcript_low_confidence_logprob: float = -1.1

    # ─── Second-Pass Rescue Decoding ───────────────────────────────────────────
    # After the initial decode, segments whose avg_logprob falls below the
    # threshold are re-decoded in isolation (fresh acoustic window, preceding
    # confident text injected as context prompt). The new hypothesis is only
    # accepted when its avg_logprob beats the original by the margin - a
    # principled "pick the higher-likelihood transcription" rule that can fix
    # boundary-artifact misrecognitions without ever degrading a segment.
    asr_second_pass_enabled: bool = True
    asr_second_pass_logprob_threshold: float = -0.5
    asr_second_pass_margin: float = 0.10
    # Segments longer than this aren't rescued (full-window redecode of a long
    # span is expensive and rarely a boundary artifact).
    asr_second_pass_max_segment_sec: float = 15.0
    # Hard cap on rescue decode attempts per channel, bounding worst-case
    # added latency on badly degraded audio where many segments would
    # otherwise qualify.
    asr_second_pass_max_attempts: int = 8

    # ─── Biometric Voiceprints ─────────────────────────────────────────────────
    # Cosine similarity threshold above which a speaker is considered matched
    # to an enrolled agent voiceprint.
    biometric_match_threshold: float = 0.85

    # ─── Upload Limits ─────────────────────────────────────────────────────────
    # Maximum accepted audio upload size (MB). A 2-hour stereo 8kHz WAV is
    # ~115MB; the default leaves headroom without letting a single request
    # exhaust disk (uploads are written to TEMP_AUDIO_DIR before processing).
    max_upload_mb: int = 300
    # Model preload at API startup: pays the ~10s Whisper weight load once in
    # a background thread instead of on the first customer upload.
    asr_preload_model: bool = True

    # ─── PII Redaction (KVKK / PCI-DSS) ───────────────────────────────────────
    # When enabled, transcripts are scanned before persistence and validated
    # personal identifiers are masked: TCKN (11-digit national ID, checksum
    # verified), credit card PANs (Luhn verified), Turkish IBANs (mod-97
    # verified) and mobile phone numbers. Banks and telecom operators must not
    # store card numbers (PCI-DSS 3.4) or unnecessary national IDs (KVKK veri
    # minimizasyonu) in call transcripts. Masking keeps the last 2-4 digits so
    # QA reviewers can still cross-reference.
    pii_redaction_enabled: bool = True

    # ─── Toxicity / Abusive Language Detection ─────────────────────────────────
    # Flags explicit profanity or aggressive insults from either party, for QA
    # escalation. Deliberately curated to a narrow, unambiguous term list -
    # see asr_pro/core/toxicity_engine.py's module docstring for why a wider
    # generic Turkish "bad word" list is unsafe to ship (false-positives on
    # ordinary business vocabulary like "hasta"/patient, "mal"/goods).
    toxicity_detection_enabled: bool = True

    # ─── CRM Auto-Note (Call Summary) ──────────────────────────────────────────
    # Generates a structured intent/issue/action/resolution summary plus a
    # short executive summary per call, via the same zero-shot classifier
    # already loaded for sentiment/churn/compliance (no extra model, no
    # external service required). Adds one more classifier pass per call.
    crm_summary_enabled: bool = True

    # ─── Crosstalk Speech Separation ───────────────────────────────────────────
    # Whether to run SepFormer source separation + a second Whisper pass on
    # detected crosstalk (overlapping speech) windows, to recover both
    # parties' words instead of one garbled/dropped transcript. Adds real
    # latency per crosstalk event - disable for high-volume deployments where
    # this isn't worth the added processing time.
    crosstalk_separation_enabled: bool = True
    # Crosstalk events shorter than this are not worth the separation +
    # re-transcription overhead (sub-100ms boundary noise rarely carries a
    # full word anyway).
    crosstalk_separation_min_duration_sec: float = 0.3

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
elif os.getenv("ASR_ENV") == "prod":
    # Without this token, DiarizationService silently degrades to text-heuristic
    # or per-channel energy-margin diarization - neither has any real acoustic
    # speaker separation, and both would corrupt churn/empathy/compliance
    # scoring without necessarily failing loudly. An enterprise deployment
    # (e.g. a large telecom contact center) must fail at startup instead of
    # discovering this per-call in logs.
    raise ValueError(
        "ASR_HF_TOKEN (or HUGGING_FACE_HUB_TOKEN/HF_TOKEN) must be set in production "
        "environment - required for pyannote.audio acoustic speaker diarization. "
        "Without it, diarization silently degrades to unreliable heuristics."
    )

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
