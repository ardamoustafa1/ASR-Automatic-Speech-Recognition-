# Configuration Guide

ASR-Pro uses `pydantic-settings` to securely and efficiently manage environment variables. You must create a `.env` file in the root directory before starting the application.

## Core Environment Variables

| Variable | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `ASR_JWT_SECRET_KEY` | `string` | **REQUIRED** | Secret key used to sign authentication tokens. Must be a strong, unpredictable string. |
| `DATABASE_URL` | `string` | `sqlite:///./asr_pro.db` | Connection string for the database (SQLAlchemy format). |
| `ASR_MODEL_SIZE` | `string` | `turbo` | The size of the whisper model to load. Options: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `turbo`. |
| `ENV_MODE` | `string` | `development` | Setting to `production` disables certain debugging features. |

## Advanced Settings

| Variable | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `PORT` | `int` | `8000` | FastAPI server port. |
| `WORKERS` | `int` | `4` | Number of uvicorn workers (disable if using Singleton ASR heavily). |
| `VAD_FILTER` | `bool` | `true` | Enables Voice Activity Detection to skip silent chunks during transcription. |
| `COMPLIANCE_STRICT_MODE` | `bool` | `false` | If true, compliance engine will use higher fuzzy-matching thresholds. |

## Generating a Secure JWT Secret
You can generate a strong secret key using Python:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
Place this value in your `.env` file:
`ASR_JWT_SECRET_KEY=your_generated_secret_here`
