# ASR-Pro Deployment Guide

## Prerequisites

- Docker Engine 24+ and Docker Compose v2+
- A Linux/macOS host with at least 4GB RAM (8GB recommended for Whisper turbo)
- Domain name + SSL certificate (for production HTTPS)

## Quick Start (Docker Compose)

### 1. Clone & Configure

```bash
git clone https://github.com/ardamoustafa/ASR-Pro.git
cd ASR-Pro

# Copy example environment file
cp .env.example .env
```

### 2. Set Required Secrets in `.env`

Open `.env` and fill in **all required** variables:

```bash
# Generate a cryptographically secure JWT key:
python -c "import secrets; print(secrets.token_hex(32))"

# Paste the result as ASR_JWT_SECRET_KEY
ASR_JWT_SECRET_KEY=<paste-generated-key-here>

# Set a strong admin password (12+ chars)
ASR_ADMIN_PASSWORD=<your-strong-password>

# Set strong database credentials
POSTGRES_USER=asr_user
POSTGRES_PASSWORD=<your-strong-db-password>
```

### 3. Launch

```bash
# Start all services (PostgreSQL + API + Streamlit + React Frontend)
docker-compose up -d

# Monitor startup
docker-compose logs -f api
```

### 4. Verify

| Service | URL | Notes |
|---|---|---|
| React Dashboard | http://localhost:5173 | Main UI |
| FastAPI Docs | http://localhost:8000/api/docs | Swagger UI |
| Streamlit ASR | http://localhost:8501 | Legacy UI |
| Health Check | http://localhost:8000/api/v1/health | Returns 200 if OK |
| Prometheus | http://localhost:8000/metrics | Metrics scrape |

---

## GPU Deployment (production ASR throughput)

CPU decoding of Whisper large-v3 cannot keep up with contact-center call
volume. On an NVIDIA GPU, faster-whisper decodes at roughly RTF 0.05–0.10
(hundreds of calls per GPU per hour):

```bash
# Host prerequisites: NVIDIA driver + container toolkit
#   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

The override builds `Dockerfile.backend.gpu` (CUDA 12.8 + cuDNN, required for
Blackwell — B100/B200/GB200, RTX 50-series — see the `CUDA_TAG` build arg
comment in that file) for the `api` and `worker` services and reserves one
GPU each. `ASRService` auto-detects CUDA, logs the GPU's compute capability,
and probes CTranslate2 compute types at model-load time
(`float16` → `int8_float16` → `int8`), falling back automatically if the
installed CTranslate2 build lacks a kernel for a given compute
type/architecture combination — check the "ASR model loaded successfully
with compute_type=..." log line to see what's actually running.

**Blackwell note:** no CTranslate2 compute type here is confirmed to use
architecture-specific fast-path kernels on Blackwell as of this writing —
the fallback chain above exists specifically so an unsupported combination
degrades to a working configuration instead of crashing, not because a
Blackwell-optimized kernel is guaranteed to be selected. If you're deploying
on Blackwell, check your installed `ctranslate2` version's release notes for
GPU architecture support and confirm which compute type actually loaded via
the log line above. Model weights are cached in the `hf_cache` volume so
restarts don't re-download.

Throughput dial: `ASR_ASR_MODEL_SIZE=large-v3` (default) or `large-v3-turbo`
(measured faster AND more accurate on our clean benchmark - 0.31x RTF /
1.64% WER vs 0.51x / 4.37% - but turbo produced a real hallucination on one
noisy real call that large-v3 didn't; see `.benchmarks/results.md` before
switching the production default). A fine-tuned in-domain model (see
[WHISPER_FINETUNING.md](WHISPER_FINETUNING.md)) can be deployed via the same
variable with a local CTranslate2 path.

---

## Production Deployment (with Nginx + SSL)

### Option A: Docker Compose with Nginx Proxy

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
    }
}
```

### Option B: Kubernetes (Helm)

```bash
# Install with Helm
helm install asr-pro ./helm-chart \
  --set secrets.jwtKey="<your-key>" \
  --set secrets.adminPassword="<your-password>" \
  --set ingress.host="your-domain.com"
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ASR_JWT_SECRET_KEY` | ✅ Yes | — | JWT signing secret (min 32 chars) |
| `ASR_ADMIN_PASSWORD` | ✅ Yes | — | Initial admin user password |
| `POSTGRES_USER` | ✅ (prod) | `asr_user` | PostgreSQL username |
| `POSTGRES_PASSWORD` | ✅ (prod) | — | PostgreSQL password |
| `POSTGRES_DB` | No | `asr_pro` | PostgreSQL database name |
| `ASR_DATABASE_URL` | No | SQLite | Full database URL |
| `ASR_REDIS_URL` | No | — | Redis URL for caching |
| `ASR_WEBHOOK_URL` | No | — | Webhook for critical alerts |
| `ASR_CORS_ORIGINS` | No | `localhost:5173` | Comma-separated allowed origins |

---

## Monitoring

Prometheus metrics are exposed at `/metrics`. Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'asr-pro'
    static_configs:
      - targets: ['localhost:8000']
```

---

## Scaling

For high-concurrency production:

1. **Enable Redis**: Set `ASR_REDIS_URL` to share cache across multiple API replicas.
2. **Celery workers**: Coming in v1.1 — see [ROADMAP.md](../ROADMAP.md).
3. **Load balancer**: Point an Nginx upstream to multiple `api` containers.

-->
