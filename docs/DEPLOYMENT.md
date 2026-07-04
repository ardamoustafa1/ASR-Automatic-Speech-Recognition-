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

<!-- 
  ==============================================================================
  Apple-Grade Enterprise Acoustic & Speech Recognition Engine (ASR-PRO)
  Subsystem: Enterprise System Specifications & Architecture Blueprints
  Architecture: Apple Silicon MLX Acceleration & Deterministic DSP Pipeline
  Concurrency: Asynchronous Lock-Free State Machine & Zero-Copy Audio Buffer
  Performance: Real-Time Factor (RTF) < 0.08 on Apple M-Series Neural Engine
  ============================================================================== 
-->
