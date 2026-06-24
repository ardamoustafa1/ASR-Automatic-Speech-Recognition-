"""FastAPI application entry point — Enterprise ASR-Pro API."""
from contextlib import asynccontextmanager
import sys
import os
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from loguru import logger

from asr_pro.config import CORS_ORIGINS

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

# ─── Structured logging setup ─────────────────────────────────────────────────
logger.remove()
if os.getenv("ENV", "dev") == "prod":
    logger.add(sys.stdout, serialize=True, level="INFO", backtrace=False, diagnose=False)
else:
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | <magenta>[{extra[trace_id]}]</magenta> | {message}",
        level="DEBUG",
        colorize=True,
    )

from asr_pro.db.session import init_db, SessionLocal
from asr_pro.api.deps import get_db, limiter
from asr_pro.services.seed_data import seed_defaults
from asr_pro.api.routes.keywords import router as keywords_router, topics_router
from asr_pro.api.routes.conversations import router as conversations_router
from asr_pro.api.routes.analytics import router as analytics_router
from asr_pro.api.routes.alerts import router as alerts_router
from asr_pro.api.routes.websocket import router as websocket_router
from asr_pro.api.routes.auth import router as auth_router

from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.backends.redis import RedisBackend
import redis.asyncio as aioredis

from prometheus_fastapi_instrumentator import Instrumentator


@asynccontextmanager
async def lifespan(app: FastAPI):
    from asr_pro.config import DATA_DIR
    DATA_DIR.mkdir(exist_ok=True)
    init_db()
    db = SessionLocal()
    try:
        seed_defaults(db)
        logger.info("Database seeding complete.")
    finally:
        db.close()

    # ─── Cache Initialization ─────────────────────────────────────────────────
    redis_url = os.getenv("ASR_REDIS_URL") or os.getenv("REDIS_URL")
    if redis_url:
        r = aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
        FastAPICache.init(RedisBackend(r), prefix="asr-pro-cache")
        logger.info(f"Cache: Redis connected at {redis_url}")
    else:
        FastAPICache.init(InMemoryBackend(), prefix="asr-pro-cache")
        logger.info("Cache: Using in-memory cache (set ASR_REDIS_URL for Redis)")

    yield
    logger.info("ASR-Pro API shutdown.")


app = FastAPI(
    title="ASR-Pro — Enterprise Speech Intelligence API",
    description=(
        "Real-time speech transcription, sentiment analysis, churn prediction, "
        "empathy scoring, and compliance monitoring for enterprise contact centers."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ─── Prometheus Instrumentation ───────────────────────────────────────────────
Instrumentator(
    should_group_status_codes=True,
    excluded_handlers=["/api/v1/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# ─── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=()"
    if os.getenv("ENV") == "prod":
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = trace_id_var.set(trace_id)
    with logger.contextualize(trace_id=trace_id):
        logger.debug(f"→ {request.method} {request.url.path}")
        response = await call_next(request)
        log = logger.info if response.status_code < 400 else logger.warning
        log(f"← {request.method} {request.url.path} [{response.status_code}]")
        response.headers["X-Trace-ID"] = trace_id
    trace_id_var.reset(token)
    return response


# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api/v1")
app.include_router(keywords_router, prefix="/api/v1")
app.include_router(topics_router, prefix="/api/v1")
app.include_router(conversations_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(alerts_router, prefix="/api/v1")
app.include_router(websocket_router)


@app.get("/api/v1/health", tags=["system"], include_in_schema=True)
def health(db: Session = Depends(get_db)):
    """Health check endpoint — returns 200 if healthy, 503 if degraded."""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "service": "asr-pro-api", "version": "1.0.0", "db": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "service": "asr-pro-api", "db": "disconnected"},
        )
