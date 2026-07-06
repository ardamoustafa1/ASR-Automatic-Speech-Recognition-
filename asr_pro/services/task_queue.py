# Distributed job queue for heavy ASR/NLP work, so it can scale independently
# of API replicas instead of running in-process via FastAPI BackgroundTasks.
from __future__ import annotations

from typing import Any, Callable

from loguru import logger

from asr_pro.config import REDIS_URL

_queue = None
_queue_init_attempted = False


def _get_queue():
    """Lazily connect to Redis/RQ. Returns None if unavailable (dev/test fallback)."""
    global _queue, _queue_init_attempted
    if _queue_init_attempted:
        return _queue
    _queue_init_attempted = True
    if not REDIS_URL:
        logger.info("ASR_REDIS_URL not set — background jobs run in-process (dev mode).")
        return None
    try:
        import redis
        from rq import Queue

        conn = redis.from_url(REDIS_URL)
        conn.ping()
        _queue = Queue("asr-pro", connection=conn)
        logger.info(f"Job queue connected to Redis at {REDIS_URL}.")
    except Exception as exc:
        logger.warning(f"Could not connect job queue to Redis ({exc}); running jobs in-process.")
        _queue = None
    return _queue


def enqueue(func: Callable, *args: Any, **kwargs: Any) -> str:
    """Run `func(*args, **kwargs)` on an RQ worker if Redis is configured,
    otherwise execute it immediately in-process (matches previous
    BackgroundTasks behavior — safe default for local dev and tests).

    Returns the RQ job id, or "inline" if executed synchronously.
    """
    queue = _get_queue()
    if queue is not None:
        job = queue.enqueue(func, *args, **kwargs)
        return job.id
    func(*args, **kwargs)
    return "inline"
