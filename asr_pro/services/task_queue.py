# Distributed job queue for heavy ASR/NLP work, so it can scale independently
# of API replicas instead of running in-process via FastAPI BackgroundTasks.
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from loguru import logger

from asr_pro.config import REDIS_URL

_queue = None
_queue_init_attempted = False
_executor: ThreadPoolExecutor | None = None


def _get_queue():
    """Lazily connect to Redis/RQ. Returns None if unavailable (dev/test fallback)."""
    global _queue, _queue_init_attempted
    if _queue_init_attempted:
        return _queue
    _queue_init_attempted = True
    if not REDIS_URL:
        logger.info("ASR_REDIS_URL not set — background jobs run on an in-process worker thread.")
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


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        # 2 workers: one long transcription plus one lighter analyze job can
        # overlap; the ASR model itself is serialized by ASRService's
        # inference lock anyway, so more workers would only queue on that lock
        # while holding memory.
        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr-jobs")
    return _executor


def _run_logged(func: Callable, args: tuple, kwargs: dict) -> None:
    try:
        func(*args, **kwargs)
    except Exception:
        # Background threads swallow exceptions silently by default - always
        # leave a trace. Job-level failure *state* is the job's own
        # responsibility (see _process_audio_upload_background's failed-status
        # handling); this is the last-resort net under that.
        logger.exception(f"Background job {getattr(func, '__name__', func)!r} crashed")


def enqueue(func: Callable, *args: Any, **kwargs: Any) -> str:
    """Run `func(*args, **kwargs)` on an RQ worker if Redis is configured,
    otherwise on an in-process worker thread.

    The previous no-Redis fallback executed the job INLINE, which meant a
    caller like POST /conversations/upload held its HTTP response open for
    the entire multi-minute transcription - the "202 Accepted, processing in
    background" contract was a lie in every non-Redis deployment. Only test
    runs keep the inline behavior (tests assert on the job's results
    immediately after the request returns).

    Returns the RQ job id, "thread" for the executor path, or "inline" under
    test mode.
    """
    queue = _get_queue()
    if queue is not None:
        job = queue.enqueue(func, *args, **kwargs)
        return job.id

    from asr_pro.config import _is_testing

    if _is_testing:
        func(*args, **kwargs)
        return "inline"

    _get_executor().submit(_run_logged, func, args, kwargs)
    return "thread"
