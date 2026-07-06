# RQ worker entrypoint — consumes the "asr-pro" queue populated by
# asr_pro.services.task_queue.enqueue(). Run with: python -m asr_pro.worker
from loguru import logger
from redis import Redis
from rq import Queue, Worker

from asr_pro.config import REDIS_URL


def main() -> None:
    if not REDIS_URL:
        raise RuntimeError("ASR_REDIS_URL must be set to run a queue worker.")
    conn = Redis.from_url(REDIS_URL)
    queue = Queue("asr-pro", connection=conn)
    logger.info(f"Worker listening on queue 'asr-pro' at {REDIS_URL}")
    Worker([queue], connection=conn).work()


if __name__ == "__main__":
    main()
