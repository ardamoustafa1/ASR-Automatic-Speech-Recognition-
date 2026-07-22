from unittest.mock import MagicMock, patch

import pytest

from asr_pro import worker


def test_main_raises_when_redis_url_not_set():
    with patch.object(worker, "REDIS_URL", ""):
        with pytest.raises(RuntimeError, match="ASR_REDIS_URL must be set"):
            worker.main()


def test_main_starts_rq_worker_when_redis_configured():
    mock_conn = MagicMock()
    mock_queue = MagicMock()
    mock_worker_instance = MagicMock()

    with (
        patch.object(worker, "REDIS_URL", "redis://localhost:6379/0"),
        patch.object(worker.Redis, "from_url", return_value=mock_conn) as mock_from_url,
        patch.object(worker, "Queue", return_value=mock_queue) as mock_queue_cls,
        patch.object(worker, "Worker", return_value=mock_worker_instance) as mock_worker_cls,
    ):
        worker.main()

    mock_from_url.assert_called_once_with("redis://localhost:6379/0")
    mock_queue_cls.assert_called_once_with("asr-pro", connection=mock_conn)
    mock_worker_cls.assert_called_once_with([mock_queue], connection=mock_conn)
    mock_worker_instance.work.assert_called_once()
