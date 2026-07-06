from unittest.mock import MagicMock, patch

from asr_pro.services import task_queue


def test_enqueue_runs_inline_when_redis_not_configured():
    task_queue._queue = None
    task_queue._queue_init_attempted = False
    with patch.object(task_queue, "REDIS_URL", ""):
        calls = []
        job_id = task_queue.enqueue(calls.append, "hello")
    assert job_id == "inline"
    assert calls == ["hello"]


def test_enqueue_uses_rq_queue_when_redis_available():
    task_queue._queue = None
    task_queue._queue_init_attempted = False
    mock_queue = MagicMock()
    mock_queue.enqueue.return_value = MagicMock(id="job-123")

    def fake_get_queue():
        task_queue._queue = mock_queue
        return mock_queue

    with patch.object(task_queue, "_get_queue", side_effect=fake_get_queue):
        job_id = task_queue.enqueue(str, "unused")

    assert job_id == "job-123"
    mock_queue.enqueue.assert_called_once()


def teardown_module(module):
    task_queue._queue = None
    task_queue._queue_init_attempted = False
