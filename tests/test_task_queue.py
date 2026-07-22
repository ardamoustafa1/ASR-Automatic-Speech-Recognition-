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


def test_get_queue_connects_successfully_when_redis_reachable():
    task_queue._queue = None
    task_queue._queue_init_attempted = False
    mock_conn = MagicMock()
    mock_rq_queue = MagicMock()
    with (
        patch.object(task_queue, "REDIS_URL", "redis://localhost:6379/0"),
        patch("redis.from_url", return_value=mock_conn) as mock_from_url,
        patch("rq.Queue", return_value=mock_rq_queue) as mock_queue_cls,
    ):
        result = task_queue._get_queue()

    mock_from_url.assert_called_once_with("redis://localhost:6379/0")
    mock_conn.ping.assert_called_once()
    mock_queue_cls.assert_called_once_with("asr-pro", connection=mock_conn)
    assert result is mock_rq_queue

    # Second call is a no-op — the connection attempt is only made once.
    with patch("redis.from_url") as mock_from_url_again:
        assert task_queue._get_queue() is mock_rq_queue
        mock_from_url_again.assert_not_called()


def test_get_queue_falls_back_to_none_when_redis_unreachable():
    task_queue._queue = None
    task_queue._queue_init_attempted = False
    with (
        patch.object(task_queue, "REDIS_URL", "redis://localhost:6379/0"),
        patch("redis.from_url", side_effect=ConnectionError("refused")),
    ):
        result = task_queue._get_queue()
    assert result is None


def test_get_executor_returns_singleton_thread_pool_executor():
    task_queue._executor = None
    first = task_queue._get_executor()
    second = task_queue._get_executor()
    assert first is second
    assert first._max_workers == 2
    task_queue._executor = None


def test_run_logged_executes_function_with_args_and_kwargs():
    calls = []
    task_queue._run_logged(lambda *a, **kw: calls.append((a, kw)), (1, 2), {"x": 3})
    assert calls == [((1, 2), {"x": 3})]


def test_run_logged_swallows_exceptions():
    def boom():
        raise ValueError("kaboom")

    # Must not raise — background threads have nowhere to propagate to.
    task_queue._run_logged(boom, (), {})


def test_enqueue_submits_to_thread_executor_outside_test_mode():
    task_queue._queue = None
    task_queue._queue_init_attempted = False
    mock_executor = MagicMock()
    with (
        patch.object(task_queue, "_get_queue", return_value=None),
        patch("asr_pro.config._is_testing", False),
        patch.object(task_queue, "_get_executor", return_value=mock_executor),
    ):
        job_id = task_queue.enqueue(str, "unused")

    assert job_id == "thread"
    mock_executor.submit.assert_called_once()


def teardown_module(module):
    task_queue._queue = None
    task_queue._queue_init_attempted = False
    task_queue._executor = None
