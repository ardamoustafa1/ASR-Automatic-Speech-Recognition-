from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from asr_pro.services import streaming_session as streaming_session_module
from asr_pro.services.asr_service import TranscriptionSegment
from asr_pro.services.streaming_session import StreamingASRSession

SAMPLE_RATE = 16000


class FakeDecoder:
    """Stand-in for AudioStreamDecoder: hands back pre-queued PCM, ignores writes."""

    def __init__(self):
        self.pending = []
        self.closed = False
        self.started = False

    async def start(self):
        self.started = True

    async def write(self, chunk):
        pass

    async def read_available(self):
        if self.pending:
            return self.pending.pop(0)
        return np.empty(0, dtype=np.float32)

    async def close(self):
        self.closed = True


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32)


def _tone(sec: float) -> np.ndarray:
    # Non-zero content so the energy-fallback path (no VAD model loaded) treats it as speech.
    return np.ones(int(sec * SAMPLE_RATE), dtype=np.float32) * 0.5


def _make_session() -> StreamingASRSession:
    return StreamingASRSession(language="tr", decoder=FakeDecoder())


@pytest.mark.asyncio
async def test_silence_is_dropped_after_grace_period():
    session = _make_session()
    mock_vad = MagicMock()
    mock_vad.loaded = True
    mock_vad.filter_speech_timestamps.return_value = []

    with patch.object(streaming_session_module.VADService, "get_instance", return_value=mock_vad):
        session.decoder.pending.append(_silence(1.0))
        msg1 = await session.push_audio(b"chunk")
        assert msg1 is None
        assert session.pcm_buffer.size > 0  # still within grace period, not dropped yet

        session.decoder.pending.append(_silence(1.0))
        msg2 = await session.push_audio(b"chunk")
        assert msg2 is None
        assert session.pcm_buffer.size == 0  # grace period exceeded -> dropped


@pytest.mark.asyncio
async def test_ongoing_speech_yields_partial_not_final():
    session = _make_session()
    mock_vad = MagicMock()
    mock_vad.loaded = True

    buffer_sec = 1.0
    # Speech runs almost to the end of the buffer -> trailing silence well under the commit threshold.
    mock_vad.filter_speech_timestamps.return_value = [
        {"start": 0, "end": int((buffer_sec - 0.1) * SAMPLE_RATE)}
    ]
    mock_asr = MagicMock()
    mock_asr.transcribe_array.return_value = (
        [TranscriptionSegment(start=0.0, end=buffer_sec - 0.1, text="merhaba")],
        buffer_sec,
    )

    with patch.object(streaming_session_module.VADService, "get_instance", return_value=mock_vad):
        with patch.object(
            streaming_session_module.ASRService, "get_instance", return_value=mock_asr
        ):
            session.decoder.pending.append(_tone(buffer_sec))
            msg = await session.push_audio(b"chunk")

    assert msg["type"] == "partial"
    assert msg["text"] == "merhaba"
    assert session.committed_text == ""
    assert session.pcm_buffer.size > 0  # nothing committed, buffer retained


@pytest.mark.asyncio
async def test_trailing_silence_triggers_final_commit():
    session = _make_session()
    mock_vad = MagicMock()
    mock_vad.loaded = True

    buffer_sec = 2.0
    speech_end_sec = 1.0  # 1.0s of trailing silence -> exceeds the 0.6s commit threshold
    mock_vad.filter_speech_timestamps.return_value = [
        {"start": 0, "end": int(speech_end_sec * SAMPLE_RATE)}
    ]
    mock_asr = MagicMock()
    mock_asr.transcribe_array.return_value = (
        [TranscriptionSegment(start=0.0, end=speech_end_sec, text="merhaba nasılsın")],
        buffer_sec,
    )

    with patch.object(streaming_session_module.VADService, "get_instance", return_value=mock_vad):
        with patch.object(
            streaming_session_module.ASRService, "get_instance", return_value=mock_asr
        ):
            session.decoder.pending.append(_tone(buffer_sec))
            msg = await session.push_audio(b"chunk")

    assert msg["type"] == "final"
    assert msg["text"] == "merhaba nasılsın"
    assert msg["transcript_so_far"] == "merhaba nasılsın"
    assert session.committed_text == "merhaba nasılsın"
    # Buffer should be trimmed down to just the post-commit remainder.
    remaining_sec = session.pcm_buffer.size / SAMPLE_RATE
    assert remaining_sec == pytest.approx(buffer_sec - speech_end_sec, abs=0.01)
    assert session.committed_offset_sec == pytest.approx(speech_end_sec, abs=0.01)


@pytest.mark.asyncio
async def test_max_pending_forces_commit_even_without_silence():
    session = _make_session()
    monkeypatch_target = streaming_session_module
    original_max = monkeypatch_target.STREAMING_MAX_PENDING_SEC
    monkeypatch_target.STREAMING_MAX_PENDING_SEC = 1.0
    try:
        mock_vad = MagicMock()
        mock_vad.loaded = True
        buffer_sec = 1.2
        # Speech runs all the way to the buffer end (no natural pause).
        mock_vad.filter_speech_timestamps.return_value = [
            {"start": 0, "end": int(buffer_sec * SAMPLE_RATE)}
        ]
        mock_asr = MagicMock()
        mock_asr.transcribe_array.return_value = (
            [TranscriptionSegment(start=0.0, end=buffer_sec, text="uzun cumle devam ediyor")],
            buffer_sec,
        )

        with patch.object(
            streaming_session_module.VADService, "get_instance", return_value=mock_vad
        ):
            with patch.object(
                streaming_session_module.ASRService, "get_instance", return_value=mock_asr
            ):
                session.decoder.pending.append(_tone(buffer_sec))
                msg = await session.push_audio(b"chunk")

        assert msg["type"] == "final"  # forced by MAX_PENDING_SEC despite no detected pause
    finally:
        monkeypatch_target.STREAMING_MAX_PENDING_SEC = original_max


@pytest.mark.asyncio
async def test_flush_final_commits_remaining_buffer():
    session = _make_session()
    mock_asr = MagicMock()
    mock_asr.transcribe_array.return_value = (
        [TranscriptionSegment(start=0.0, end=0.5, text="son soz")],
        0.5,
    )
    session.pcm_buffer = _tone(0.5)

    with patch.object(streaming_session_module.ASRService, "get_instance", return_value=mock_asr):
        msg = await session.flush_final()

    assert msg["type"] == "final"
    assert msg["text"] == "son soz"
    assert session.pcm_buffer.size == 0


@pytest.mark.asyncio
async def test_flush_final_on_empty_buffer_returns_none():
    session = _make_session()
    msg = await session.flush_final()
    assert msg is None
