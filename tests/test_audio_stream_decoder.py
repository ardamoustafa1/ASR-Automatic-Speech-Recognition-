import asyncio
import subprocess
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from asr_pro.services import audio_stream_decoder as decoder_module
from asr_pro.services.audio_stream_decoder import (
    _FFMPEG_BIN,
    AudioDecodeError,
    AudioStreamDecoder,
)

pytestmark = pytest.mark.skipif(not _FFMPEG_BIN, reason="ffmpeg binary not available")

SAMPLE_RATE = 16000
TONE_SEC = 1.5


def _make_webm_fixture(tmp_path) -> bytes:
    """Synthesize a short sine tone, encode it to WebM/Opus with the same ffmpeg
    binary the decoder uses, and return the raw container bytes."""
    wav_path = tmp_path / "tone.wav"
    t = np.linspace(0, TONE_SEC, int(SAMPLE_RATE * TONE_SEC), endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(tone.tobytes())

    webm_path = tmp_path / "tone.webm"
    subprocess.run(
        [
            _FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-c:a",
            "libopus",
            str(webm_path),
        ],
        check=True,
    )
    return webm_path.read_bytes()


@pytest.mark.asyncio
async def test_decoder_streams_pcm_from_chunked_webm(tmp_path):
    webm_bytes = _make_webm_fixture(tmp_path)
    assert len(webm_bytes) > 0

    decoder = AudioStreamDecoder()
    await decoder.start()
    try:
        all_pcm = []
        chunk_size = 4096
        for i in range(0, len(webm_bytes), chunk_size):
            await decoder.write(webm_bytes[i : i + chunk_size])
            piece = await decoder.read_available()
            if piece.size:
                all_pcm.append(piece)
        # Give ffmpeg a moment to flush trailing output after the last write.
        for _ in range(20):
            piece = await decoder.read_available()
            if piece.size:
                all_pcm.append(piece)
            else:
                break
    finally:
        await decoder.close()

    assert all_pcm, "decoder produced no PCM output at all"
    pcm = np.concatenate(all_pcm)
    decoded_sec = pcm.size / SAMPLE_RATE
    # Opus encoding adds a small amount of pre/post padding; allow generous tolerance.
    assert decoded_sec == pytest.approx(TONE_SEC, abs=0.3)
    rms = float(np.sqrt(np.mean(np.square(pcm))))
    assert rms > 0.05, "decoded PCM looks silent, decode likely failed"


@pytest.mark.asyncio
async def test_decoder_close_is_idempotent(tmp_path):
    decoder = AudioStreamDecoder()
    await decoder.start()
    await decoder.close()
    await decoder.close()  # must not raise


def test_init_raises_when_ffmpeg_binary_missing():
    with patch.object(decoder_module, "_FFMPEG_BIN", None):
        with pytest.raises(AudioDecodeError, match="ffmpeg binary not found"):
            AudioStreamDecoder()


@pytest.mark.asyncio
async def test_write_raises_when_not_started():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._process = None
    decoder._closed = False
    with pytest.raises(AudioDecodeError, match="not started"):
        await decoder.write(b"data")


@pytest.mark.asyncio
async def test_write_raises_on_broken_pipe():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._closed = False
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdin.write.side_effect = BrokenPipeError("pipe closed")
    decoder._process = mock_process
    with pytest.raises(AudioDecodeError, match="stdin closed unexpectedly"):
        await decoder.write(b"data")


@pytest.mark.asyncio
async def test_read_available_returns_empty_when_not_started():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._process = None
    decoder._closed = False
    result = await decoder.read_available()
    assert result.size == 0


@pytest.mark.asyncio
async def test_read_available_returns_empty_when_closed():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._process = MagicMock()
    decoder._closed = True
    result = await decoder.read_available()
    assert result.size == 0


@pytest.mark.asyncio
async def test_read_available_raises_on_stdout_read_failure():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._closed = False
    mock_process = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stdout.read = AsyncMock(side_effect=RuntimeError("pipe broke"))
    decoder._process = mock_process
    with pytest.raises(AudioDecodeError, match="ffmpeg stdout read failed"):
        await decoder.read_available()


@pytest.mark.asyncio
async def test_read_available_drops_trailing_odd_byte():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._closed = False
    mock_process = MagicMock()
    mock_process.stdout = MagicMock()
    # 3 bytes: only one full int16 sample (2 bytes) should be used.
    mock_process.stdout.read = AsyncMock(side_effect=[b"\x01\x00\x02", asyncio.TimeoutError()])
    decoder._process = mock_process
    result = await decoder.read_available()
    assert result.size == 1


@pytest.mark.asyncio
async def test_drain_stderr_returns_immediately_when_no_process():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._process = None
    await decoder._drain_stderr()  # must not raise


@pytest.mark.asyncio
async def test_close_kills_process_on_wait_timeout():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._closed = False
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), None])
    decoder._process = mock_process
    decoder._stderr_task = None

    await decoder.close()

    mock_process.kill.assert_called_once()
    assert decoder._closed is True


@pytest.mark.asyncio
async def test_close_swallows_unexpected_wait_exception():
    decoder = AudioStreamDecoder.__new__(AudioStreamDecoder)
    decoder._closed = False
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.wait = AsyncMock(side_effect=RuntimeError("weird failure"))
    decoder._process = mock_process
    decoder._stderr_task = None

    await decoder.close()  # must not raise

    mock_process.kill.assert_called_once()
