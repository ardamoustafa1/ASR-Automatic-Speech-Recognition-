# Persistent ffmpeg subprocess that decodes an incrementally-arriving WebM/Opus
# stream into 16kHz mono PCM without ever re-decoding already-consumed bytes.
import asyncio
import shutil
from typing import Optional

import numpy as np
from loguru import logger

try:
    import imageio_ffmpeg

    _FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    _FFMPEG_BIN = shutil.which("ffmpeg")

SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # s16le


class AudioDecodeError(RuntimeError):
    """Raised when the underlying ffmpeg decode process fails or is unavailable."""


class AudioStreamDecoder:
    """Feeds WebM/Opus chunks to a long-lived ffmpeg process and reads back PCM.

    One instance is created per live WebSocket connection. Because the ffmpeg
    process is never restarted mid-session, it only ever decodes each byte
    once, regardless of how long the call runs.
    """

    def __init__(self) -> None:
        if not _FFMPEG_BIN:
            raise AudioDecodeError(
                "ffmpeg binary not found (imageio-ffmpeg missing and no system ffmpeg)"
            )
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._closed = False

    async def start(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            _FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "webm",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        try:
            async for line in self._process.stderr:
                logger.debug(f"ffmpeg(stream-decoder): {line.decode(errors='ignore').strip()}")
        except Exception:
            pass

    async def write(self, chunk: bytes) -> None:
        if self._process is None or self._process.stdin is None or self._closed:
            raise AudioDecodeError("decoder not started or already closed")
        try:
            self._process.stdin.write(chunk)
            await self._process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise AudioDecodeError(f"ffmpeg stdin closed unexpectedly: {exc}") from exc

    async def read_available(self) -> np.ndarray:
        """Non-blocking-ish read of whatever PCM ffmpeg has produced so far."""
        if self._process is None or self._process.stdout is None or self._closed:
            return np.empty(0, dtype=np.float32)
        chunks = []
        try:
            while True:
                try:
                    piece = await asyncio.wait_for(self._process.stdout.read(65536), timeout=0.05)
                except asyncio.TimeoutError:
                    break
                if not piece:
                    break
                chunks.append(piece)
        except Exception as exc:
            raise AudioDecodeError(f"ffmpeg stdout read failed: {exc}") from exc
        if not chunks:
            return np.empty(0, dtype=np.float32)
        raw = b"".join(chunks)
        usable_len = len(raw) - (len(raw) % _BYTES_PER_SAMPLE)
        if usable_len <= 0:
            return np.empty(0, dtype=np.float32)
        pcm_i16 = np.frombuffer(raw[:usable_len], dtype=np.int16)
        return (pcm_i16.astype(np.float32)) / 32768.0

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return
        try:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except Exception:
                    pass
            await asyncio.wait_for(self._process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self._process.kill()
            await self._process.wait()
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        if self._stderr_task is not None:
            self._stderr_task.cancel()
