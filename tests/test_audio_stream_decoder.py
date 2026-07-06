import subprocess
import wave

import numpy as np
import pytest

from asr_pro.services.audio_stream_decoder import _FFMPEG_BIN, AudioStreamDecoder

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
