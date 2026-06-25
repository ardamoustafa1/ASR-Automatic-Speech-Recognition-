from typing import Optional

"""WebSocket route for Live ASR streaming.

Security: JWT is validated from the FIRST WebSocket message (not the URL query string),
preventing tokens from appearing in browser history, proxy logs, or server access logs.
"""
import asyncio
import os
import tempfile
import time

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from asr_pro.api.routes.auth import ALGORITHM, SECRET_KEY
from asr_pro.services.asr_service import ASRService

router = APIRouter(tags=["live-asr"])

MIN_CHUNK_SIZE = 64 * 1024  # 64 KB — avoids O(n²) re-transcription on every tiny packet


def _validate_token(token: str) -> Optional[dict]:
    """Validate a JWT token. Returns payload dict or None if invalid."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


@router.websocket("/ws/live-asr")
async def websocket_asr_endpoint(websocket: WebSocket):
    """
    Live ASR WebSocket endpoint.

    Protocol:
      1. Client connects (no token in URL).
      2. Server sends: {"type": "auth_required"}
      3. Client sends JSON: {"type": "auth", "token": "<jwt>"}
      4. Server validates token.
         - On success: {"type": "auth_ok"}
         - On failure: close with code 1008 (Policy Violation)
      5. Client streams binary audio chunks.
      6. Server responds with transcription JSON after each chunk batch.
    """
    await websocket.accept()

    # ── Step 1: Challenge ──────────────────────────────────────────────────────
    await websocket.send_json({"type": "auth_required"})

    # ── Step 2: Receive auth message ──────────────────────────────────────────
    try:
        auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except (asyncio.TimeoutError, Exception):
        await websocket.close(code=1008, reason="Authentication timeout")
        return

    if auth_msg.get("type") != "auth" or not auth_msg.get("token"):
        await websocket.close(code=1008, reason="Invalid auth message")
        return

    payload = _validate_token(auth_msg["token"])
    if payload is None:
        await websocket.close(code=1008, reason="Invalid or expired token")
        return

    username = payload.get("sub", "unknown")
    logger.info(f"WS Live-ASR: user '{username}' authenticated.")
    await websocket.send_json({"type": "auth_ok", "user": username})

    # ── Step 3: Prepare ASR ───────────────────────────────────────────────────
    asr = ASRService.get_instance()
    asr.load_model("turbo")

    temp_file_path: Optional[str] = None
    try:
        fd, temp_file_path = tempfile.mkstemp(suffix=".webm")
        os.close(fd)

        chunk_buffer = b""
        session_start = time.monotonic()

        while True:
            data = await websocket.receive_bytes()
            chunk_buffer += data

            # Wait until we have a meaningful audio batch before transcribing
            if len(chunk_buffer) < MIN_CHUNK_SIZE:
                continue

            with open(temp_file_path, "ab") as f:
                f.write(chunk_buffer)
            chunk_buffer = b""

            try:
                t0 = time.monotonic()
                segments, duration = await asyncio.to_thread(asr.transcribe, temp_file_path)
                latency_ms = int((time.monotonic() - t0) * 1000)
                elapsed = time.monotonic() - session_start

                current_text = " ".join(s.text for s in segments)
                await websocket.send_json({
                    "type": "transcript",
                    "status": "success",
                    "transcript": current_text,
                    "duration": round(duration, 2),
                    "session_elapsed": round(elapsed, 1),
                    "latency_ms": latency_ms,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in segments
                    ],
                })
                logger.debug(f"WS: transcribed {len(segments)} segs in {latency_ms}ms")

            except Exception as exc:
                logger.warning(f"WS transcription chunk error: {exc}")
                await websocket.send_json({
                    "type": "warning",
                    "status": "warning",
                    "message": f"Chunk transcription failed: {str(exc)}",
                })

    except WebSocketDisconnect:
        logger.info(f"WS Live-ASR: user '{username}' disconnected.")
    except Exception as exc:
        logger.error(f"WS Live-ASR unexpected error: {exc}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
