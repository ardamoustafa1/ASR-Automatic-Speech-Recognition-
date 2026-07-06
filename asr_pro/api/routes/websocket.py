# WebSocket endpoint for low-latency live streaming speech-to-text transcription.
from typing import Optional

"""WebSocket route for Live ASR streaming.

Security: JWT is validated from the FIRST WebSocket message (not the URL query string),
preventing tokens from appearing in browser history, proxy logs, or server access logs.

Protocol:
  1. Client connects (no token in URL).
  2. Server sends: {"type": "auth_required"}
  3. Client sends JSON: {"type": "auth", "token": "<jwt>"}
  4. Server validates token.
     - On success: {"type": "auth_ok"}
     - On failure: close with code 1008 (Policy Violation)
  5. Client streams binary WebM/Opus audio chunks.
  6. Server incrementally decodes + transcribes and replies with:
     - {"type": "partial", "text": ..., "segments": [...]}   — tentative, not persisted
     - {"type": "final", "text": ..., "segments": [...], "transcript_so_far": ...} — committed + persisted
     - {"type": "error", "message": ...} — decode/transcription failure, connection is closed after
  Server never re-transcribes audio it has already committed as final: only the bounded
  pending window is re-run on each new chunk, and VAD-detected speech pauses (or a max
  pending-window timeout) trigger a commit.
"""
import asyncio
import time

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from asr_pro.api.routes.auth import ALGORITHM, SECRET_KEY
from asr_pro.config import MAX_WS_CONNECTIONS
from asr_pro.db.models import Conversation, TranscriptSegmentRow
from asr_pro.db.session import SessionLocal
from asr_pro.observability.metrics import ws_active_connections, ws_messages_total
from asr_pro.services.audio_stream_decoder import AudioDecodeError
from asr_pro.services.streaming_session import StreamingASRSession

router = APIRouter(tags=["live-asr"])

_active_connections = 0


def _validate_token(token: str) -> Optional[dict]:
    """Validate a JWT token. Returns payload dict or None if invalid."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


def _create_conversation_sync(agent_id: str) -> str:
    db = SessionLocal()
    try:
        conv = Conversation(agent_id=agent_id, sector="omni")
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv.id
    finally:
        db.close()


def _persist_final_sync(conversation_id: str, segments: list, transcript_so_far: str) -> None:
    db = SessionLocal()
    try:
        for seg in segments:
            db.add(
                TranscriptSegmentRow(
                    conversation_id=conversation_id,
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    speaker=seg.get("speaker"),
                )
            )
        conv = db.get(Conversation, conversation_id)
        if conv is not None:
            conv.full_transcript = transcript_so_far
        db.commit()
    finally:
        db.close()


def _finalize_conversation_sync(conversation_id: str, duration_sec: float) -> None:
    db = SessionLocal()
    try:
        conv = db.get(Conversation, conversation_id)
        if conv is not None:
            conv.duration_sec = round(duration_sec, 2)
        db.commit()
    finally:
        db.close()


@router.websocket("/ws/live-asr")
async def websocket_asr_endpoint(websocket: WebSocket):
    global _active_connections
    await websocket.accept()

    if _active_connections >= MAX_WS_CONNECTIONS:
        await websocket.close(code=1013, reason="Server busy, try again later")
        return
    _active_connections += 1
    ws_active_connections.inc()

    # ── Step 1: Challenge ──────────────────────────────────────────────────────
    await websocket.send_json({"type": "auth_required"})

    username = "unknown"
    session: Optional[StreamingASRSession] = None
    conversation_id: Optional[str] = None
    try:
        # ── Step 2: Receive auth message ──────────────────────────────────────
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

        # ── Step 3: Prepare streaming session ──────────────────────────────────
        session = StreamingASRSession(language="tr")
        await session.start()
        conversation_id = await asyncio.to_thread(_create_conversation_sync, username)
        session_start = time.monotonic()

        while True:
            data = await websocket.receive_bytes()

            t0 = time.monotonic()
            try:
                message = await session.push_audio(data)
            except AudioDecodeError as exc:
                logger.warning(f"WS Live-ASR audio decode error for '{username}': {exc}")
                ws_messages_total.labels(type="error").inc()
                await websocket.send_json(
                    {"type": "error", "message": "Audio stream could not be decoded."}
                )
                break
            except Exception as exc:
                logger.warning(f"WS transcription chunk error: {exc}")
                await websocket.send_json(
                    {"type": "warning", "status": "warning", "message": f"Chunk transcription failed: {exc}"}
                )
                continue

            if message is None:
                continue

            latency_ms = int((time.monotonic() - t0) * 1000)
            elapsed = time.monotonic() - session_start
            message["latency_ms"] = latency_ms
            message["session_elapsed"] = round(elapsed, 1)

            if message["type"] in ("partial", "final"):
                from asr_pro.services.live_coaching_service import LiveCoachingService
                chunk_text = message.get("text", "") or message.get("transcript_so_far", "")
                alert = LiveCoachingService.evaluate_chunk(
                    session_id=conversation_id,
                    text=chunk_text,
                    latency_ms=latency_ms,
                    session_elapsed=elapsed,
                )
                if alert:
                    message["coaching_alert"] = alert

            ws_messages_total.labels(type=message["type"]).inc()
            await websocket.send_json(message)

            if message["type"] == "final":
                await asyncio.to_thread(
                    _persist_final_sync,
                    conversation_id,
                    message["segments"],
                    message["transcript_so_far"],
                )
                logger.debug(f"WS: committed final segment for '{username}' in {latency_ms}ms")

    except WebSocketDisconnect:
        logger.info(f"WS Live-ASR: user '{username}' disconnected.")
    except Exception as exc:
        logger.error(f"WS Live-ASR unexpected error: {exc}")
    finally:
        _active_connections -= 1
        ws_active_connections.dec()
        if session is not None:
            try:
                final_message = await session.flush_final()
                if final_message is not None and conversation_id is not None:
                    ws_messages_total.labels(type="final_flush").inc()
                    await asyncio.to_thread(
                        _persist_final_sync,
                        conversation_id,
                        final_message["segments"],
                        final_message["transcript_so_far"],
                    )
            except Exception as exc:
                logger.warning(f"WS Live-ASR: flush_final failed for '{username}': {exc}")
            if conversation_id is not None:
                try:
                    await asyncio.to_thread(
                        _finalize_conversation_sync, conversation_id, session.committed_offset_sec
                    )
                except Exception as exc:
                    logger.warning(f"WS Live-ASR: finalize conversation failed: {exc}")
            try:
                await session.close()
            except Exception:
                pass
