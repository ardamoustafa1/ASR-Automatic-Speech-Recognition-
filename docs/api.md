# ASR-Pro API Reference

This document provides examples for interacting with the ASR-Pro FastAPI endpoints.

## Base URL

All API requests should be prefixed with:
`http://localhost:8000/api/v1`

## Authentication

Most endpoints require a JWT bearer token. Obtain one by logging in.

### Login

**Endpoint:** `POST /auth/login`

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=admin&password=your_password_here"
```

**Response:**

```json
{
  "access_token": "eyJhbGci...<jwt_token>",
  "token_type": "bearer"
}
```

## Endpoints

### 1. Queue Conversation Analysis

**Endpoint:** `POST /api/v1/conversations/analyze`
Queues already-transcribed segments for NLP analysis and persistence. Use the
Streamlit UI on `:8501` or your own ASR client to create the transcript segments
first, then send them to this endpoint.

**Python Example:**

```python
import requests

url = "http://localhost:8000/api/v1/conversations/analyze"
headers = {"Authorization": "Bearer YOUR_JWT_TOKEN"}
payload = {
    "sector": "banking",
    "full_transcript": "Kredili mevduat hesabımı kapatmak istiyorum.",
    "segments": [
        {"start": 0.0, "end": 4.2, "text": "Kredili mevduat hesabımı kapatmak istiyorum."}
    ],
    "asr_confidence": 94.5,
    "quality_gate_passed": True,
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

**Response:**

```json
{
  "message": "Analiz işlemi arka plana alındı.",
  "status": "processing"
}
```

### 2. Analyze Raw Text

**Endpoint:** `POST /api/v1/conversations/analyze-text`
Runs keyword and topic detection for a raw text snippet without saving a
conversation.

```bash
curl -X POST "http://localhost:8000/api/v1/conversations/analyze-text" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text":"Fatura itirazı için arıyorum.","sector":"omni"}'
```

### 3. Live ASR WebSocket

**Endpoint:** `WS /ws/live-asr`
Streams audio chunks and returns live transcription deltas. Uses initial JSON payload for authentication.

**JavaScript Example:**

```javascript
const socket = new WebSocket("ws://localhost:8000/ws/live-asr");

socket.onopen = () => {
  // Send auth message first
  socket.send(JSON.stringify({ type: "auth", token: "YOUR_JWT_TOKEN" }));
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "auth_ok") {
    console.log("Authenticated. You can now send audio blobs.");
    // Send audio chunks (e.g. from MediaRecorder)
    // socket.send(audioBlob);
  } else if (data.type === "transcript") {
    console.log("Current Transcript: ", data.transcript);
  }
};
```

### 4. Get Conversations

**Endpoint:** `GET /conversations`

```bash
curl -X GET "http://localhost:8000/api/v1/conversations?limit=10" \
     -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Webhooks

ASR-Pro supports firing webhooks when a critical alert is triggered. Configure `WEBHOOK_URL` in your `.env` file.

-->
