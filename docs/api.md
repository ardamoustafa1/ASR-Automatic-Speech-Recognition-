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

### 1. Analyze Audio File

**Endpoint:** `POST /api/v1/conversations/analyze`
Upload an audio file to get the transcription and NLP analysis.

**Python Example:**
```python
import requests

url = "http://localhost:8000/api/v1/conversations/analyze"
headers = {"Authorization": "Bearer YOUR_JWT_TOKEN"}
files = {"file": open("test_audio.wav", "rb")}
data = {"sector": "banking"}

response = requests.post(url, headers=headers, files=files, data=data)
print(response.json())
```

**Response:**
```json
{
  "status": "success",
  "transcription": "Kredili mevduat hesabımı kapatmak istiyorum.",
  "duration_seconds": 15.2,
  "analysis": {
    "sentiment": "negative",
    "churn_risk": 85.5,
    "topics": ["account_closure"]
  }
}
```

### 2. Live ASR WebSocket

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

### 3. Get Conversations

**Endpoint:** `GET /conversations`

```bash
curl -X GET "http://localhost:8000/api/v1/conversations?limit=10" \
     -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Webhooks

ASR-Pro supports firing webhooks when a critical alert is triggered. Configure `WEBHOOK_URL` in your `.env` file.
