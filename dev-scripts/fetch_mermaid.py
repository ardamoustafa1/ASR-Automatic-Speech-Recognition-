import base64
import json
import os
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

mermaid_code = """
graph TD
    User((Call Center Agent))
    Client[React Web Application]
    API[FastAPI Gateway]
    ASREngine[ASR Service]
    Compliance[Compliance Engine]
    Trend[Trend & Analytics Engine]
    DB[(SQLite Database)]
    AuthDB[(Auth Database)]

    User -->|Uses| Client
    Client -->|REST & WebSockets| API
    API -->|Audio Chunks| ASREngine
    ASREngine -->|Transcripts| API
    API -->|Analyze| Compliance
    API -->|Trends| Trend
    API -->|Auth| AuthDB
    Trend -->|Write| DB
    Compliance -->|Save| DB
"""

payload = {
    "code": mermaid_code,
    "mermaid": {"theme": "default"}
}

json_payload = json.dumps(payload).encode('utf-8')
b64_payload = base64.urlsafe_b64encode(json_payload).decode('utf-8')

url = f"https://mermaid.ink/img/{b64_payload}"
try:
    os.makedirs('docs/assets', exist_ok=True)
    urllib.request.urlretrieve(url, 'docs/assets/architecture.png')
    print("architecture.png saved successfully.")
except Exception as e:
    print(f"Failed to fetch image: {e}")
    # Create an empty file as fallback
    with open('docs/assets/architecture.png', 'wb') as f:
        pass
