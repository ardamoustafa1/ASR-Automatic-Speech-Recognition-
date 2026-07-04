# 🛡️ Privacy & Data Retention Policy (GDPR / KVKK)

ASR-Pro is designed with enterprise-grade security and compliance at its core. This document outlines how voice and text data are handled, processed, and destroyed.

## 1. Data Processing (Zero-Trust Architecture)
- **Audio Processing:** ASR-Pro processes audio entirely on-premise or within your isolated virtual private cloud (VPC). Unless explicitly configured, no audio data is sent to external APIs (OpenAI, Google, etc.).
- **In-Memory Analysis:** Voice streams sent via WebSocket are transcribed in memory and discarded immediately unless storage is explicitly enabled.

## 2. Data Retention Policies
By default, the following retention rules apply:
- **Audio Recordings (`/temp_audio_uploads`):** 24 Hours. A chron job automatically purges audio files older than 24 hours.
- **Transcripts (`conversations` table):** 365 Days (Customizable via `DATA_RETENTION_DAYS` env var).
- **Compliance Metadata:** Retained indefinitely for aggregated anonymized trends.

## 3. Right to be Forgotten (GDPR / KVKK)
Administrators can purge specific `customer_id` or `conversation_id` records. This hard-deletes the transcript and all associated `KeywordHit` and `AuditLog` metadata.

## 4. Audit Logging
Every action (login, manual deletion, configuration change) performed by an authenticated user is logged in the `audit_logs` table. This guarantees a 100% auditable trail of who accessed or deleted what data.

## 5. Security Measures
- All REST API endpoints are protected by JWT Bearer tokens.
- Passwords are hashed using bcrypt.
- The `AuditLog` table is append-only by the application logic.

-->
