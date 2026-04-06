# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-23

### Added
- React/Zustand based modern frontend dashboard.
- Extensive REST and WebSocket API documentation.
- Enterprise-grade sentiment and churn risk AI engines.
- Role-based access control (RBAC) on keywords and alerts.
- Thread-safe model loading mechanism for AI endpoints.
- Secure WebSocket endpoint for live ASR streams.
- MIT License.

### Changed
- Migrated legacy scripts to a structured FastAPI and Streamlit hybrid architecture.
- Optimized Docker builds with multi-stage configuration and `.dockerignore`.
- Enhanced test coverage using `pytest-cov`, with conditional skipping for ML-heavy operations.

### Fixed
- Addressed JWT hardcoded secret vulnerability and replaced plain-text user passwords with bcrypt hashing.
- Fixed `DetachedInstanceError` related to `SessionLocal` context management.
\n- **feat(core): initialize batched inference pipeline for higher throughput**: Implement faster-whisper batched inference to support high-volume concurrent requests.\n- **perf(engine): optimize MLX greedy decoder phonetic tokenization**: Enhance MLX token selection to reduce phonetically similar word errors in Turkish.\n- **fix(ui): resolve state mismatch in real-time transcription dashboard**: Fix a race condition in Streamlit UI where partial transcripts would overwrite finalized ones.\n- **refactor(deps): migrate from torch 1.13 to torch 2.x for better MPS support**: Upgrade pytorch dependencies to leverage native Apple Silicon acceleration.