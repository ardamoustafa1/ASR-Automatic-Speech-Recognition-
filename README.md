<div align="center">

# ASR-Pro: Enterprise Speech Intelligence Platform
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react)](https://react.dev/)
[![Docker Pulls](https://img.shields.io/docker/pulls/ardamoustafa/asr-pro-backend.svg)](https://hub.docker.com/r/ardamoustafa/asr-pro-backend)
[![Coverage](https://img.shields.io/badge/Coverage-90%25-brightgreen.svg)]()
[![GitHub Stars](https://img.shields.io/github/stars/ardamoustafa/ASR-Pro?style=social)](https://github.com/ardamoustafa/ASR-Pro/stargazers)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/ardamoustafa/ASR-Pro)](https://github.com/ardamoustafa/ASR-Pro/commits/main)
[![GitHub Open Issues](https://img.shields.io/github/issues/ardamoustafa/ASR-Pro)](https://github.com/ardamoustafa/ASR-Pro/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An advanced, zero-shot AI-powered speech intelligence platform designed for enterprise contact centers. Converts speech to text and analyzes sentiment, churn risk, compliance, and empathy in real-time.*

[Türkçe Dokümantasyon (Turkish README)](README_TR.md) • [API Documentation](docs/api.md) • [Architecture](docs/ARCHITECTURE.md)

<img src="docs/assets/demo.gif" alt="ASR Pro Demo" width="800" />

</div>

---

## 🌟 Why ASR-Pro?

ASR-Pro is not just another transcription service. It is a full-fledged customer intelligence platform designed to catch compliance breaches, predict customer churn, and monitor agent soft-skills in **real-time** or in batch mode. We engineered it to handle high-concurrency enterprise workloads with built-in thread safety and sliding-window websocket performance.

### ✨ Feature Highlights
- **🎙️ Real-Time ASR (WebSockets):** Streaming audio transcription with sub-second latency and O(1) buffer optimization.
- **🧠 Zero-Shot AI Engines:** Uses HuggingFace and local LLMs to detect sentiment, churn risk, and empathy without massive fine-tuning.
- **🛡️ Compliance & QA:** Automatically scores agent compliance against custom, fuzzy-matched keyword rules.
- **📊 Real-time Analytics Dashboard:** Built with React, Zustand, and Recharts, presenting real-time risk scores and trend anomalies.
- **🔐 Enterprise Ready:** JWT-based authentication, PBKDF2/Bcrypt password hashing, strict RBAC, and SQLite/PostgreSQL support.

---

## ⚡ Quick Start

Start the entire platform (Backend + Frontend + Database) using Docker Compose in one command:

```bash
git clone https://github.com/ardamoustafa/ASR-Pro.git
cd ASR-Pro

# Environment Variables
cp .env.example .env
# Edit .env and ensure JWT_SECRET_KEY is set!

# Start services
docker-compose up -d
```

The **React Dashboard** will be available at `http://localhost:5173`.  
The **API Documentation** (Swagger UI) will be available at `http://localhost:8000/api/docs`.

### Local Development (Without Docker)

You can use the provided Makefile to run the development environment:

```bash
# 1. Install backend & frontend dependencies
pip install -r requirements.txt
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env and set ASR_JWT_SECRET_KEY and ASR_ADMIN_PASSWORD

# 3. Initialize the database and seed default data
python -m asr_pro.db.seed

# 4. Run backend (FastAPI on :8000) and frontend (Vite on :5173)
make dev
```

---

## 🛠️ Architecture

ASR-Pro consists of loosely-coupled intelligent engines and a strict Separation of Concerns (SoC) methodology.

- **FastAPI Route Engine**: Handles live audio streams and REST endpoints.
- **ASR Service**: Thread-safe Singleton utilizing `faster-whisper`.
- **Compliance Engine**: Fuzzy-matches keywords to ensure agents mention mandatory legal statements.
- **Sentiment & Churn Engines**: NLP models to detect user frustration or anger and calculate early-warning churn probability.
- **Trend Engine**: Aggregates topic occurrences to predict anomalies and forecast call volumes.

For an in-depth view of the system diagrams, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 📚 API Reference

For integration purposes, ASR-Pro provides a comprehensive REST and WebSocket API. 
Read the full API Documentation at [docs/api.md](docs/api.md).

---

## 🗺️ Roadmap

- [x] JWT Authentication & RBAC
- [x] O(1) Delta Streaming for WebSockets
- [x] React Frontend with Zustand State Management
- [x] Dockerization & CI/CD Pipelines
- [ ] Multi-tenant support (SaaS mode)
- [ ] Live Agent Assist (LLM-based suggestion prompts)
- [ ] Native mobile SDKs for iOS/Android

---

## 🤝 Contributing

We welcome contributions from the community! Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming conventions, Conventional Commits, and local development setup.

Make sure to read the [SECURITY.md](SECURITY.md) before reporting any vulnerabilities.

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ardamoustafa/ASR-Pro&type=Date)](https://star-history.com/#ardamoustafa/ASR-Pro&Date)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
