#!/usr/bin/env bash
# ASR PRO — API sunucusunu başlat
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m uvicorn asr_pro.api.main:app --host 0.0.0.0 --port 8000 --reload
