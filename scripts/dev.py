#!/usr/bin/env python3
"""
Cross-platform ASR-Pro dev launcher - one command on macOS, Windows and Linux.

    python scripts/dev.py                # backend + frontend
    python scripts/dev.py --with-lab     # + legacy Streamlit ASR Lab (port 8501)
    python scripts/dev.py --backend-only

What it does:
  * Picks the right Python environment automatically (.venv if present).
  * Starts FastAPI (uvicorn, port 8000) and the Vite React frontend (port 5173).
  * Streams both processes' output with [api] / [ui] prefixes.
  * Ctrl+C stops everything cleanly on every OS (no orphaned processes).

Engine note: the ASR engine picks the best backend for the host by itself -
Apple Silicon Mac -> MLX (Metal), Windows/Linux + NVIDIA GPU -> faster-whisper
CUDA float16, otherwise CPU int8. No configuration needed.

Prerequisites (once):
    python -m venv .venv
    .venv/bin/pip install -r requirements.txt        (Windows: .venv\\Scripts\\pip)
    npm install
    cp .env.example .env   (Windows: copy .env.example .env)  and set secrets
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"


def venv_bin(name: str) -> str:
    """Resolve a command from .venv if it exists, else fall back to PATH."""
    candidates = (
        [ROOT / ".venv" / "Scripts" / f"{name}.exe", ROOT / ".venv" / "Scripts" / name]
        if IS_WINDOWS
        else [ROOT / ".venv" / "bin" / name]
    )
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which(name)
    if not found:
        sys.exit(
            f"'{name}' not found. Create the venv and install deps first:\n"
            f"  python -m venv .venv && "
            + (
                r".venv\Scripts\pip install -r requirements.txt"
                if IS_WINDOWS
                else ".venv/bin/pip install -r requirements.txt"
            )
        )
    return found


def npm_cmd() -> str:
    for name in ("npm.cmd", "npm") if IS_WINDOWS else ("npm",):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("npm not found. Install Node.js first: https://nodejs.org")


def stream(proc: subprocess.Popen, tag: str) -> None:
    for line in iter(proc.stdout.readline, b""):
        sys.stdout.write(f"[{tag}] {line.decode(errors='replace')}")
        sys.stdout.flush()


def spawn(cmd: list[str], tag: str) -> subprocess.Popen:
    # CREATE_NEW_PROCESS_GROUP lets us send CTRL_BREAK on Windows; on POSIX a
    # new session makes killpg() reap the whole tree (vite spawns esbuild etc).
    kwargs: dict = {
        "cwd": str(ROOT),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    threading.Thread(target=stream, args=(proc, tag), daemon=True).start()
    return proc


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=8)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASR-Pro dev stack.")
    parser.add_argument("--backend-only", action="store_true", help="API only (no frontend)")
    parser.add_argument("--frontend-only", action="store_true", help="Frontend only (no API)")
    parser.add_argument(
        "--with-lab", action="store_true", help="Also start the legacy Streamlit ASR Lab (8501)"
    )
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=5173)
    args = parser.parse_args()

    if not (ROOT / ".env").exists():
        sys.exit(
            ".env not found. Create it first:\n"
            + ("  copy .env.example .env" if IS_WINDOWS else "  cp .env.example .env")
            + "\nthen set ASR_JWT_SECRET_KEY and ASR_ADMIN_PASSWORD."
        )

    procs: list[subprocess.Popen] = []
    try:
        if not args.frontend_only:
            procs.append(
                spawn(
                    [
                        venv_bin("uvicorn"),
                        "asr_pro.api.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.api_port),
                        "--reload",
                        "--reload-dir",
                        "asr_pro",
                    ],
                    "api",
                )
            )
        if not args.backend_only:
            procs.append(spawn([npm_cmd(), "run", "dev", "--", "--port", str(args.ui_port)], "ui"))
        if args.with_lab:
            procs.append(
                spawn(
                    [
                        venv_bin("streamlit"),
                        "run",
                        "tools/legacy_streamlit/ASR/ASR.py",
                        "--server.address=127.0.0.1",
                        "--server.port=8501",
                    ],
                    "lab",
                )
            )

        print()
        print("  ASR-Pro dev stack")
        if not args.frontend_only:
            print(f"    API      http://localhost:{args.api_port}/api/docs")
        if not args.backend_only:
            print(f"    Frontend http://localhost:{args.ui_port}")
        if args.with_lab:
            print("    ASR Lab  http://localhost:8501")
        print("  Ctrl+C ile durdurun.")
        print()

        # Exit when any child dies; otherwise wait for Ctrl+C.
        while all(p.poll() is None for p in procs):
            try:
                procs[0].wait(timeout=1)
            except subprocess.TimeoutExpired:
                continue
        dead = next(p for p in procs if p.poll() is not None)
        print(f"\nBir süreç sonlandı (exit {dead.returncode}); diğerleri durduruluyor...")
    except KeyboardInterrupt:
        print("\nDurduruluyor...")
    finally:
        for p in procs:
            terminate(p)


if __name__ == "__main__":
    main()
