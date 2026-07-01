"""
ATM-Net++ Full Project Startup Script
======================================
Starts:
  1. Flask demo server    → http://localhost:5000  (standalone, no auth)
  2. FastAPI backend      → http://localhost:8000  (full API + DB + auth)

Usage:
  py startup.py             # start both
  py startup.py --flask     # Flask only
  py startup.py --api       # FastAPI only
  py startup.py --frontend  # also start Next.js frontend (requires npm)
"""

import argparse
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║          ATM-Net++ Spine AI — Full Stack             ║
╠══════════════════════════════════════════════════════╣
║  Flask demo  → http://localhost:5000                 ║
║  FastAPI     → http://localhost:8000                 ║
║  API docs    → http://localhost:8000/docs            ║
║  Frontend    → http://localhost:3000  (if started)   ║
╚══════════════════════════════════════════════════════╝
""")


def run_flask():
    print("[Flask] Starting standalone demo server on :5000 ...")
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=str(ROOT),
    )
    return proc


def run_fastapi():
    print("[FastAPI] Starting backend API server on :8000 ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", "8000",
         "--reload", "--reload-dir", "backend"],
        cwd=str(ROOT),
    )
    return proc


def run_frontend():
    fe_dir = ROOT / "frontend"
    if not fe_dir.exists():
        print("[Frontend] frontend/ directory not found — skipping")
        return None
    # Check if node_modules installed
    if not (fe_dir / "node_modules").exists():
        print("[Frontend] Installing npm packages (first time) ...")
        subprocess.run(["npm", "install"], cwd=str(fe_dir), shell=True)
    print("[Frontend] Starting Next.js on :3000 ...")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(fe_dir),
        shell=True,
    )
    return proc


def wait_and_open(url: str, delay: int = 4):
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flask",    action="store_true", help="Flask only")
    parser.add_argument("--api",      action="store_true", help="FastAPI only")
    parser.add_argument("--frontend", action="store_true", help="Also start Next.js")
    args = parser.parse_args()

    print_banner()
    procs = []

    if args.flask:
        procs.append(run_flask())
        threading.Thread(target=wait_and_open, args=("http://localhost:5000",), daemon=True).start()
    elif args.api:
        procs.append(run_fastapi())
        threading.Thread(target=wait_and_open, args=("http://localhost:8000/docs",), daemon=True).start()
    else:
        # Default: both Flask + FastAPI
        procs.append(run_flask())
        procs.append(run_fastapi())
        if args.frontend:
            p = run_frontend()
            if p: procs.append(p)
        threading.Thread(target=wait_and_open, args=("http://localhost:5000",), daemon=True).start()

    print("\nPress Ctrl+C to stop all servers.\n")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping all servers...")
        for p in procs:
            try: p.terminate()
            except: pass
        print("[Shutdown] Done.")
