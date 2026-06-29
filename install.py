from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / ".agent_runtime"
VENV_DIR = RUNTIME_DIR / "venv"
PIP_CACHE_DIR = RUNTIME_DIR / "pip_cache"
INSTALL_LOG_DIR = RUNTIME_DIR / "install_logs"


def venv_python() -> Path:
    if platform.system().lower().startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd: list[str], name: str, check: bool = True) -> int:
    INSTALL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = INSTALL_LOG_DIR / f"{name}.log"
    print("$", " ".join(cmd))
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(f"❌ 실패: {log_path}")
        if check:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        print(f"✅ 완료: {name}")
    return proc.returncode


def has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, timeout=5)
        return True
    except Exception:
        return False


def create_venv(reset: bool = False) -> Path:
    if reset and VENV_DIR.exists():
        print(f"🧹 기존 runtime 삭제: {RUNTIME_DIR}")
        shutil.rmtree(RUNTIME_DIR, ignore_errors=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    PIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    py = venv_python()
    if not py.exists():
        print(f"🔧 venv 생성: {VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)], "create_venv")
    if not py.exists():
        raise FileNotFoundError(f"venv python을 찾지 못했습니다: {py}")
    return py


def install_torch(py: Path, torch_mode: str) -> None:
    if torch_mode == "auto":
        torch_mode = "cuda" if has_nvidia_gpu() else "cpu"
    if torch_mode == "skip":
        print("⏭️ torch 설치 건너뜀")
        return
    if torch_mode == "cuda":
        print("🔧 PyTorch CUDA wheel 설치 시도")
        rc = run([str(py), "-m", "pip", "install", "--cache-dir", str(PIP_CACHE_DIR), "torch", "--index-url", "https://download.pytorch.org/whl/cu121"], "pip_torch_cuda", check=False)
        if rc == 0:
            return
        print("⚠️ CUDA torch 설치 실패. CPU torch로 fallback합니다.")
    print("🔧 PyTorch CPU wheel 설치")
    run([str(py), "-m", "pip", "install", "--cache-dir", str(PIP_CACHE_DIR), "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], "pip_torch_cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Delete .agent_runtime before install")
    ap.add_argument("--torch", choices=["auto", "cpu", "cuda", "skip"], default="auto")
    args = ap.parse_args()

    py = create_venv(reset=args.reset)
    req = PROJECT_ROOT / "requirements" / "base.txt"
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], "pip_upgrade")
    install_torch(py, args.torch)
    run([str(py), "-m", "pip", "install", "--cache-dir", str(PIP_CACHE_DIR), "-r", str(req)], "pip_base")
    print("\n✅ 설치 완료")
    print(f"실행: {py} run_app.py")


if __name__ == "__main__":
    main()
