from __future__ import annotations

import os
import tempfile
from pathlib import Path


def find_project_root() -> Path:
    env = os.environ.get("AUDIO_DOCX_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # paths.py -> src/meeting_docx_agent/paths.py, project root is parents[2]
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = find_project_root()
RUNTIME_DIR = Path(os.environ.get("AUDIO_DOCX_RUNTIME_DIR", PROJECT_ROOT / ".agent_runtime")).resolve()
VENV_DIR = RUNTIME_DIR / "venv"
WORKSPACE_DIR = Path(os.environ.get("AUDIO_DOCX_WORKSPACE_DIR", PROJECT_ROOT / "workspace")).resolve()
INPUT_DIR = WORKSPACE_DIR / "inputs"
OUTPUT_DIR = WORKSPACE_DIR / "outputs"
LOG_DIR = WORKSPACE_DIR / "logs"
MODEL_DIR = RUNTIME_DIR / "models"
HF_HOME = RUNTIME_DIR / "hf_home"
HF_HUB_CACHE = HF_HOME / "hub"
HF_XET_CACHE = HF_HOME / "xet"
HF_ASSETS_CACHE = HF_HOME / "assets"
PIP_CACHE_DIR = RUNTIME_DIR / "pip_cache"
TORCH_HOME = RUNTIME_DIR / "torch_home"
GRADIO_TEMP_DIR = RUNTIME_DIR / "gradio_tmp"
TMP_DIR = RUNTIME_DIR / "tmp"
PANDOC_DIR = RUNTIME_DIR / "tools" / "pandoc"
REFERENCE_DIR = RUNTIME_DIR / "reference_docx"


def ensure_dirs() -> None:
    for p in [
        RUNTIME_DIR,
        WORKSPACE_DIR,
        INPUT_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        MODEL_DIR,
        HF_HOME,
        HF_HUB_CACHE,
        HF_XET_CACHE,
        HF_ASSETS_CACHE,
        PIP_CACHE_DIR,
        TORCH_HOME,
        GRADIO_TEMP_DIR,
        TMP_DIR,
        PANDOC_DIR,
        REFERENCE_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def configure_environment() -> None:
    """Force caches and temporary files into the project-local runtime folder.

    This function must run before importing transformers / huggingface_hub / gradio.
    """
    ensure_dirs()
    env_map = {
        "HF_HOME": HF_HOME,
        "HF_HUB_CACHE": HF_HUB_CACHE,
        "HF_XET_CACHE": HF_XET_CACHE,
        "HF_ASSETS_CACHE": HF_ASSETS_CACHE,
        # Backward-compatible variables used by older packages.
        "HUGGINGFACE_HUB_CACHE": HF_HUB_CACHE,
        "HUGGINGFACE_ASSETS_CACHE": HF_ASSETS_CACHE,
        "TRANSFORMERS_CACHE": HF_HUB_CACHE,
        "TORCH_HOME": TORCH_HOME,
        "PIP_CACHE_DIR": PIP_CACHE_DIR,
        "GRADIO_TEMP_DIR": GRADIO_TEMP_DIR,
        "TMPDIR": TMP_DIR,
        "TEMP": TMP_DIR,
        "TMP": TMP_DIR,
        "XDG_CACHE_HOME": RUNTIME_DIR / "xdg_cache",
    }
    for key, value in env_map.items():
        os.environ[key] = str(value)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    tempfile.tempdir = str(TMP_DIR)


SUPPORTED_AUDIO_EXTS = {
    ".mp3", ".m4a", ".wav", ".flac", ".aac", ".ogg", ".opus", ".webm", ".mp4", ".mov", ".mkv"
}
