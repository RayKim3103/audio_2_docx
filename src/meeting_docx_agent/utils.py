from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable

INVALID_FILENAME_CHARS = r'<>:"/\\|?*'


def safe_name(name: str, max_len: int = 120) -> str:
    name = str(name).strip()
    for ch in INVALID_FILENAME_CHARS:
        name = name.replace(ch, "_")
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    return (name[:max_len].strip() or "untitled")


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")
    return path


def write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def seconds_to_hms(seconds: float | int | None) -> str:
    if seconds is None:
        seconds = 0
    seconds = int(max(0, float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def clean_text_for_xml(text: str) -> str:
    # XML 1.0 allows tab, newline, carriage return, and characters >= 0x20 except surrogate ranges.
    if text is None:
        return ""
    return "".join(ch for ch in str(text) if ch in "\t\n\r" or (ord(ch) >= 32 and not (0xD800 <= ord(ch) <= 0xDFFF)))


def make_zip(zip_path: Path, files: Iterable[Path], base_dir: Path | None = None) -> Path:
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            p = Path(p)
            if not p.exists() or p.is_dir():
                continue
            arc = p.relative_to(base_dir) if base_dir and p.is_relative_to(base_dir) else p.name
            zf.write(p, arcname=str(arc))
    return zip_path


def copy_to_dir(src: Path, dst_dir: Path) -> Path:
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / safe_name(src.name)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst
