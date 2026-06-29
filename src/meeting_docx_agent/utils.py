from __future__ import annotations

import ast
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



def humanize_llm_value(value: Any, max_len: int = 1200) -> str:
    """Convert accidental JSON/Python-literal-looking LLM text into readable prose.

    Small local LLMs sometimes return a dictionary/list as a *string*, e.g.
    "{'heading': '회의록 제목', 'bullets': ['회의 날짜: ...']}".  This helper
    keeps validation tolerant while preventing those raw literal strings from
    leaking into Markdown/DOCX.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [humanize_llm_value(v, max_len=max_len) for v in value]
        return "; ".join(p for p in parts if p)[:max_len]
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ["heading", "title", "topic", "name", "term"]:
            v = value.get(key)
            if v:
                s = humanize_llm_value(v, max_len=200)
                if s:
                    parts.append(s)
                    break
        for key in ["summary", "text", "point", "description", "desc", "result", "follow_up", "후속 조치", "주요 내용", "결과"]:
            v = value.get(key)
            if v:
                s = humanize_llm_value(v, max_len=max_len)
                if s:
                    parts.append(s)
        for key in ["bullets", "details", "items", "key_points", "내용"]:
            v = value.get(key)
            if v:
                s = humanize_llm_value(v, max_len=max_len)
                if s:
                    parts.append(s)
        # Remove duplicates while preserving order.
        uniq: list[str] = []
        for p in parts:
            p = re.sub(r"\s+", " ", p).strip()
            if p and p not in uniq:
                uniq.append(p)
        return " / ".join(uniq)[:max_len]

    s = clean_text_for_xml(str(value)).strip()
    # Parse accidental literal strings only when the entire string looks like a literal.
    stripped = s.strip()
    if len(stripped) <= 6000 and stripped[:1] in "[{" and stripped[-1:] in "]}":
        for parser in (ast.literal_eval, json.loads):
            try:
                parsed = parser(stripped)
                if parsed is not value:
                    return humanize_llm_value(parsed, max_len=max_len)
            except Exception:
                pass
    return re.sub(r"\s+", " ", s)[:max_len]


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
