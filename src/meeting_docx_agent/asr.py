from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .model_store import faster_whisper_model_dir
from .paths import configure_environment
from .utils import seconds_to_hms, write_json, write_text


@dataclass
class ASROptions:
    model_name: str
    device: str
    compute_type: str
    beam_size: int = 1
    language: str | None = None
    vad_filter: bool = True
    cpu_threads: int = 0


class ASRTranscriber:
    def __init__(self) -> None:
        self._model_key: tuple[str, str, str] | None = None
        self._model = None

    def load_model(self, opts: ASROptions):
        configure_environment()
        key = (opts.model_name, opts.device, opts.compute_type)
        if self._model is not None and self._model_key == key:
            return self._model
        self._model = None
        from faster_whisper import WhisperModel
        try:
            self._model = WhisperModel(
                opts.model_name,
                device=opts.device,
                compute_type=opts.compute_type,
                download_root=str(faster_whisper_model_dir()),
                cpu_threads=opts.cpu_threads or 0,
                num_workers=1,
            )
            self._model_key = key
            return self._model
        except Exception as e:
            if opts.device == "cuda":
                # CUDA 설정/드라이버 문제로 실패하면 CPU int8로 안전하게 fallback.
                self._model = WhisperModel(
                    opts.model_name,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(faster_whisper_model_dir()),
                    cpu_threads=opts.cpu_threads or 0,
                    num_workers=1,
                )
                self._model_key = (opts.model_name, "cpu", "int8")
                return self._model
            raise RuntimeError(f"ASR 모델 로드 실패: {e}") from e

    def transcribe(self, audio_path: Path, opts: ASROptions) -> dict:
        model = self.load_model(opts)
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=None if opts.language in (None, "auto", "") else opts.language,
            beam_size=opts.beam_size,
            vad_filter=opts.vad_filter,
        )
        segments: List[Dict] = []
        for idx, seg in enumerate(segments_iter, start=1):
            text = (seg.text or "").strip()
            if not text:
                continue
            segments.append(
                {
                    "id": f"S{idx:04d}",
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "start_hms": seconds_to_hms(seg.start),
                    "end_hms": seconds_to_hms(seg.end),
                    "text": text,
                }
            )
        return {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "segments": segments,
            "text": "\n".join(s["text"] for s in segments).strip(),
        }


def segments_to_timestamped(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        lines.append(f"[{s['id']} {s['start_hms']}-{s['end_hms']}] {s['text']}")
    return "\n".join(lines)


def save_transcript_outputs(asr_result: dict, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = write_text(out_dir / f"{stem}.txt", asr_result.get("text", ""))
    timestamped = segments_to_timestamped(asr_result.get("segments", []))
    timestamped_path = write_text(out_dir / f"{stem}.timestamped.txt", timestamped)
    seg_path = write_json(out_dir / f"{stem}.segments.json", asr_result)
    return {"txt": txt_path, "timestamped": timestamped_path, "segments": seg_path}
