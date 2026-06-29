from __future__ import annotations

import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .asr import ASROptions, ASRTranscriber, save_transcript_outputs
from .docx_exporter import markdown_to_docx
from .markdown_builder import build_markdown
from .paths import OUTPUT_DIR, SUPPORTED_AUDIO_EXTS, configure_environment
from .profiles import get_profile
from .summarizer import summarize_segments
from .utils import copy_to_dir, make_zip, safe_name, write_json, write_text


@dataclass
class PipelineOptions:
    profile_name: str = "auto"
    language: str = "ko"
    output_language: str = "ko"
    glossary: str = ""
    document_detail_level: str = "detailed"  # brief, standard, detailed
    include_transcript_appendix: bool = False
    font_size_pt: int = 8
    font_name: str = "Malgun Gothic"
    allow_model_download: bool = True
    use_final_llm: bool = True
    skip_existing_asr: bool = True


class MeetingDocxAgent:
    def __init__(self) -> None:
        configure_environment()
        self.transcriber = ASRTranscriber()

    def process_files(
        self,
        audio_paths: Iterable[Path],
        options: PipelineOptions,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> dict:
        run_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        run_dir = OUTPUT_DIR / run_id
        upload_dir = run_dir / "uploads"
        transcript_dir = run_dir / "transcripts"
        segment_dir = run_dir / "segments"
        md_dir = run_dir / "notes_md"
        docx_dir = run_dir / "docx"
        json_dir = run_dir / "json"
        log_dir = run_dir / "logs"
        for d in [upload_dir, transcript_dir, segment_dir, md_dir, docx_dir, json_dir, log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        results = []
        files_to_zip: list[Path] = []
        profile = get_profile(options.profile_name)
        if log_cb:
            log_cb(f"🧭 선택 프로필: {profile.name} ({profile.label})")
            log_cb(f"🎧 ASR: {profile.asr_model} / {profile.asr_device} / {profile.asr_compute_type}")
            log_cb(f"🤖 LLM: {profile.llm_model} / {profile.llm_device}")
            log_cb(f"📝 문서 상세도: {options.document_detail_level}")

        audio_list = [Path(p) for p in audio_paths]
        for idx, src in enumerate(audio_list, start=1):
            if src.suffix.lower() not in SUPPORTED_AUDIO_EXTS:
                results.append({"audio": str(src), "status": "skipped", "reason": "unsupported extension"})
                continue
            stem = safe_name(src.stem)
            try:
                if log_cb:
                    log_cb(f"\n[{idx}/{len(audio_list)}] 📥 입력 처리: {src.name}")
                audio = copy_to_dir(src, upload_dir)
                asr_opts = ASROptions(
                    model_name=profile.asr_model,
                    device=profile.asr_device,
                    compute_type=profile.asr_compute_type,
                    beam_size=profile.asr_beam_size,
                    language=None if options.language in ("auto", "", None) else options.language,
                )
                if log_cb:
                    log_cb(f"🎙️ 전사 시작: {audio.name}")
                asr_result = self.transcriber.transcribe(audio, asr_opts)
                transcript_paths = save_transcript_outputs(asr_result, transcript_dir, stem)
                segments_copy = segment_dir / f"{stem}.segments.json"
                shutil.copy2(transcript_paths["segments"], segments_copy)
                if log_cb:
                    log_cb(f"✅ 전사 완료: segment {len(asr_result['segments'])}개")

                if log_cb:
                    log_cb("🧠 LLM 기반 회의록 정리 시작")
                summary = summarize_segments(
                    asr_result["segments"],
                    title=stem,
                    profile=profile,
                    language=options.output_language,
                    glossary=options.glossary,
                    allow_download=options.allow_model_download,
                    use_final_llm=options.use_final_llm,
                    detail_level=options.document_detail_level,
                    log_cb=log_cb,
                )
                json_path = write_json(json_dir / f"{stem}.summary.json", summary)
                run_config_path = write_json(json_dir / f"{stem}.run_config.json", summary.get("run_config", {}))
                md = build_markdown(
                    stem,
                    summary["final"],
                    asr_result["segments"],
                    include_transcript_appendix=options.include_transcript_appendix,
                    detail_level=options.document_detail_level,
                    run_config=summary.get("run_config", {}),
                )
                md_path = write_text(md_dir / f"{stem}.md", md)
                if log_cb:
                    log_cb(f"📄 Markdown 생성 완료: {len(md)} chars")
                docx_path = markdown_to_docx(md_path, docx_dir / f"{stem}.docx", font_size_pt=options.font_size_pt, font_name=options.font_name)
                if log_cb:
                    log_cb(f"✅ DOCX 생성 완료: {docx_path.name}")

                result = {
                    "audio": str(audio),
                    "status": "success",
                    "docx": str(docx_path),
                    "markdown": str(md_path),
                    "transcript": str(transcript_paths["txt"]),
                    "timestamped": str(transcript_paths["timestamped"]),
                    "segments": str(segments_copy),
                    "summary_json": str(json_path),
                    "run_config_json": str(run_config_path),
                }
                results.append(result)
                files_to_zip.extend([docx_path, md_path, transcript_paths["txt"], transcript_paths["timestamped"], segments_copy, json_path, run_config_path])
            except Exception as e:
                err = {"audio": str(src), "status": "failed", "error": repr(e)}
                results.append(err)
                if log_cb:
                    log_cb(f"❌ 실패: {src.name}: {repr(e)}")
        results_path = write_json(log_dir / "process_results.json", results)
        files_to_zip.append(results_path)
        zip_path = make_zip(run_dir / "audio_2_docx_outputs.zip", files_to_zip, base_dir=run_dir)
        return {"run_dir": str(run_dir), "zip": str(zip_path), "results": results}
