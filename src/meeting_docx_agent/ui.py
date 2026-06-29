from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .hardware import detect_hardware
from .paths import OUTPUT_DIR, PROJECT_ROOT, RUNTIME_DIR, WORKSPACE_DIR, configure_environment
from .pipeline import MeetingDocxAgent, PipelineOptions
from .profiles import PROFILES


def _file_path(obj: Any) -> Path:
    if obj is None:
        raise ValueError("빈 파일")
    if isinstance(obj, (str, os.PathLike)):
        return Path(obj)
    if isinstance(obj, dict):
        return Path(obj.get("path") or obj.get("name"))
    return Path(getattr(obj, "name", str(obj)))


def build_profile_markdown() -> str:
    lines = ["### 사용 가능한 프로필"]
    for name, p in PROFILES.items():
        lines.append(f"- **{name}**: {p.label} — ASR `{p.asr_model}`, LLM `{p.llm_model}`. {p.description}")
    return "\n".join(lines)


def build_ui():
    configure_environment()
    import gradio as gr

    agent = MeetingDocxAgent()
    state_logs: list[str] = []

    def get_status():
        hw = detect_hardware()
        return hw.to_markdown() + f"\n\n### 로컬 저장 위치\n- Project: `{PROJECT_ROOT}`\n- Runtime/packages/models/cache: `{RUNTIME_DIR}`\n- Outputs: `{OUTPUT_DIR}`"

    def process(files, profile, language, output_language, glossary, detail_level, include_transcript, font_size, allow_download, use_final_llm, progress=gr.Progress(track_tqdm=False)):
        if not files:
            return "오디오 파일을 업로드하세요.", None, ""
        if not isinstance(files, list):
            files = [files]
        paths = [_file_path(f) for f in files]
        logs: list[str] = []
        def log_cb(msg: str):
            logs.append(msg)
            progress(0, desc=msg[:80])
        opts = PipelineOptions(
            profile_name=profile,
            language=language,
            output_language=output_language,
            glossary=glossary or "",
            document_detail_level=detail_level or "detailed",
            include_transcript_appendix=bool(include_transcript),
            font_size_pt=int(font_size),
            allow_model_download=bool(allow_download),
            use_final_llm=bool(use_final_llm),
        )
        result = agent.process_files(paths, opts, log_cb=log_cb)
        ok_count = sum(1 for r in result["results"] if r.get("status") == "success")
        fail_count = sum(1 for r in result["results"] if r.get("status") == "failed")
        summary = f"✅ 완료: 성공 {ok_count}개, 실패 {fail_count}개\n결과 폴더: {result['run_dir']}"
        return summary, result["zip"], "\n".join(logs)

    with gr.Blocks(title="Local Meeting Audio → DOCX Agent", analytics_enabled=False) as demo:
        gr.Markdown(
            """
# Local Meeting Audio → DOCX Agent

보안이 중요한 환경을 위해 **외부 LLM API 없이**, 현재 실행 중인 컴퓨터에서 오디오를 전사하고 회의록 DOCX를 생성합니다.

중요: 브라우저 URL로 접속하더라도 연산은 이 앱이 실행되는 컴퓨터에서 수행됩니다. 각 사용자의 PC 자원을 쓰려면 각 사용자가 자기 PC에서 이 앱을 실행해야 합니다.
""".strip()
        )
        with gr.Tab("1. 사양 분석"):
            status = gr.Markdown(value=get_status())
            gr.Button("사양 다시 분석").click(fn=get_status, outputs=status)
            gr.Markdown(build_profile_markdown())
        with gr.Tab("2. 회의 오디오 → DOCX"):
            with gr.Row():
                with gr.Column(scale=1):
                    files = gr.File(label="오디오 파일 업로드", file_count="multiple", file_types=["audio", "video"])
                    profile = gr.Dropdown(label="실행 프로필", choices=["auto"] + list(PROFILES.keys()), value="auto")
                    language = gr.Dropdown(label="음성 언어", choices=["auto", "ko", "en", "ja", "zh", "es", "fr", "de"], value="ko")
                    output_language = gr.Dropdown(label="문서 출력 언어", choices=["ko", "en"], value="ko")
                    glossary = gr.Textbox(label="고유명사/전문용어 힌트(선택)", lines=3, placeholder="예: 프로젝트명, 회사명, 약어, 참석자 이름 등")
                    detail_level = gr.Dropdown(label="문서 상세도", choices=[("간단 요약", "brief"), ("표준 회의록", "standard"), ("상세 회의록", "detailed")], value="detailed", info="결과가 빈약하면 '상세 회의록'을 사용하세요. CPU에서는 시간이 더 걸릴 수 있습니다.")
                    with gr.Accordion("고급 옵션", open=False):
                        include_transcript = gr.Checkbox(label="전체 transcript 부록 포함", value=False)
                        font_size = gr.Slider(label="DOCX 글씨 크기(pt)", minimum=8, maximum=12, step=1, value=8)
                        allow_download = gr.Checkbox(label="필요한 모델이 없으면 다운로드 허용", value=True)
                        use_final_llm = gr.Checkbox(label="최종 병합에도 LLM 사용", value=True)
                    run_btn = gr.Button("회의록 DOCX 생성", variant="primary")
                with gr.Column(scale=1):
                    summary = gr.Textbox(label="상태", lines=5)
                    output_zip = gr.File(label="결과 ZIP 다운로드")
                    logs = gr.Textbox(label="실행 로그", lines=20)
            run_btn.click(
                fn=process,
                inputs=[files, profile, language, output_language, glossary, detail_level, include_transcript, font_size, allow_download, use_final_llm],
                outputs=[summary, output_zip, logs],
            )
        with gr.Tab("3. 보안/운영 안내"):
            gr.Markdown(
                f"""
## 로컬 저장 구조

- 패키지/모델/cache/runtime: `{RUNTIME_DIR}`
- 입력/결과 작업 폴더: `{WORKSPACE_DIR}`

`.agent_runtime` 폴더를 삭제하면 이 앱이 설치/다운로드한 패키지, 모델, cache를 한 번에 정리할 수 있습니다.

## URL 운영 방식

- 개인 로컬: `http://127.0.0.1:7860`
- 내부망 공유: 실행 시 `--lan --auth user:password` 사용
- 외부 공개 공유는 보안상 권장하지 않습니다.
""".strip()
            )
    return demo
