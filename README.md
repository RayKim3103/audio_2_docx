# Local Meeting Audio → DOCX Agent

General-domain meeting/recording audio files are transcribed and summarized into DOCX files without using external LLM APIs. The app runs as a local Gradio web UI.

## What this system does

1. Detects the hardware of the computer running the app.
2. Chooses an ASR + LLM profile.
3. Transcribes uploaded audio/video files with `faster-whisper`.
4. Summarizes the transcript with a local Hugging Face text LLM.
5. Builds Markdown and converts it to Word DOCX with Pandoc.
6. Returns a ZIP containing DOCX, Markdown, transcript, timestamped transcript, and JSON logs.

## Important architecture note

A browser URL does not run Python on the visitor's GPU/CPU. The computation runs on the computer/server where this app is launched.

- For private local use: each user runs this app on their own PC and opens `http://127.0.0.1:7860`.
- For internal LAN use: run on one workstation/server and share a LAN URL. Audio files are uploaded to that host, so use authentication.
- External public sharing is disabled by default and not recommended for confidential recordings.

## Quick start on Windows

1. Unzip this folder.
2. Double-click `run_windows.bat`.
3. Wait for `.agent_runtime` to be created and packages to be installed.
4. Browser opens `http://127.0.0.1:7860`.
5. Upload one or more audio files and click **회의록 DOCX 생성**.

For LAN use, double-click `run_lan_windows.bat` and set an id/password.

## Quick start on macOS/Linux

```bash
cd meeting_docx_agent_general_local
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

## Runtime isolation

All packages, downloaded models, pip cache, Hugging Face cache, Pandoc tools, temporary files are placed under:

```text
.agent_runtime/
```

Delete `.agent_runtime` to remove installed packages and model files:

Windows:
```powershell
Remove-Item -Recurse -Force .agent_runtime
```

macOS/Linux:
```bash
rm -rf .agent_runtime
```

Generated user outputs are under:

```text
workspace/outputs/
```

Delete `workspace` if you want to remove processed files.

## Profiles

- `auto`: hardware-based selection.
- `cpu_low`: faster-whisper base + Qwen2.5-0.5B.
- `cpu_standard`: faster-whisper small + Qwen2.5-1.5B.
- `gpu_light`: faster-whisper small + Qwen2.5-1.5B on CUDA.
- `gpu_balanced`: faster-whisper medium + Qwen2.5-3B on CUDA.
- `gpu_quality`: faster-whisper large-v3 + Qwen2.5-7B on CUDA.

Models are downloaded on first use into `.agent_runtime/models` and `.agent_runtime/hf_home`.

## Troubleshooting

### GPU exists but app uses CPU

If the hardware tab says NVIDIA GPU is detected but torch CUDA is unavailable, reinstall runtime with CUDA torch:

```powershell
Remove-Item -Recurse -Force .agent_runtime
python install.py --torch cuda
.agent_runtime\venv\Scripts\python.exe run_app.py
```

### DOCX does not open in Word

This app uses Pandoc for DOCX generation and validates the DOCX zip/XML structure. If a file still fails, check `workspace/outputs/<run_id>/notes_md/*.md` and `logs/process_results.json`.

### Model download is slow

Choose `cpu_low` first to test the pipeline. Larger GPU profiles download larger LLMs.
