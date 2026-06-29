# Local Meeting Audio → DOCX Agent

A local AI agent that converts general-domain meeting or recording audio files into structured DOCX meeting notes.

This project runs without external LLM APIs such as OpenAI, Gemini, or Claude API.
Audio transcription and summarization are performed on the computer or server where this app is launched.

---

## 1. What this system does

This app provides a local Gradio web UI for converting audio/video recordings into structured Word documents.

The pipeline is:

```text
Audio / Video file
→ Hardware detection
→ ASR + LLM profile selection
→ faster-whisper transcription
→ Local Hugging Face LLM summarization
→ Markdown generation
→ Pandoc DOCX conversion
→ ZIP download
```

The final ZIP contains:

```text
DOCX meeting note
Markdown note
plain transcript
timestamped transcript
structured JSON logs
processing logs
```

---

## 2. Key features

* Local web UI based on Gradio
* No external LLM API required
* Supports CPU and NVIDIA GPU environments
* Automatically detects hardware and recommends an execution profile
* Uses `faster-whisper` for speech-to-text
* Uses local Hugging Face text LLMs for summarization
* Converts Markdown to DOCX using Pandoc
* Stores packages, models, caches, and temporary files under `.agent_runtime/`
* Easy cleanup by deleting `.agent_runtime/`
* Supports private local use and internal LAN sharing

---

## 3. Important architecture note

A browser URL does **not** run Python on the visitor's GPU or CPU.

The computation runs on the computer or server where this app is launched.

### Private local use

Each user runs this app on their own PC and opens:

```text
http://127.0.0.1:7860
```

In this mode, audio files and generated documents stay on that user's computer.

### Internal LAN use

Run the app on one workstation or server and share a LAN URL:

```text
http://<server-ip>:7860
```

In this mode, uploaded audio files are processed and stored on the host machine.
Use authentication when sharing the app inside a company network.

### Public external sharing

Public sharing is disabled by default and is not recommended for confidential recordings.

---

## 4. Requirements

### Required

* Git
* Python 3.10, 3.11, or 3.12
* Internet connection for the first installation and model download
* Windows, macOS, or Linux

### Optional

* NVIDIA GPU with CUDA support
* Larger RAM/VRAM for better ASR and LLM profiles

### Recommended minimum

| Environment    | Recommendation          |
| -------------- | ----------------------- |
| CPU only       | 16 GB RAM recommended   |
| NVIDIA GPU     | 8 GB+ VRAM recommended  |
| Better quality | 12 GB+ VRAM recommended |

---

## 5. Clone the repository

Open a terminal, PowerShell, CMD, or Git Bash.

```bash
git clone https://github.com/RayKim3103/audio_2_docx.git
cd audio_2_docx
```

If the repository is private, sign in to GitHub first or use a Personal Access Token / GitHub Desktop.

---

## 6. Quick start on Windows

### Option A. Simple local execution

Double-click:

```text
run_windows.bat
```

The script will:

```text
1. Create .agent_runtime/
2. Create a Python virtual environment
3. Install required packages
4. Download models on first use
5. Launch the Gradio web UI
```

Then open:

```text
http://127.0.0.1:7860
```

Upload one or more audio files and click:

```text
회의록 DOCX 생성
```

---

### Option B. Run from PowerShell or CMD

```powershell
cd path\to\audio_2_docx
run_windows.bat
```

If you want to force CUDA PyTorch installation:

```powershell
python install.py --torch cuda
.agent_runtime\venv\Scripts\python.exe run_app.py
```

If CUDA installation fails or your machine has no NVIDIA GPU:

```powershell
python install.py --torch cpu
.agent_runtime\venv\Scripts\python.exe run_app.py
```

---

## 7. Quick start on macOS / Linux

```bash
cd audio_2_docx
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

Then open:

```text
http://127.0.0.1:7860
```

---

## 8. Internal LAN execution

Use this only on a trusted internal network.

### Windows

```text
run_lan_windows.bat
```

You will be asked to set an ID and password.

Other users on the same network can access:

```text
http://<host-ip>:7860
```

### macOS / Linux

You can run:

```bash
.agent_runtime/venv/bin/python run_app.py --host 0.0.0.0 --port 7860
```

Use authentication or a reverse proxy if needed.

---

## 9. How to use the web UI

1. Open the web UI.
2. Check the hardware analysis section.
3. Upload one or more audio/video files.
4. Choose an execution profile:

   * `auto` is recommended.
5. Select the input language.

   * For Korean audio, choose `ko`.
   * For automatic language detection, choose `auto`.
6. Optionally enter domain-specific terms or glossary hints.
7. Click **회의록 DOCX 생성**.
8. Download the result ZIP.

---

## 10. Output structure

Generated files are stored under:

```text
workspace/outputs/
```

Each run creates a separate output folder:

```text
workspace/outputs/<run_id>/
├─ transcripts/
│  ├─ original.txt
│  └─ timestamped.txt
├─ segments/
│  └─ segments.json
├─ notes_md/
│  └─ meeting_note.md
├─ docx/
│  └─ meeting_note.docx
├─ json/
│  └─ structured_summary.json
├─ logs/
│  └─ process_results.json
└─ meeting_docx_outputs.zip
```

The ZIP file contains the main deliverables.

---

## 11. Runtime isolation

All installed packages, downloaded models, pip cache, Hugging Face cache, Pandoc tools, and temporary files are placed under:

```text
.agent_runtime/
```

This includes:

```text
.agent_runtime/
├─ venv/
├─ models/
├─ hf_home/
├─ pip_cache/
├─ torch_home/
├─ tools/
├─ gradio_tmp/
└─ tmp/
```

To remove installed packages and downloaded models, delete:

```text
.agent_runtime/
```

### Windows

```powershell
Remove-Item -Recurse -Force .agent_runtime
```

### macOS / Linux

```bash
rm -rf .agent_runtime
```

Generated outputs are stored separately under:

```text
workspace/
```

To remove processed files:

### Windows

```powershell
Remove-Item -Recurse -Force workspace
```

### macOS / Linux

```bash
rm -rf workspace
```

---

## 12. Execution profiles

The app can automatically choose a profile based on hardware.

| Profile        | ASR model                 | LLM model     | Target environment     |
| -------------- | ------------------------- | ------------- | ---------------------- |
| `auto`         | Auto-selected             | Auto-selected | Recommended default    |
| `cpu_low`      | faster-whisper `base`     | Qwen2.5-0.5B  | Low-resource CPU       |
| `cpu_standard` | faster-whisper `small`    | Qwen2.5-1.5B  | Better CPU environment |
| `gpu_light`    | faster-whisper `small`    | Qwen2.5-1.5B  | Entry NVIDIA GPU       |
| `gpu_balanced` | faster-whisper `medium`   | Qwen2.5-3B    | Balanced GPU quality   |
| `gpu_quality`  | faster-whisper `large-v3` | Qwen2.5-7B    | High quality GPU mode  |

Models are downloaded on first use into:

```text
.agent_runtime/models/
.agent_runtime/hf_home/
```

---

## 13. Security and privacy

This project is designed for local or internal execution.

* No OpenAI API
* No Gemini API
* No Claude API
* No external LLM API call
* Audio is processed on the host machine
* Generated transcripts and DOCX files are stored locally

However, if you run the app on a shared server, uploaded files are stored and processed on that server.
Do not use public sharing for confidential recordings.

---

## 14. GitHub upload policy

Do not commit runtime files, downloaded models, audio files, or generated documents.

The repository should include source code only.

Recommended `.gitignore` entries:

```gitignore
.agent_runtime/
workspace/
audio/
outputs/
outputs_*/
__pycache__/
*.pyc

*.mp3
*.m4a
*.wav
*.flac
*.mp4
*.webm

*.docx
*.zip
*.safetensors
*.gguf
*.bin
*.pt
*.pth

.env
*.key
*.pem
```

---

## 15. Troubleshooting

### Git is not recognized on Windows

Install Git for Windows, then restart CMD or PowerShell.

Check:

```powershell
git --version
```

---

### Author identity unknown during git commit

Set Git user information:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

Then commit again:

```powershell
git commit -m "Initial commit"
```

---

### GPU exists but the app uses CPU

Check CUDA availability:

```powershell
.agent_runtime\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable, reinstall runtime with CUDA PyTorch:

```powershell
Remove-Item -Recurse -Force .agent_runtime
python install.py --torch cuda
.agent_runtime\venv\Scripts\python.exe run_app.py
```

If your machine has no NVIDIA GPU, use CPU mode:

```powershell
python install.py --torch cpu
.agent_runtime\venv\Scripts\python.exe run_app.py
```

---

### Model download is slow

The first execution downloads ASR and LLM models.
Start with a smaller profile first:

```text
cpu_low
```

or

```text
gpu_light
```

After confirming that the pipeline works, use larger profiles.

---

### DOCX does not open in Word

This app uses Pandoc for DOCX generation and validates the DOCX zip/XML structure.
If a DOCX file still fails, check:

```text
workspace/outputs/<run_id>/notes_md/
workspace/outputs/<run_id>/logs/process_results.json
```

---

### Gradio UI does not open

Try manually opening:

```text
http://127.0.0.1:7860
```

If another process is already using the port, run:

```powershell
.agent_runtime\venv\Scripts\python.exe run_app.py --port 7861
```

Then open:

```text
http://127.0.0.1:7861
```

---

## 16. Kaggle / Colab experiments

This project can also be tested on GPU notebook environments such as Kaggle or Google Colab.

In such cases, the computation runs on the notebook VM, not on the visitor's computer.

For Kaggle or Colab experiments:

```bash
pip install -r requirements/base.txt
python run_app.py --host 0.0.0.0 --port 7860 --share
```

The `share` URL is useful for testing but is not recommended for confidential recordings.

---

## 17. Development notes

Main source code is under:

```text
src/meeting_docx_agent/
```

Important modules:

```text
hardware.py       # hardware detection
profiles.py       # ASR/LLM profile selection
asr.py            # faster-whisper transcription
llm.py            # local Hugging Face LLM loading and generation
summarizer.py     # chunk summarization and JSON repair
markdown_builder.py
docx_exporter.py
pipeline.py
ui.py             # Gradio UI
```

---

## 18. Recommended workflow for developers

```bash
git clone https://github.com/RayKim3103/audio_2_docx.git
cd audio_2_docx

# Windows
run_windows.bat

# macOS/Linux
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

After modifying code:

```bash
git status
git add .
git commit -m "Describe your change"
git push
```

Before pushing, confirm that runtime files are not included:

```bash
git status --short
```

Do not push:

```text
.agent_runtime/
workspace/
audio/
*.mp3
*.wav
*.docx
*.zip
*.safetensors
*.gguf
```

---

## 19. License

Add your preferred license file, for example:

```text
MIT
Apache-2.0
Private internal use only
```

If this repository is for internal company use, keep it private and define an internal usage policy.

---

## 20. Summary

This project provides a local AI agent for converting meeting audio into structured DOCX documents.

It is designed for:

```text
private local use
internal company use
GPU server experiments
Kaggle / Colab testing
```

It is not designed as a public hosted SaaS service by default.
