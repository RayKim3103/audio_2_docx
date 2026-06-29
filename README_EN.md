# Local Meeting Audio → DOCX Agent

A local AI agent that converts general-domain meeting, lecture, interview, or recording audio/video files into structured DOCX notes.

This project does **not** use external LLM APIs such as OpenAI, Gemini, or Claude API. Audio transcription and summarization are performed on the computer or server where this app is launched.

---

## What this system does

```text
Audio / Video file
→ Hardware detection
→ ASR + LLM profile selection
→ faster-whisper transcription
→ Local Hugging Face LLM information extraction and summarization
→ Markdown generation
→ Pandoc DOCX conversion
→ ZIP download
```

The final ZIP contains DOCX, Markdown, transcript, timestamped transcript, segment JSON, summary JSON, run configuration, and processing logs.

---

## Key improvements in this version

This version is tuned to avoid overly sparse DOCX outputs.

- Increased LLM output token budgets for chunk and final generation.
- Added transcript chunk overlap to preserve context across chunk boundaries.
- Changed the chunk prompt from simple summarization to information extraction.
- Added a **Document detail level** option: `brief`, `standard`, `detailed`.
- Added a dedicated **Topic-by-topic detailed notes** section to the DOCX.
- Preserves chunk-level details even when the final LLM merge is too short.
- Writes `run_config.json` so you can inspect the actual model, device, token limits, chunk count, and fallback status used for each run.
- Removed prompt placeholder phrases from the expected output path to reduce template-copy failures.

---

## Important architecture note

A browser URL does **not** run Python on the visitor's GPU/CPU. The computation runs on the computer/server where this app is launched.

- Private local use: each user runs this app on their own PC and opens `http://127.0.0.1:7860`.
- Internal LAN use: run on one workstation/server and share a LAN URL. Audio files are uploaded to that host, so use authentication.
- External public sharing is disabled by default and is not recommended for confidential recordings.

---

## Quick start

```bash
git clone https://github.com/RayKim3103/audio_2_docx.git
cd audio_2_docx
```

### Windows

Double-click:

```text
run_windows.bat
```

Or run from PowerShell/CMD:

```powershell
run_windows.bat
```

### macOS / Linux

```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

Then open:

```text
http://127.0.0.1:7860
```

---

## How to use the web UI

1. Upload one or more audio/video files.
2. Choose profile. `auto` is recommended.
3. Select input language. For Korean audio, use `ko`.
4. Select output language.
5. Optionally enter glossary/domain terms.
6. Choose **Document detail level**:
   - `brief`: short output, fast.
   - `standard`: balanced meeting notes.
   - `detailed`: richer notes with more topic-level detail. Recommended when DOCX output feels sparse.
7. Click **회의록 DOCX 생성**.
8. Download the result ZIP.

---

## Runtime isolation

All packages, downloaded models, pip cache, Hugging Face cache, Pandoc tools, and temporary files are placed under:

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

---

## Profiles

| Profile | ASR model | LLM model | Target environment |
|---|---|---|---|
| `auto` | Auto-selected | Auto-selected | Recommended default |
| `cpu_low` | faster-whisper `base` | Qwen2.5-0.5B | Low-resource CPU |
| `cpu_standard` | faster-whisper `small` | Qwen2.5-1.5B | Better CPU environment |
| `gpu_light` | faster-whisper `small` | Qwen2.5-1.5B | Entry NVIDIA GPU |
| `gpu_balanced` | faster-whisper `medium` | Qwen2.5-3B | Balanced GPU quality |
| `gpu_quality` | faster-whisper `large-v3` | Qwen2.5-7B | High quality GPU mode |

Models are downloaded on first use into `.agent_runtime/models` and `.agent_runtime/hf_home`.

---

## Output structure

```text
workspace/outputs/<run_id>/
├─ transcripts/
├─ segments/
├─ notes_md/
├─ docx/
├─ json/
│  ├─ <file>.summary.json
│  └─ <file>.run_config.json
├─ logs/
└─ audio_2_docx_outputs.zip
```

`run_config.json` records actual ASR/LLM model, device, token settings, chunk count, transcript length, and fallback status.

---

## Troubleshooting

### Output is too sparse

Use:

```text
Document detail level = detailed
```

Also check `workspace/outputs/<run_id>/json/<file>.run_config.json` to confirm the actual LLM model, device, token budget, chunk count, and whether fallback was used.

### GPU exists but app uses CPU

```powershell
Remove-Item -Recurse -Force .agent_runtime
python install.py --torch cuda
.agent_runtime\venv\Scripts\python.exe run_app.py
```

### DOCX does not open in Word

This app uses Pandoc for DOCX generation and validates the DOCX zip/XML structure. If a file still fails, check `workspace/outputs/<run_id>/notes_md/*.md` and `logs/process_results.json`.

### Model download is slow

Choose `cpu_low` or `gpu_light` first to test the pipeline. Larger profiles download larger LLMs.

### v3 output-format changes

- `PIPELINE_VERSION` is now the short form `general_meeting_v3`.
- `run_config.json` is still saved in the output ZIP, but execution settings are no longer inserted into the human-facing DOCX.
- Section `1. 한 페이지 요약` is rendered as readable prose paragraphs instead of bullet lists.
- Prompting and post-processing were improved to reduce raw JSON/Python-literal leakage such as `{'heading': ..., 'bullets': ...}` in the DOCX.
