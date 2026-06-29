# Local Meeting Audio → DOCX Agent


## v8 GPU quality update

Version `general_meeting_v8` improves GPU profiles, especially `gpu_balanced`, by using smaller transcript chunks, a transcript-aware final synthesis prompt, optional Korean language/style repair, and less noisy evidence rendering. The goal is to improve DOCX quality while keeping the full LLM pipeline for GPU users.

Recommended GPU settings:

```text
Profile: gpu_balanced
Processing strategy: full
Document detail: detailed
Use final LLM merge: enabled
```

If the final document still contains ASR mistakes, add domain terms and correction hints in the glossary box.


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

This version is optimized for CPU practicality and richer human-readable DOCX outputs.

- Added a **Processing strategy** option: `auto`, `fast`, `full`, `extractive`.
- In CPU profiles, `auto` now uses a faster hybrid strategy: deterministic transcript extraction + one final LLM polish call. This avoids slow per-chunk LLM generation on CPU.
- GPU profiles keep the richer full LLM strategy by default.
- Added repetition guards to reduce outputs such as one phrase repeated many times.
- Added a quality guard: if the final LLM output is sparse or repetitive, the document is rebuilt from chunk/extractive notes instead of saving a poor result.
- Kept ASR-error-aware prompting for Whisper/faster-whisper transcription mistakes.
- Writes `run_config.json` so you can inspect the actual model, device, chunk count, processing strategy, LLM call count, and fallback status used for each run.

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
5. Optionally enter glossary/domain terms and ASR correction hints. This helps when names, acronyms, numbers, or domain terms are likely to be mistranscribed.
6. Choose **Document detail level**:
   - `brief`: short output, fast.
   - `standard`: balanced meeting notes.
   - `detailed`: richer notes with more topic-level detail. Recommended when DOCX output feels sparse.
7. Choose **Processing strategy**:
   - `auto`: recommended. CPU uses fast hybrid; GPU uses full LLM.
   - `fast`: deterministic extraction + one final LLM polish call. Recommended for CPU.
   - `full`: chunk-level LLM extraction + final LLM merge. Recommended for GPU quality tests.
   - `extractive`: no LLM summarization; fastest fallback.
8. Click **회의록 DOCX 생성**.
9. Download the result ZIP.

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

`run_config.json` records actual ASR/LLM model, device, token settings, chunk count, transcript length, processing strategy, LLM call count, fallback status, and ASR-error-aware settings.

---

## Troubleshooting

### Output is too sparse

Use:

```text
Document detail level = detailed
Processing strategy = auto or fast on CPU
Processing strategy = full on GPU
```

If CPU processing is too slow, keep `auto` or `fast` instead of `full`. Also check `workspace/outputs/<run_id>/json/<file>.run_config.json` to confirm the actual LLM model, device, token budget, chunk count, and whether fallback was used.

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

### v6 CPU optimization changes

- `PIPELINE_VERSION` is now `general_meeting_v8`.
- CPU `auto` mode no longer performs LLM generation for every chunk; it uses extractive chunk notes plus one LLM polishing call.
- This is designed to reduce CPU runtime from very long runs to a more practical range while preserving structured details.
- The LLM generation settings include stronger repetition control.
- If the LLM output is repetitive or sparse, the pipeline falls back to extractive/chunk-based details rather than saving a poor DOCX.
- ASR-error-aware prompting from v4 is retained.
- `run_config.json` includes `processing_strategy_effective` and `llm_calls` for troubleshooting.
