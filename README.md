# 로컬 회의 오디오 → DOCX AI Agent

회의, 강의, 인터뷰, 녹음 파일 등 일반 도메인의 오디오/비디오 파일을 전사하고, 핵심 내용을 정리하여 Word DOCX 문서로 생성하는 로컬 AI Agent입니다.

이 프로젝트는 OpenAI, Gemini, Claude API 같은 외부 LLM API를 사용하지 않습니다.
음성 전사와 요약은 앱이 실행되는 로컬 PC 또는 사내 서버에서 수행됩니다.

---

## 1. 이 시스템이 하는 일

이 앱은 Gradio 기반 로컬 웹 UI를 통해 오디오/비디오 녹음 파일을 구조화된 Word 문서로 변환합니다.

전체 처리 흐름은 다음과 같습니다.

```text
Audio / Video file
→ 실행 컴퓨터 사양 분석
→ ASR + LLM 프로필 자동 선택
→ faster-whisper로 음성 전사
→ 로컬 Hugging Face LLM으로 회의 내용 정리
→ Markdown 생성
→ Pandoc으로 DOCX 변환
→ ZIP 파일 다운로드
```

최종 ZIP 파일에는 다음 결과물이 포함됩니다.

```text
회의록 DOCX
Markdown 정리본
일반 transcript
timestamp 포함 transcript
구조화 JSON 로그
처리 로그
```

---

## 2. 주요 기능

* Gradio 기반 로컬 웹 UI 제공
* OpenAI / Gemini / Claude 같은 외부 LLM API 미사용
* CPU 환경과 NVIDIA GPU 환경 모두 지원
* 실행 컴퓨터의 사양을 분석하여 실행 프로필 추천
* `faster-whisper` 기반 음성 전사
* 로컬 Hugging Face 텍스트 LLM 기반 회의록 정리
* Markdown 생성 후 Pandoc 기반 DOCX 변환
* 패키지, 모델, cache, 임시 파일을 `.agent_runtime/` 아래에 격리 저장
* `.agent_runtime/` 폴더 삭제만으로 설치 패키지와 모델 파일 정리 가능
* 개인 로컬 실행 및 사내 LAN 공유 실행 지원

---

## 3. 중요한 아키텍처 설명

브라우저 URL에 접속한다고 해서 접속자의 컴퓨터 GPU/CPU에서 Python 코드가 실행되는 것은 아닙니다.

실제 연산은 항상 **이 앱을 실행한 컴퓨터 또는 서버**에서 수행됩니다.

### 개인 로컬 사용

각 사용자가 자기 PC에서 앱을 실행하고 다음 주소로 접속합니다.

```text
http://127.0.0.1:7860
```

이 방식에서는 오디오 파일, transcript, DOCX 결과물이 모두 사용자 PC 안에 저장됩니다.

### 사내 LAN 공유 사용

한 대의 워크스테이션 또는 서버에서 앱을 실행하고 LAN URL을 공유합니다.

```text
http://<server-ip>:7860
```

이 방식에서는 사용자가 브라우저로 파일을 업로드하지만, 실제 처리는 서버에서 수행됩니다.
업로드된 오디오 파일과 결과물도 해당 서버에 저장됩니다.

사내망에서 공유할 경우 반드시 인증을 설정하는 것을 권장합니다.

### 외부 공개 공유

외부 공개 공유는 기본적으로 비활성화되어 있으며, 기밀 회의 녹음에는 권장하지 않습니다.

---

## 4. 요구사항

### 필수

* Git
* Python 3.10, 3.11, 또는 3.12
* 최초 설치 및 모델 다운로드를 위한 인터넷 연결
* Windows, macOS, Linux 중 하나

### 선택

* CUDA를 지원하는 NVIDIA GPU
* 더 큰 ASR/LLM 모델을 사용하기 위한 충분한 RAM/VRAM

### 권장 최소 사양

| 환경         | 권장 사항           |
| ---------- | --------------- |
| CPU only   | RAM 16GB 이상 권장  |
| NVIDIA GPU | VRAM 8GB 이상 권장  |
| 고품질 처리     | VRAM 12GB 이상 권장 |

---

## 5. GitHub에서 코드 가져오기

터미널, PowerShell, CMD, Git Bash 중 하나를 엽니다.

```bash
git clone https://github.com/RayKim3103/audio_2_docx.git
cd audio_2_docx
```

저장소가 private인 경우 GitHub 로그인이 필요합니다.
GitHub Desktop 또는 Personal Access Token을 사용할 수 있습니다.

---

## 6. Windows 빠른 실행

### 방법 A. 가장 쉬운 실행

프로젝트 폴더에서 아래 파일을 더블클릭합니다.

```text
run_windows.bat
```

스크립트는 다음 작업을 자동으로 수행합니다.

```text
1. .agent_runtime/ 폴더 생성
2. Python 가상환경 생성
3. 필요한 패키지 설치
4. 최초 사용 시 모델 다운로드
5. Gradio 웹 UI 실행
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:7860
```

오디오 파일을 업로드한 뒤 **회의록 DOCX 생성** 버튼을 누르면 됩니다.

---

### 방법 B. PowerShell 또는 CMD에서 실행

```powershell
cd path\to\audio_2_docx
run_windows.bat
```

CUDA PyTorch 설치를 강제로 시도하려면 다음을 실행합니다.

```powershell
python install.py --torch cuda
.agent_runtime\venv\Scripts\python.exe run_app.py
```

CUDA 설치에 실패하거나 NVIDIA GPU가 없는 경우 CPU 모드로 설치합니다.

```powershell
python install.py --torch cpu
.agent_runtime\venv\Scripts\python.exe run_app.py
```

---

## 7. macOS / Linux 빠른 실행

```bash
cd audio_2_docx
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

그 후 브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:7860
```

---

## 8. 사내 LAN 실행

신뢰할 수 있는 내부 네트워크에서만 사용하는 것을 권장합니다.

### Windows

```text
run_lan_windows.bat
```

실행 시 ID와 비밀번호를 설정합니다.

같은 네트워크의 사용자는 다음 주소로 접속할 수 있습니다.

```text
http://<host-ip>:7860
```

### macOS / Linux

```bash
.agent_runtime/venv/bin/python run_app.py --host 0.0.0.0 --port 7860
```

필요하면 인증 또는 reverse proxy를 함께 설정하세요.

---

## 9. 웹 UI 사용 방법

1. 웹 UI를 엽니다.
2. 하드웨어 분석 결과를 확인합니다.
3. 오디오 또는 비디오 파일을 하나 이상 업로드합니다.
4. 실행 프로필을 선택합니다.

   * 기본값은 `auto`를 권장합니다.
5. 입력 언어를 선택합니다.

   * 한국어 녹음은 `ko`를 선택합니다.
   * 자동 감지는 `auto`를 선택합니다.
6. 필요하면 고유명사, 프로젝트명, 전문용어 힌트를 입력합니다.
7. **회의록 DOCX 생성** 버튼을 클릭합니다.
8. 결과 ZIP 파일을 다운로드합니다.

---

## 10. 출력 파일 구조

생성된 파일은 다음 폴더 아래에 저장됩니다.

```text
workspace/outputs/
```

각 실행마다 별도의 output 폴더가 생성됩니다.

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

주요 결과물은 ZIP 파일 안에 함께 포함됩니다.

---

## 11. Runtime 격리 구조

설치된 패키지, 다운로드된 모델, pip cache, Hugging Face cache, Pandoc 도구, 임시 파일은 모두 아래 폴더에 저장됩니다.

```text
.agent_runtime/
```

구조는 다음과 같습니다.

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

설치된 패키지와 다운로드된 모델을 삭제하려면 `.agent_runtime/` 폴더를 삭제하면 됩니다.

### Windows

```powershell
Remove-Item -Recurse -Force .agent_runtime
```

### macOS / Linux

```bash
rm -rf .agent_runtime
```

생성된 결과 파일은 별도로 `workspace/` 아래에 저장됩니다.

처리 결과까지 삭제하려면 다음을 실행합니다.

### Windows

```powershell
Remove-Item -Recurse -Force workspace
```

### macOS / Linux

```bash
rm -rf workspace
```

---

## 12. 실행 프로필

앱은 하드웨어 사양을 기반으로 실행 프로필을 자동 선택할 수 있습니다.

| Profile        | ASR model                 | LLM model    | 대상 환경          |
| -------------- | ------------------------- | ------------ | -------------- |
| `auto`         | 자동 선택                     | 자동 선택        | 기본 권장          |
| `cpu_low`      | faster-whisper `base`     | Qwen2.5-0.5B | 저사양 CPU        |
| `cpu_standard` | faster-whisper `small`    | Qwen2.5-1.5B | 일반 CPU 환경      |
| `gpu_light`    | faster-whisper `small`    | Qwen2.5-1.5B | 보급형 NVIDIA GPU |
| `gpu_balanced` | faster-whisper `medium`   | Qwen2.5-3B   | 품질/속도 균형 GPU   |
| `gpu_quality`  | faster-whisper `large-v3` | Qwen2.5-7B   | 고품질 GPU 모드     |

모델은 최초 사용 시 다음 경로에 다운로드됩니다.

```text
.agent_runtime/models/
.agent_runtime/hf_home/
```

---

## 13. 보안 및 개인정보

이 프로젝트는 로컬 또는 사내 내부 실행을 목표로 설계되었습니다.

* OpenAI API 미사용
* Gemini API 미사용
* Claude API 미사용
* 외부 LLM API 호출 없음
* 오디오는 앱이 실행된 host에서 처리됨
* 생성된 transcript와 DOCX는 로컬에 저장됨

다만 사내 서버에서 공유 실행하는 경우, 사용자가 업로드한 파일은 해당 서버에 저장되고 처리됩니다.
기밀 녹음 파일은 public sharing으로 처리하지 않는 것을 권장합니다.

---

## 14. GitHub 업로드 정책

GitHub에는 소스코드만 올리는 것을 권장합니다.

다음 파일과 폴더는 commit하지 마세요.

```text
.agent_runtime/
workspace/
audio/
outputs/
모델 파일
cache 파일
회의 녹음 파일
생성된 DOCX/ZIP 결과물
```

권장 `.gitignore` 예시는 다음과 같습니다.

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

## 15. 문제 해결

### Windows에서 `git` 명령을 인식하지 못하는 경우

Git for Windows를 설치한 뒤 CMD 또는 PowerShell을 다시 엽니다.

확인:

```powershell
git --version
```

---

### `git commit` 시 Author identity unknown이 나오는 경우

Git 사용자 정보를 설정합니다.

```powershell
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

그 후 다시 commit합니다.

```powershell
git commit -m "Initial commit"
```

---

### GPU가 있는데 앱이 CPU를 사용하는 경우

CUDA 사용 가능 여부를 확인합니다.

```powershell
.agent_runtime\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

CUDA를 사용할 수 없다면 runtime을 삭제한 뒤 CUDA PyTorch로 재설치합니다.

```powershell
Remove-Item -Recurse -Force .agent_runtime
python install.py --torch cuda
.agent_runtime\venv\Scripts\python.exe run_app.py
```

NVIDIA GPU가 없는 환경에서는 CPU 모드를 사용합니다.

```powershell
python install.py --torch cpu
.agent_runtime\venv\Scripts\python.exe run_app.py
```

---

### 모델 다운로드가 너무 느린 경우

최초 실행 시 ASR 모델과 LLM 모델을 다운로드합니다.
처음에는 작은 프로필로 파이프라인을 먼저 확인하는 것을 권장합니다.

```text
cpu_low
```

또는

```text
gpu_light
```

정상 동작을 확인한 뒤 더 큰 프로필을 사용하세요.

---

### DOCX가 Word에서 열리지 않는 경우

이 앱은 Pandoc 기반으로 DOCX를 생성하고, DOCX zip/XML 구조를 검증합니다.
문제가 계속되면 다음 파일을 확인하세요.

```text
workspace/outputs/<run_id>/notes_md/
workspace/outputs/<run_id>/logs/process_results.json
```

---

### Gradio UI가 열리지 않는 경우

브라우저에서 직접 다음 주소를 열어보세요.

```text
http://127.0.0.1:7860
```

포트가 이미 사용 중이면 다른 포트로 실행합니다.

```powershell
.agent_runtime\venv\Scripts\python.exe run_app.py --port 7861
```

그 후 다음 주소를 엽니다.

```text
http://127.0.0.1:7861
```

---

## 16. Kaggle / Colab 실험

이 프로젝트는 Kaggle 또는 Google Colab 같은 GPU Notebook 환경에서도 실험할 수 있습니다.

이 경우 실제 연산은 접속자의 컴퓨터가 아니라 Notebook VM에서 수행됩니다.

Kaggle 또는 Colab에서 테스트할 때는 다음과 같이 실행할 수 있습니다.

```bash
pip install -r requirements/base.txt
python run_app.py --host 0.0.0.0 --port 7860 --share
```

`share` URL은 테스트에는 편리하지만, 기밀 녹음 파일에는 권장하지 않습니다.

---

## 17. 개발자 참고

주요 소스코드는 다음 폴더에 있습니다.

```text
src/meeting_docx_agent/
```

주요 모듈은 다음과 같습니다.

```text
hardware.py          # 하드웨어 분석
profiles.py          # ASR/LLM 프로필 선택
asr.py               # faster-whisper 전사
llm.py               # 로컬 Hugging Face LLM 로드 및 생성
summarizer.py        # chunk 요약 및 JSON repair
markdown_builder.py  # Markdown 생성
docx_exporter.py     # DOCX 변환
pipeline.py          # 전체 파이프라인
ui.py                # Gradio UI
```

---

## 18. 개발자 권장 작업 흐름

```bash
git clone https://github.com/RayKim3103/audio_2_docx.git
cd audio_2_docx

# Windows
run_windows.bat

# macOS/Linux
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

코드를 수정한 뒤에는 다음 순서로 commit/push합니다.

```bash
git status
git add .
git commit -m "Describe your change"
git push
```

push 전에 runtime 파일이 포함되지 않았는지 확인하세요.

```bash
git status --short
```

다음 파일들은 push하지 마세요.

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

원하는 라이선스 파일을 추가하세요.

예시:

```text
MIT
Apache-2.0
Private internal use only
```

사내 사용 목적이라면 저장소를 private으로 유지하고 내부 사용 정책을 명확히 정의하는 것을 권장합니다.

---

## 20. 요약

이 프로젝트는 회의 오디오를 구조화된 DOCX 문서로 변환하는 로컬 AI Agent입니다.

주요 사용 목적은 다음과 같습니다.

```text
개인 로컬 사용
사내 내부 사용
GPU 서버 실험
Kaggle / Colab 테스트
```

기본적으로 public SaaS 서비스로 운영하는 것을 목표로 하지 않습니다.
