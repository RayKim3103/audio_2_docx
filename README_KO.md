# Local Meeting Audio → DOCX Agent 사용 설명서

이 프로젝트는 외부 LLM API 없이, 사용자의 로컬 컴퓨터 또는 내부 서버에서 녹음 파일을 회의록 DOCX로 변환하는 로컬 웹 앱입니다.

## 핵심 기능

1. 현재 앱이 실행 중인 컴퓨터 사양 분석
2. CPU/GPU/RAM에 맞는 ASR·LLM 모델 자동 선택
3. 오디오/동영상 파일 다중 업로드
4. faster-whisper 기반 전사
5. Hugging Face 로컬 LLM 기반 회의록 정리
6. Markdown 생성
7. Pandoc 기반 DOCX 생성
8. DOCX, Markdown, TXT, timestamped TXT, JSON 로그를 ZIP으로 다운로드

## 가장 중요한 구조적 주의점

브라우저 URL로 접속한다고 해서 방문자의 컴퓨터에서 Python 코드가 실행되는 것은 아닙니다. 연산은 항상 이 앱이 실행 중인 컴퓨터에서 수행됩니다.

따라서 보안이 중요한 로컬 환경에서는 다음 방식이 가장 안전합니다.

- 각 사용자가 자기 PC에서 앱 실행
- 브라우저에서 `http://127.0.0.1:7860` 접속
- 녹음 파일과 모델이 모두 자기 PC 안에만 존재

내부망에서 여러 사용자가 하나의 서버를 함께 쓰려면 `run_lan_windows.bat`을 사용하고 인증을 설정하세요. 이 경우 업로드 파일은 서버에 저장됩니다.

## Windows 실행 방법

1. ZIP을 압축 해제합니다.
2. `run_windows.bat`을 더블클릭합니다.
3. 처음 실행 시 `.agent_runtime` 폴더가 생성되고 패키지가 설치됩니다.
4. 브라우저가 열리면 오디오 파일을 업로드합니다.
5. `회의록 DOCX 생성` 버튼을 클릭합니다.
6. 결과 ZIP을 다운로드합니다.

## 내부망 공유 실행

```powershell
run_lan_windows.bat
```

실행 중 `user:password` 형식의 인증 정보를 입력하세요.

예:

```text
admin:myStrongPassword
```

내부망 사용자는 다음 주소로 접속합니다.

```text
http://서버IP:7860
```

## 설치물/모델/cache 삭제

패키지, 모델, Hugging Face cache, pip cache, Pandoc, 임시 파일은 모두 아래 폴더에 있습니다.

```text
.agent_runtime/
```

이 폴더를 지우면 앱이 설치한 패키지와 모델이 모두 삭제됩니다.

Windows PowerShell:

```powershell
Remove-Item -Recurse -Force .agent_runtime
```

결과물까지 삭제하려면:

```powershell
Remove-Item -Recurse -Force workspace
```

## 모델 프로필

| 프로필 | 용도 |
|---|---|
| auto | 앱이 사양 분석 후 자동 선택 |
| cpu_low | 저사양 노트북, 안정성 우선 |
| cpu_standard | RAM 16GB 이상 CPU 환경 |
| gpu_light | CUDA 가능 6~10GB VRAM |
| gpu_balanced | RTX 3060~3090 등 대부분 GPU |
| gpu_quality | 20GB 이상 VRAM, 품질 우선 |

## 보안 권장값

- 기본 실행은 `share=False`입니다.
- 외부 공개 URL은 권장하지 않습니다.
- 내부망 공유 시 반드시 `--auth`를 사용하세요.
- 처리 후 민감한 결과물은 `workspace`에서 삭제하세요.
