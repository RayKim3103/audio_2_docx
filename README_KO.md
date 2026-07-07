# 로컬 회의 오디오 → DOCX AI Agent



## v14 품질 개선 사항

- GPU 프로필에서 최종 DOCX 생성 경로를 transcript-first Markdown writer 중심으로 변경했습니다.
- 짧거나 중간 길이의 transcript는 JSON 병합 결과에 의존하지 않고 최종 writer가 원문 흐름을 직접 보고 문서를 작성합니다.
- 최종 Markdown 초안에 JSON 누출, 중국어/일본어/한자 혼입, 조각난 발화체가 섞이면 1회 repair를 수행한 뒤, 그래도 부족하면 안전한 원문 기반 Markdown으로 대체합니다.
- 이 변경은 특정 샘플이 아니라 general domain 회의·강의·인터뷰·발표·설명 영상 정리 품질을 높이는 것을 목표로 합니다.

## v14 GPU 품질 개선 사항

`general_meeting_v14`에서는 GPU profile, 특히 `gpu_balanced`의 DOCX 품질을 개선했습니다. transcript chunk를 조금 더 작게 나누고, 최종 병합 단계에서 chunk note뿐 아니라 시간순 transcript digest를 함께 사용합니다. 또한 한국어 문체/중국어 혼입 보정, 근거 표기 개선, general domain용 prompt를 강화했습니다.

GPU 권장 설정:

```text
실행 프로필: gpu_balanced
처리 전략: full 또는 자동 추천
문서 상세도: detailed
최종 병합에도 LLM 사용: 체크
```

ASR 오인식이 많은 녹음은 웹 UI의 고유명사/전문용어·ASR 보정 힌트에 회사명, 참석자명, 약어, 프로젝트명 등을 넣으면 품질이 좋아질 수 있습니다.


회의, 강의, 인터뷰, 녹음 파일 등 일반 도메인의 오디오/비디오 파일을 전사하고 핵심 내용을 정리하여 Word DOCX 문서로 생성하는 로컬 AI Agent입니다.

이 프로젝트는 OpenAI, Gemini, Claude API 같은 외부 LLM API를 사용하지 않습니다. 음성 전사와 요약은 앱이 실행되는 로컬 PC 또는 사내 서버에서 수행됩니다.

---

## 이 시스템이 하는 일

```text
Audio / Video file
→ 실행 컴퓨터 사양 분석
→ ASR + LLM 프로필 선택
→ faster-whisper로 음성 전사
→ 로컬 Hugging Face LLM으로 정보 추출 및 회의록 정리
→ Markdown 생성
→ Pandoc으로 DOCX 변환
→ ZIP 파일 다운로드
```

최종 ZIP에는 DOCX, Markdown, 일반 transcript, timestamped transcript, segment JSON, summary JSON, run configuration, 처리 로그가 포함됩니다.

---

## 이번 버전의 핵심 개선 사항

이 버전은 CPU 환경에서 시간이 너무 오래 걸리고 DOCX 내용이 반복적·빈약하게 나오는 문제를 줄이기 위해 조정되었습니다.

- 웹 UI에 **처리 전략** 옵션을 추가했습니다: `자동 추천`, `빠른 균형`, `전체 LLM`, `LLM 없이 빠른 추출`.
- CPU 프로필의 `auto`는 이제 chunk마다 LLM을 호출하지 않고, transcript 추출 결과를 만든 뒤 최종 LLM 1회로 문서를 다듬습니다.
- GPU 프로필의 `auto`는 기존처럼 풍부한 chunk별 LLM 정리 방식을 유지합니다.
- LLM이 같은 문구를 반복하는 현상을 줄이기 위해 repetition guard와 후처리 deduplication을 추가했습니다.
- 최종 LLM 결과가 반복적이거나 빈약하면, 추출 기반 chunk note로 자동 보강합니다.
- ASR 전사 오류 가능성을 고려하는 prompt는 유지했습니다.
- 각 실행마다 실제 사용 모델, device, chunk 수, 처리 전략, LLM 호출 횟수, fallback 여부를 `run_config.json`에 기록합니다.

---

## 중요한 구조 설명

브라우저 URL에 접속한다고 해서 접속자의 컴퓨터 GPU/CPU에서 Python 코드가 실행되는 것은 아닙니다. 실제 연산은 항상 이 앱을 실행한 컴퓨터 또는 서버에서 수행됩니다.

- 개인 로컬 사용: 각 사용자가 자기 PC에서 실행하고 `http://127.0.0.1:7860`으로 접속합니다.
- 사내 LAN 사용: 한 대의 워크스테이션/서버에서 실행하고 내부망 URL을 공유합니다. 업로드 파일은 해당 host에서 처리되므로 인증을 사용하세요.
- 외부 공개 공유는 기밀 녹음에는 권장하지 않습니다.

---

## 빠른 시작

```bash
git clone https://github.com/RayKim3103/audio_2_docx.git
cd audio_2_docx
```

### Windows

아래 파일을 더블클릭합니다.

```text
run_windows.bat
```

또는 PowerShell/CMD에서 실행합니다.

```powershell
run_windows.bat
```

### macOS / Linux

```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:7860
```

---

## 웹 UI 사용 방법

1. 오디오/비디오 파일을 업로드합니다.
2. 실행 프로필을 선택합니다. 기본값은 `auto`를 권장합니다.
3. 입력 언어를 선택합니다. 한국어 녹음은 `ko`를 선택합니다.
4. 문서 출력 언어를 선택합니다.
5. 필요하면 고유명사/전문용어·ASR 보정 힌트를 입력합니다. ASR 전사 오류가 예상되는 용어를 함께 적으면 정리 품질이 좋아집니다.
6. **문서 상세도**를 선택합니다.
   - `간단 요약`: 빠르고 짧은 결과.
   - `표준 회의록`: 일반적인 회의록 품질.
   - `상세 회의록`: 더 풍부한 주제별 상세 정리. 결과가 빈약하면 이 옵션을 권장합니다.
7. **처리 전략**을 선택합니다.
   - `자동 추천`: 권장값. CPU는 빠른 균형, GPU는 전체 LLM 방식 사용
   - `빠른 균형`: transcript 추출 + 최종 LLM 1회. CPU 권장
   - `전체 LLM`: chunk별 LLM + 최종 LLM. GPU 품질 검증 권장
   - `LLM 없이 빠른 추출`: 가장 빠른 fallback
8. **회의록 DOCX 생성** 버튼을 클릭합니다.
9. 결과 ZIP 파일을 다운로드합니다.

---

## Runtime 격리 구조

설치된 패키지, 다운로드된 모델, pip cache, Hugging Face cache, Pandoc 도구, 임시 파일은 모두 아래 폴더에 저장됩니다.

```text
.agent_runtime/
```

설치 패키지와 모델 파일을 지우려면 `.agent_runtime`을 삭제합니다.

Windows:

```powershell
Remove-Item -Recurse -Force .agent_runtime
```

macOS/Linux:

```bash
rm -rf .agent_runtime
```

생성된 결과 파일은 다음 폴더에 저장됩니다.

```text
workspace/outputs/
```

---

## 실행 프로필

| Profile | ASR model | LLM model | 대상 환경 |
|---|---|---|---|
| `auto` | 자동 선택 | 자동 선택 | 기본 권장 |
| `cpu_low` | faster-whisper `base` | Qwen2.5-0.5B | 저사양 CPU |
| `cpu_standard` | faster-whisper `small` | Qwen2.5-1.5B | 일반 CPU 환경 |
| `gpu_light` | faster-whisper `small` | Qwen2.5-1.5B | 보급형 NVIDIA GPU |
| `gpu_balanced` | faster-whisper `medium` | Qwen2.5-3B | 품질/속도 균형 GPU |
| `gpu_quality` | faster-whisper `large-v3` | Qwen2.5-7B | 고품질 GPU 모드 |

모델은 최초 사용 시 `.agent_runtime/models`와 `.agent_runtime/hf_home` 아래로 다운로드됩니다.

---

## 출력 구조

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

`run_config.json`에는 실제 사용 ASR/LLM 모델, device, token 설정, chunk 수, transcript 길이, 처리 전략, LLM 호출 횟수, fallback 여부, ASR 오류 인지 설정이 기록됩니다.

---

## 문제 해결

### 결과가 너무 빈약하거나 CPU에서 너무 오래 걸리는 경우

웹 UI에서 다음 옵션을 먼저 사용하세요.

```text
문서 상세도 = 상세 회의록
처리 전략 = 자동 추천 또는 빠른 균형
```

GPU에서 품질을 높이고 싶을 때만 `처리 전략 = 전체 LLM`을 사용하세요. 그리고 `workspace/outputs/<run_id>/json/<file>.run_config.json`에서 실제 LLM 모델, device, token 수, chunk 수, 처리 전략, LLM 호출 횟수, fallback 여부를 확인하세요.

### GPU가 있는데 CPU를 사용하는 경우

```powershell
Remove-Item -Recurse -Force .agent_runtime
python install.py --torch cuda
.agent_runtime\venv\Scripts\python.exe run_app.py
```

### DOCX가 Word에서 열리지 않는 경우

이 앱은 Pandoc 기반으로 DOCX를 생성하고 DOCX zip/XML 구조를 검증합니다. 문제가 계속되면 `workspace/outputs/<run_id>/notes_md/*.md`와 `logs/process_results.json`을 확인하세요.

### 모델 다운로드가 느린 경우

처음에는 `cpu_low` 또는 `gpu_light`로 파이프라인을 확인한 뒤 큰 모델을 사용하세요.

### v6 CPU 최적화 변경 사항

- `PIPELINE_VERSION`은 `general_meeting_v14`입니다.
- CPU `auto` 모드는 더 이상 모든 chunk마다 LLM을 호출하지 않습니다. 추출 기반 chunk note를 만든 뒤 최종 LLM 1회로 문서를 다듬습니다.
- 이 방식은 CPU 실행 시간을 줄이면서도 주제별 상세 내용을 보존하도록 설계되었습니다.
- 반복 문구를 줄이기 위한 generation 설정과 후처리 로직을 추가했습니다.
- LLM 결과가 반복적이거나 빈약하면 추출 기반 내용으로 자동 보강합니다.
- v4의 ASR 오류 인지 prompt는 유지됩니다.
- `run_config.json`에 `processing_strategy_effective`와 `llm_calls`가 기록됩니다.


## v14 Quality Update

Version v14 changes the GPU full-quality path to a direct transcript-first Markdown writer. Earlier versions could still produce sparse documents because intermediate JSON or chunk-based notes dominated the final DOCX. v14 asks the GPU LLM to write the final human-facing report directly from the timestamped transcript, then applies a quality check for missing sections, JSON leakage, mixed CJK characters, excessive `명시적으로 확인되지 않음`, and low-value keyword lists.

For GPU experiments, recommended settings are:

```text
Profile: gpu_quality or gpu_balanced
Processing strategy: full
Detail level: detailed
Output language: ko
Use final LLM: enabled
```

For domain-specific recordings, provide glossary hints such as names, project names, product names, acronyms, and known ASR correction pairs.
