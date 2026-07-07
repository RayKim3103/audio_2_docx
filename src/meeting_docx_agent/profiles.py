from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    label: str
    asr_model: str
    asr_device: str
    asr_compute_type: str
    asr_beam_size: int
    llm_model: str
    llm_device: str
    max_chars_per_chunk: int
    chunk_overlap_chars: int
    max_new_tokens_chunk: int
    max_new_tokens_final: int
    description: str


# v15 defaults: CPU auto keeps the practical direct-writer fast mode.
# GPU profiles use a grounded content pack + document-architect Markdown writer.
# The goal is to synthesize a human-readable document instead of copying transcript chunks.
PROFILES: Dict[str, RuntimeProfile] = {
    "cpu_low": RuntimeProfile(
        name="cpu_low",
        label="CPU 저사양 / 안정 우선",
        asr_model="base",
        asr_device="cpu",
        asr_compute_type="int8",
        asr_beam_size=1,
        llm_model="Qwen/Qwen2.5-0.5B-Instruct",
        llm_device="cpu",
        max_chars_per_chunk=3400,
        chunk_overlap_chars=250,
        max_new_tokens_chunk=800,
        max_new_tokens_final=2600,
        description="RAM 8~12GB급 노트북. auto는 direct writer fast 모드로 동작하여 LLM 호출을 최소화합니다. 상세도는 standard 이하 권장.",
    ),
    "cpu_standard": RuntimeProfile(
        name="cpu_standard",
        label="CPU 표준 / 품질 균형",
        asr_model="small",
        asr_device="cpu",
        asr_compute_type="int8",
        asr_beam_size=1,
        llm_model="Qwen/Qwen2.5-1.5B-Instruct",
        llm_device="cpu",
        max_chars_per_chunk=4600,
        chunk_overlap_chars=400,
        max_new_tokens_chunk=1200,
        max_new_tokens_final=4200,
        description="RAM 16~32GB급 CPU 환경. 기본 auto는 시간순 digest를 사람용 Markdown으로 직접 작성하여 속도와 가독성을 균형화합니다.",
    ),
    "gpu_light": RuntimeProfile(
        name="gpu_light",
        label="GPU 경량 / 6~10GB VRAM",
        asr_model="small",
        asr_device="cuda",
        asr_compute_type="float16",
        asr_beam_size=3,
        llm_model="Qwen/Qwen2.5-1.5B-Instruct",
        llm_device="cuda",
        max_chars_per_chunk=3200,
        chunk_overlap_chars=320,
        max_new_tokens_chunk=1500,
        max_new_tokens_final=3600,
        description="CUDA가 가능한 보급형 GPU. v15에서는 content pack 기반 document-architect writer로 가독성을 개선했습니다.",
    ),
    "gpu_balanced": RuntimeProfile(
        name="gpu_balanced",
        label="GPU 균형 / RTX 3060~3090 권장",
        asr_model="medium",
        asr_device="cuda",
        asr_compute_type="float16",
        asr_beam_size=3,
        llm_model="Qwen/Qwen2.5-3B-Instruct",
        llm_device="cuda",
        # v9: balanced profile uses smaller extraction chunks and a transcript-first
        # final writer.  Final document quality comes from sectioned Markdown
        # writing rather than a single huge JSON merge.
        max_chars_per_chunk=3600,
        chunk_overlap_chars=360,
        max_new_tokens_chunk=1900,
        max_new_tokens_final=4600,
        description="대부분의 CUDA GPU 서버/데스크톱에 적합한 기본 품질 프로필. v15에서는 content pack 기반 document-architect writer를 우선 사용해 사람이 쓴 문서처럼 정리합니다.",
    ),
    "gpu_quality": RuntimeProfile(
        name="gpu_quality",
        label="GPU 고품질 / 20GB+ VRAM",
        asr_model="large-v3",
        asr_device="cuda",
        asr_compute_type="float16",
        asr_beam_size=5,
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        llm_device="cuda",
        max_chars_per_chunk=5000,
        chunk_overlap_chars=520,
        max_new_tokens_chunk=3000,
        max_new_tokens_final=7200,
        description="RTX 3090/4090 또는 24GB급 이상에서 품질 우선. v15에서는 content pack 기반 document-architect writer를 우선 사용하여 7B 모델이 구조화된 문서를 쓰도록 합니다.",
    ),
}


def get_profile(name: str) -> RuntimeProfile:
    if name == "auto":
        from .hardware import detect_hardware
        name = detect_hardware().recommended_profile
    if name not in PROFILES:
        raise KeyError(f"Unknown profile: {name}. Available: {list(PROFILES)}")
    return PROFILES[name]


def profile_choices() -> list[str]:
    return ["auto"] + list(PROFILES.keys())
