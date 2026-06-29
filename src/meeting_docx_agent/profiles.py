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
    max_new_tokens_chunk: int
    max_new_tokens_final: int
    description: str


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
        max_chars_per_chunk=2800,
        max_new_tokens_chunk=900,
        max_new_tokens_final=1300,
        description="RAM 8~12GB급 노트북. 속도와 안정성 우선.",
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
        max_chars_per_chunk=3600,
        max_new_tokens_chunk=1200,
        max_new_tokens_final=1800,
        description="RAM 16~32GB급 CPU 환경. CPU에서는 느릴 수 있음.",
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
        max_chars_per_chunk=4200,
        max_new_tokens_chunk=1300,
        max_new_tokens_final=2000,
        description="CUDA가 가능한 보급형 GPU.",
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
        max_chars_per_chunk=5200,
        max_new_tokens_chunk=1500,
        max_new_tokens_final=2400,
        description="대부분의 CUDA GPU 서버/데스크톱에 적합한 기본 품질 프로필.",
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
        max_chars_per_chunk=6500,
        max_new_tokens_chunk=1800,
        max_new_tokens_final=3000,
        description="RTX 3090/4090 또는 24GB급 이상에서 품질 우선. 다운로드/메모리 사용량 큼.",
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
