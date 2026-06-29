from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class HardwareInfo:
    os: str
    python: str
    cpu_count: int
    ram_gb: float
    cuda_available: bool
    torch_cuda_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    torch_version: str | None
    recommended_profile: str
    notes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        gpu = self.gpu_name or "감지되지 않음"
        vram = f"{self.gpu_vram_gb:.1f} GB" if self.gpu_vram_gb else "-"
        notes = "\n".join([f"- {n}" for n in self.notes]) or "- 특이사항 없음"
        return f"""
### 현재 실행 컴퓨터 사양

| 항목 | 값 |
|---|---|
| OS | {self.os} |
| Python | {self.python} |
| CPU logical cores | {self.cpu_count} |
| RAM | {self.ram_gb:.1f} GB |
| CUDA GPU 감지 | {self.cuda_available} |
| torch CUDA 사용 가능 | {self.torch_cuda_available} |
| GPU | {gpu} |
| GPU VRAM | {vram} |
| torch | {self.torch_version or '-'} |
| 추천 프로필 | **{self.recommended_profile}** |

{notes}
""".strip()


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def _nvidia_smi_info() -> tuple[bool, Optional[str], Optional[float], str | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, None, None, None
    try:
        cmd = [
            exe,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5).strip()
        first = out.splitlines()[0] if out else ""
        if not first:
            return True, None, None, None
        parts = [p.strip() for p in first.split(",")]
        name = parts[0]
        vram_gb = float(parts[1]) / 1024 if len(parts) > 1 and parts[1] else None
        return True, name, vram_gb, None
    except Exception as e:
        return True, None, None, repr(e)


def detect_hardware() -> HardwareInfo:
    notes: list[str] = []
    cuda_hw, gpu_name, gpu_vram_gb, smi_error = _nvidia_smi_info()
    if smi_error:
        notes.append(f"nvidia-smi 실행은 감지되었지만 상세 조회 실패: {smi_error}")

    torch_cuda = False
    torch_version = None
    try:
        import torch
        torch_version = getattr(torch, "__version__", None)
        torch_cuda = bool(torch.cuda.is_available())
        if torch_cuda:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_vram_gb = props.total_memory / (1024 ** 3)
            except Exception:
                pass
    except Exception as e:
        notes.append(f"torch import 불가 또는 미설치: {type(e).__name__}")

    ram = _ram_gb()
    profile = choose_recommended_profile(cuda_hw, torch_cuda, gpu_vram_gb, ram)

    if cuda_hw and not torch_cuda:
        notes.append(
            "NVIDIA GPU는 감지되었지만 현재 torch가 CUDA를 사용할 수 없습니다. "
            "GPU를 쓰려면 install.py 실행 시 --torch cuda 옵션을 사용하세요."
        )
    if ram and ram < 12:
        notes.append("RAM이 12GB 미만이므로 CPU 저사양 프로필을 권장합니다.")

    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=ram,
        cuda_available=bool(cuda_hw),
        torch_cuda_available=torch_cuda,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        torch_version=torch_version,
        recommended_profile=profile,
        notes=notes,
    )


def choose_recommended_profile(cuda_hw: bool, torch_cuda: bool, gpu_vram_gb: Optional[float], ram_gb: float) -> str:
    if torch_cuda and gpu_vram_gb:
        if gpu_vram_gb >= 20:
            return "gpu_quality"
        if gpu_vram_gb >= 10:
            return "gpu_balanced"
        if gpu_vram_gb >= 6:
            return "gpu_light"
    if ram_gb >= 24:
        return "cpu_standard"
    return "cpu_low"


def hardware_json() -> str:
    return json.dumps(detect_hardware().to_dict(), ensure_ascii=False, indent=2)
