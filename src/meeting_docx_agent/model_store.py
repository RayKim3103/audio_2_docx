from __future__ import annotations

from pathlib import Path

from .paths import HF_HUB_CACHE, MODEL_DIR, configure_environment


def repo_to_dir(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def transformers_model_dir(repo_id: str) -> Path:
    return MODEL_DIR / "transformers" / repo_to_dir(repo_id)


def faster_whisper_model_dir() -> Path:
    return MODEL_DIR / "faster_whisper"


def looks_like_transformers_model_dir(path: Path) -> bool:
    path = Path(path)
    if not (path / "config.json").exists():
        return False
    has_tokenizer = any((path / name).exists() for name in ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"])
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")) or (path / "model.safetensors.index.json").exists()
    return has_tokenizer and has_weights


def ensure_transformers_model(repo_id: str, allow_download: bool = True) -> Path:
    """Download a HF model into .agent_runtime/models, then always load from local path."""
    configure_environment()
    target = transformers_model_dir(repo_id)
    if looks_like_transformers_model_dir(target):
        return target
    if not allow_download:
        raise FileNotFoundError(f"모델이 로컬에 없습니다: {repo_id} -> {target}")

    target.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    kwargs = dict(
        repo_id=repo_id,
        local_dir=str(target),
        cache_dir=str(HF_HUB_CACHE),
        allow_patterns=[
            "*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "vocab*", "merges.txt", "*.py"
        ],
        ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.gguf", "*.bin"],
    )
    try:
        snapshot_download(local_dir_use_symlinks=False, **kwargs)
    except TypeError:
        snapshot_download(**kwargs)

    if not looks_like_transformers_model_dir(target):
        raise RuntimeError(f"모델 다운로드 후에도 필요한 파일을 확인하지 못했습니다: {target}")
    return target
