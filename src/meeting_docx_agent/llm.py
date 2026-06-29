from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .model_store import ensure_transformers_model
from .paths import configure_environment


@dataclass
class LLMOptions:
    model_id: str
    device: str = "cpu"  # cpu, cuda, auto
    allow_download: bool = True
    max_new_tokens: int = 1200
    temperature: float = 0.0


class LocalTransformersLLM:
    def __init__(self, opts: LLMOptions):
        self.opts = opts
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self.model_path: Optional[Path] = None

    def load(self) -> "LocalTransformersLLM":
        configure_environment()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = ensure_transformers_model(self.opts.model_id, allow_download=self.opts.allow_download)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_cuda = self.opts.device in ("cuda", "auto") and torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        if use_cuda:
            dtype = torch.float16
            kwargs = dict(torch_dtype=dtype, device_map="auto")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), local_files_only=True, **kwargs)
            except TypeError:
                kwargs = dict(dtype=dtype, device_map="auto")
                self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), local_files_only=True, **kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), local_files_only=True)
            self.model.to("cpu")
        self.model.eval()
        return self

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        assert self.tokenizer is not None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>\n<assistant>\n"
        return self.tokenizer(text, return_tensors="pt")

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: Optional[int] = None) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        import torch
        inputs = self._build_inputs(system_prompt, user_prompt)
        if self.device == "cuda":
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        max_new = max_new_tokens or self.opts.max_new_tokens
        gen_kwargs = dict(
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.07,
            no_repeat_ngram_size=4,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


_LLM_CACHE: dict[tuple[str, str], LocalTransformersLLM] = {}


def get_llm(model_id: str, device: str, allow_download: bool = True) -> LocalTransformersLLM:
    key = (model_id, device)
    if key not in _LLM_CACHE:
        _LLM_CACHE[key] = LocalTransformersLLM(LLMOptions(model_id=model_id, device=device, allow_download=allow_download)).load()
    return _LLM_CACHE[key]


def unload_all_llms() -> None:
    for llm in list(_LLM_CACHE.values()):
        llm.unload()
    _LLM_CACHE.clear()
