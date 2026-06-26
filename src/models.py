"""models.py — model registry + quantized loading for the efficiency sweep.

One entry point, `load_model(hf_id, quant, family)`, returns `(model, processor)`
with the requested bitsandbytes quantization applied at load time. The same loader
serves prompting and (post-merge) LoRA cells; LoRA training loads fp16 then attaches
adapters in train_lora.py.
"""
from __future__ import annotations

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

try:  # transformers ≥5 renamed Vision2Seq → ImageTextToText
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:  # transformers <5
    from transformers import AutoModelForVision2Seq as AutoVLM


def _bnb_config(quant: dict) -> BitsAndBytesConfig | None:
    """Translate one `quant:` matrix entry into a BitsAndBytesConfig (or None for fp16)."""
    if quant.get("bnb_8bit"):
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant.get("bnb_4bit"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quant.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_model(hf_id: str, quant: dict, family: str = "smolvlm", device: str = "cuda:0"):
    """Load a VLM + its processor under the given quant config.

    fp16 lands directly on `device`; bitsandbytes paths use device_map to place the
    quantized weights. AutoModelForVision2Seq covers both SmolVLM (Idefics3) and
    Qwen2-VL via their auto classes.
    """
    bnb = _bnb_config(quant)
    load_dtype = getattr(torch, quant.get("load_dtype", "float16"))

    processor = AutoProcessor.from_pretrained(hf_id)
    kwargs = dict(torch_dtype=load_dtype, trust_remote_code=True)
    if bnb is not None:
        kwargs.update(quantization_config=bnb, device_map={"": device})
    model = AutoVLM.from_pretrained(hf_id, **kwargs)
    if bnb is None:
        model = model.to(device)
    model.eval()
    return model, processor


def model_footprint_mb(model) -> float:
    """Approx on-device weight footprint in MB (counts quantized byte sizes correctly)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 ** 2)
