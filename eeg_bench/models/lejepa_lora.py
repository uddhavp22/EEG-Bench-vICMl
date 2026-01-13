from __future__ import annotations

from typing import Iterable

import torch.nn as nn


_DEFAULT_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_qkv")


def _matches_target(name: str, target_modules: Iterable[str]) -> bool:
    return any(name.endswith(target) for target in target_modules)


def apply_lejepa_lora(backbone: nn.Module, lora_config) -> nn.Module:
    if not lora_config or not lora_config.enabled:
        return backbone

    target_modules = lora_config.target_modules or list(_DEFAULT_TARGET_MODULES)
    matched = [
        name
        for name, module in backbone.named_modules()
        if isinstance(module, nn.Linear) and _matches_target(name, target_modules)
    ]
    if not matched:
        raise ValueError(
            "LoRA enabled but no QKV Linear modules matched. "
            f"Targets: {list(target_modules)}"
        )

    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(backbone, lora)
    for name, param in peft_model.named_parameters():
        param.requires_grad = "lora_" in name
    return peft_model
