from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch

from ..config import get_config_value


REVE_BACKBONE_ID = "brain-bzh/reve-base"
REVE_POSITIONS_ID = "brain-bzh/reve-positions"


def normalize_reve_output(output: Any) -> torch.Tensor:
    """Normalize REVE outputs to [B, C, P, D] for downstream pooling."""
    if isinstance(output, dict):
        output = (
            output.get("last_hidden_state")
            or output.get("hidden_states")
            or output.get("output")
            or output.get("features")
        )
    elif hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state

    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported REVE output type: {type(output)!r}")

    if output.ndim == 4:
        return output
    if output.ndim == 3:
        return output.unsqueeze(1)
    if output.ndim == 2:
        return output.unsqueeze(1).unsqueeze(1)
    raise ValueError(f"Unexpected REVE output shape: {tuple(output.shape)}")


def pool_reve_features(output: Any) -> torch.Tensor:
    """Mean-pool REVE tokens to a fixed-width embedding."""
    tokens = normalize_reve_output(output)
    batch, _, _, hidden = tokens.shape
    return tokens.reshape(batch, -1, hidden).mean(dim=1)


def get_reve_cache_root() -> Path:
    cache_dir = get_config_value("embedding_cache_dir")
    if cache_dir is None:
        cache_dir = get_config_value("cache")
    if cache_dir is None:
        cache_dir = get_config_value("data")
    if cache_dir is None:
        cache_dir = "."
    root = Path(cache_dir).expanduser() / "reve_embedding_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_reve_cache_path(prefix: str, payload: Dict[str, Any]) -> Path:
    cache_key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return get_reve_cache_root() / f"{prefix}_{cache_key}.pt"
