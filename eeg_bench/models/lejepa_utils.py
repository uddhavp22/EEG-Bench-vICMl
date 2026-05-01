# Shared helpers for the BCI and clinical EEGLeJEPA model wrappers.
#
# These helpers were originally duplicated byte-for-byte across
# ``eeg_bench/models/bci/EEGLeJEPA_model.py`` and
# ``eeg_bench/models/clinical/EEGLejepa_model.py``. Centralizing them here
# keeps the two wrappers in sync without changing their architecture.

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Cache version env var (shared so the sweep runner only sets it once).
EMBED_CACHE_VERSION = os.getenv("EMBED_CACHE_VERSION", "v2")


# eegfm classes are populated dynamically via _setup_eegfm_imports().
EEGLEJEPAConfig = None
ConvPatchEmbedderConfig = None
DynamicChannelMixerConfig = None
EncoderConfig = None


def _setup_eegfm_imports(eegfm_path: Optional[str] = None):
    """Populate eegfm class globals, optionally adding ``eegfm_path`` to sys.path."""
    global EEGLEJEPAConfig, ConvPatchEmbedderConfig, DynamicChannelMixerConfig, EncoderConfig

    if eegfm_path and eegfm_path not in sys.path:
        sys.path.insert(0, eegfm_path)
        logger.info(f"Added eegfm path to sys.path: {eegfm_path}")

    from eegfmchallenge.models.eeglejepa import EEGLEJEPAConfig as _EEGLEJEPAConfig
    from eegfmchallenge.models.patch_embedder import ConvPatchEmbedderConfig as _ConvPatchEmbedderConfig
    from eegfmchallenge.models.channel_mixer import DynamicChannelMixerConfig as _DynamicChannelMixerConfig
    from eegfmchallenge.models.common import EncoderConfig as _EncoderConfig

    EEGLEJEPAConfig = _EEGLEJEPAConfig
    ConvPatchEmbedderConfig = _ConvPatchEmbedderConfig
    DynamicChannelMixerConfig = _DynamicChannelMixerConfig
    EncoderConfig = _EncoderConfig


def probe_layer_suffix(probe_layer: Optional[float]) -> str:
    """Filename/cache-key suffix encoding the fractional probe layer (or empty).

    Example: ``0.333 -> "_Lf033"``. Returns ``""`` when ``probe_layer is None``
    so existing caches/filenames are preserved.
    """
    if probe_layer is None:
        return ""
    return f"_Lf{round(probe_layer * 100):03d}"


def extract_layer_cls(
    backbone, x: torch.Tensor, cb: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    """Extract pooled representation from a specific encoder block.

    The block is selected by 0-indexed ``layer_idx`` without modifying the
    backbone. Uses CLS pooling when the backbone has a CLS token, otherwise
    falls back to mean pooling over tokens.
    """
    if "V2" in backbone.channel_mixer.__class__.__name__:
        patch_data = backbone._dynamic_create_patches_and_embeddings(x, cb)
    else:
        patch_data = backbone._create_patches_and_embeddings(x, cb)
    tokens = patch_data["tokens"]
    context_input = tokens
    if backbone.CLS is not None:
        cls_tokens = backbone.CLS.expand(tokens.size(0), -1, -1)
        context_input = torch.cat((cls_tokens, tokens), dim=1)
    _, intermediates = backbone.context_encoder(context_input, return_hiddens=True)
    h = intermediates.hiddens[layer_idx]
    return h[:, 0, :] if backbone.CLS is not None else h.mean(dim=1)


def build_simple_probe_head(dim: int, out_dim: int, probe_head: str) -> nn.Module:
    """Build a linear or MLP probe head.

    Returns ``None`` for ``"attentive"`` so callers can plug their own attentive
    implementation (clinical) or raise (BCI).
    """
    if probe_head == "linear":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
        )
    if probe_head == "mlp":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(dim, out_dim),
        )
    if probe_head == "attentive":
        return None
    raise ValueError(f"Unsupported LeJEPA probe head: {probe_head}")
