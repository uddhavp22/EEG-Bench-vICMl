# lejepa_bci_model.py

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..abstract_model import AbstractModel

# EEG-Bench label utilities (you already used these in LaBraM)
from .LaBraM.utils_2 import reverse_map_label, n_unique_labels

# LeJEPA imports (your repo)
from eegfmchallenge.models.eeglejepa import EEGLEJEPAConfig
from eegfmchallenge.models.patch_embedder import ConvPatchEmbedderConfig
from eegfmchallenge.models.channel_mixer import DynamicChannelMixerConfig
from eegfmchallenge.models.common import EncoderConfig

# Positions model
from transformers import AutoModel


def _y_to_class_index(y: np.ndarray) -> np.ndarray:
    """
    Accepts:
      - shape (N,) ints
      - shape (N,K) one-hot / probs
      - shape (N,) strings/objects -> maps to 0..K-1
    Returns:
      - shape (N,) int64 indices
      - and also returns mapping for string labels via side-channel if needed (handled elsewhere)
    """
    if y is None:
        raise ValueError("y is None in fit()")

    y = np.asarray(y)

    # One-hot / multi-dim
    if y.ndim > 1:
        return np.argmax(y, axis=1).astype(np.int64)

    # Already numeric
    if np.issubdtype(y.dtype, np.integer):
        return y.astype(np.int64)

    # Strings/objects -> map
    uniq = np.unique(y)
    lut = {lab: i for i, lab in enumerate(uniq.tolist())}
    return np.array([lut[v] for v in y.tolist()], dtype=np.int64)


class ConcreteLeJEPA(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freeze_encoder: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        DIM = 384
        PRED_DIM = 128
        PROJ_DIM = 16

        self.config = EEGLEJEPAConfig(
            name="EEGLEJEPA",
            dim=DIM,
            proj_dim=PROJ_DIM,
            patch_size=25,
            max_time=1500,
            n_channels=128,
            patch_embedder=ConvPatchEmbedderConfig(
                name="ConvPatchEmbedder",
                preserve_channels=False
            ),
            encoder_config=EncoderConfig(dim=DIM, depth=12, heads=6, use_flash_attn=True),
            predictor_config=EncoderConfig(dim=PRED_DIM, depth=4, heads=4, use_flash_attn=True),
            channel_mixer_config=DynamicChannelMixerConfig(
                name="DynamicChannelMixer",
                coord_dim=3,
                output_channels=64
            ),
            masking={
                "mask_ratio": 0.5,
                "block_size_range": [5, 6, 7, 8, 9, 10],
                "strategy_probs": [1.0, 0.0, 0.0],
            },
            use_scaler=False,
        )

        self.backbone = self.config.build()

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("model.", ""): v for k, v in state.items()}
            self.backbone.load_state_dict(state, strict=False)

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.train()

        self.head = nn.Sequential(
            nn.LayerNorm(DIM),
            nn.Linear(DIM, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, channel_locations: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        channel_locations: (B, C, 3)
        """
        outputs = self.backbone(
            x=x,
            channel_locations=channel_locations,
            n_global=1,
            n_local=0,
            global_patch_len=20,
            local_patch_len=0,
        )

        cls_token = outputs["global"]["cls_token"]  # (B, 1, DIM) or (B, DIM)
        if cls_token.dim() == 3:
            cls_token = cls_token.squeeze(1)
        return self.head(cls_token)


class LeJEPABBCIModel(AbstractModel):
    def __init__(
        self,
        freeze_encoder: bool = True,
        pretrained_path: Optional[str] = None,
        batch_size: int = 64,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_workers: int = 4,
    ):
        super().__init__("LeJEPA-BCI")
        assert torch.cuda.is_available(), "CUDA is required for this model"
        self.device = torch.device("cuda")

        self.freeze_encoder = freeze_encoder
        self.pretrained_path = pretrained_path

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = num_workers

        # pos_bank(coords) should work (per your note)
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            torch_dtype="auto",
        )

        self.model: Optional[ConcreteLeJEPA] = None

        # cache coords by exact channel list tuple
        self._coords_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

        self._task_name: Optional[str] = None

    def _get_coords(self, channel_names: List[str]) -> torch.Tensor:
        key = tuple([c.replace("EEG", "").strip() for c in channel_names])
        if key in self._coords_cache:
            return self._coords_cache[key]

        coords = list(key)  # list[str]
        coords_t = self.pos_bank(coords)  # should return (C,3) tensor per your assumption

        if isinstance(coords_t, dict):
            # common pattern in HF remote_code models
            # try a few conventional keys
            for k in ("coords", "positions", "embeddings", "last_hidden_state"):
                if k in coords_t:
                    coords_t = coords_t[k]
                    break

        if not torch.is_tensor(coords_t):
            raise TypeError("pos_bank(coords) did not return a Tensor-like output")

        # ensure shape (C,3)
        if coords_t.dim() == 3 and coords_t.shape[0] == 1:
            coords_t = coords_t.squeeze(0)
        coords_t = coords_t.to(self.device)
        coords_t = coords_t.float()

        self._coords_cache[key] = coords_t
        return coords_t

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        if len(meta) == 0:
            raise ValueError("meta is empty")

        self.validate_meta(meta[0])
        task_name = meta[0].get("task_name", None)
        if task_name is None:
            raise ValueError("meta[0] must contain 'task_name' for EEG-Bench tasks")
        self._task_name = task_name

        num_classes = n_unique_labels(task_name)

        if self.model is None:
            self.model = ConcreteLeJEPA(
                num_classes=num_classes,
                freeze_encoder=self.freeze_encoder,
                pretrained_path=self.pretrained_path,
            ).to(self.device)

        criterion = nn.CrossEntropyLoss()

        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)

        # Build per-dataset loaders (handles varying channels/meta across datasets)
        loaders: List[Tuple[DataLoader, torch.Tensor]] = []
        for X_i, y_i, meta_i in zip(X, y, meta):
            self.validate_meta(meta_i)
            xi = torch.tensor(np.asarray(X_i), dtype=torch.float32)
            yi = torch.tensor(_y_to_class_index(np.asarray(y_i)), dtype=torch.long)

            ds = TensorDataset(xi, yi)
            dl = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            coords = self._get_coords(meta_i["channel_names"])  # (C,3) on device
            loaders.append((dl, coords))

        self.model.train()
        if self.freeze_encoder:
            self.model.backbone.eval()

        for _ in range(self.epochs):
            for dl, coords in loaders:
                for xb, yb in tqdm(dl, desc="LeJEPA train", leave=False):
                    xb = xb.to(self.device, non_blocking=True)
                    yb = yb.to(self.device, non_blocking=True)

                    # (B,C,3)
                    cb = coords.unsqueeze(0).expand(xb.size(0), -1, -1)

                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(xb, cb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        if len(meta) == 0:
            raise ValueError("meta is empty")

        self.model.eval()

        all_preds: List[np.ndarray] = []

        for X_i, meta_i in zip(X, meta):
            self.validate_meta(meta_i)

            xi = torch.tensor(np.asarray(X_i), dtype=torch.float32)
            dl = DataLoader(
                TensorDataset(xi),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            coords = self._get_coords(meta_i["channel_names"])  # (C,3)
            preds_i: List[np.ndarray] = []

            for (xb,) in tqdm(dl, desc="LeJEPA infer", leave=False):
                xb = xb.to(self.device, non_blocking=True)
                cb = coords.unsqueeze(0).expand(xb.size(0), -1, -1)

                logits = self.model(xb, cb)
                pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
                preds_i.append(pred_idx)

            pred_idx = np.concatenate(preds_i, axis=0)

            # Map to benchmark label space (strings) like LaBraM does
            task_name = meta_i.get("task_name", self._task_name)
            mapped = np.array([reverse_map_label(int(i), task_name) for i in pred_idx])
            all_preds.append(mapped)

        return np.concatenate(all_preds, axis=0)
