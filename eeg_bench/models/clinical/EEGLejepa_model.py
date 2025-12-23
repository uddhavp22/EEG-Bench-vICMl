# lejepa_clinical_model.py

from __future__ import annotations

from typing import List, Dict, Optional, Tuple, cast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from ..abstract_model import AbstractModel
from ...config import get_config_value

# reuse your clinical dataset builder + label mapping
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from .LaBraM import utils  # for get_input_chans

# LeJEPA (your repo)
import sys
sys.path.append("/teamspace/studios/this_studio")
from eegfmchallenge.models.eeglejepa import EEGLEJEPAConfig
from eegfmchallenge.models.patch_embedder import ConvPatchEmbedderConfig
from eegfmchallenge.models.channel_mixer import DynamicChannelMixerConfig
from eegfmchallenge.models.common import EncoderConfig

# positions bank (you said pos_bank(coords) works)
from transformers import AutoModel


class ConcreteLeJEPAClinical(nn.Module):
    """
    Backbone + head. Supports:
      - single-label: logits (B, K)
      - multilabel-per-chunk: logits (B, K, L)
    """
    def __init__(
        self,
        num_classes: int,
        num_labels_per_chunk: Optional[int],
        freeze_encoder: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        DIM = 384
        PRED_DIM = 128
        PROJ_DIM = 16

        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk
        self.is_multilabel_task = num_labels_per_chunk is not None

        cfg = EEGLEJEPAConfig(
            name="EEGLEJEPA",
            dim=DIM,
            proj_dim=PROJ_DIM,
            patch_size=25,
            max_time=1500,
            n_channels=128,
            patch_embedder=ConvPatchEmbedderConfig(name="ConvPatchEmbedder", preserve_channels=False),
            encoder_config=EncoderConfig(dim=DIM, depth=12, heads=6, use_flash_attn=True),
            predictor_config=EncoderConfig(dim=PRED_DIM, depth=4, heads=4, use_flash_attn=True),
            channel_mixer_config=DynamicChannelMixerConfig(name="DynamicChannelMixer", coord_dim=3, output_channels=64),
            masking={"mask_ratio": 0.5, "block_size_range": [5, 6, 7, 8, 9, 10], "strategy_probs": [1.0, 0.0, 0.0]},
            use_scaler=False,
        )
        self.backbone = cfg.build()

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

        out_dim = num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(DIM),
            nn.Linear(DIM, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

        self.loss_fn = nn.CrossEntropyLoss()

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
        cls = outputs["global"]["cls_token"]
        if cls.dim() == 3:
            cls = cls.squeeze(1)  # (B, DIM)

        logits = self.head(cls)  # (B, K) or (B, K*L)

        if self.is_multilabel_task:
            B = logits.shape[0]
            logits = logits.view(B, self.num_classes, self.num_labels_per_chunk)  # (B, K, L)
        return logits


def _ce_loss(model: ConcreteLeJEPAClinical, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    single-label:
      logits (B,K), y (B,)
    multilabel-per-chunk:
      logits (B,K,L), y (B,L)
      -> flatten to (B*L,K) / (B*L,)
    """
    if not model.is_multilabel_task:
        return model.loss_fn(logits, y)

    # logits: (B,K,L), y: (B,L)
    logits2 = logits.permute(0, 2, 1).contiguous().view(-1, model.num_classes)  # (B*L, K)
    y2 = y.contiguous().view(-1)  # (B*L,)
    return model.loss_fn(logits2, y2)


@torch.no_grad()
def _infer_indices_majority(pred_idx: np.ndarray, indices_mapping: np.ndarray) -> np.ndarray:
    unique_idx = np.unique(indices_mapping)
    out = []
    for idx in unique_idx:
        votes = pred_idx[indices_mapping == idx]
        out.append(Counter(votes).most_common(1)[0][0])
    return np.asarray(out)


class EEGLeJEPAClinicalModel(AbstractModel):
    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        freeze_encoder: bool = True,
        pretrained_path: Optional[str] = None,
        batch_size: int = 64,
        max_epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_workers: int = 8,
        val_split: float = 0.2,
        patience: int = 10,
    ):
        super().__init__("LeJEPAClinical")
        assert torch.cuda.is_available(), "CUDA is required"

        self.device = torch.device("cuda")
        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk
        self.is_multilabel_task = num_labels_per_chunk is not None

        # match your clinical LaBraM logic
        self.chunk_len_s = None if num_labels_per_chunk is None else 16

        self.freeze_encoder = freeze_encoder
        self.pretrained_path = pretrained_path

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.val_split = val_split
        self.patience = patience

        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            torch_dtype="auto",
            token="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"
        )

        self.model = ConcreteLeJEPAClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            freeze_encoder=freeze_encoder,
            pretrained_path=pretrained_path,
        ).to(self.device)

    def _coords(self, channel_names: List[str]) -> torch.Tensor:
        coords = [c.replace("EEG", "").strip() for c in channel_names]
        c = self.pos_bank(coords)  # per you: returns (C,3)
        if isinstance(c, dict):
            # if remote_code returns dict-like, prefer a common key
            for k in ("coords", "positions", "last_hidden_state"):
                if k in c:
                    c = c[k]
                    break
        if c.dim() == 3 and c.shape[0] == 1:
            c = c.squeeze(0)
        return c.to(self.device).float()

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        task_name = meta[0]["task_name"]

        # class weights same as clinical LaBraM
        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        dataset_train = make_dataset_2(
            X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=True
        )
        ch_names = [c.upper() for c in dataset_train.ch_names]
        print("[LeJEPA] dataset_train.ch_names (first 20):", ch_names[:20], "len=", len(ch_names))
        if len(dataset_train) == 0:
            raise ValueError(
            f"[LeJEPA] dataset_train is empty. "
            f"Check channel selection + windowing. "
            f"Example meta[0]: name={meta[0].get('name')} sfreq={meta[0].get('sampling_frequency')} "
            f"n_ch={len(meta[0].get('channel_names', []))} chunk_len_s={self.chunk_len_s}"

        )

        if self.val_split is not None:
            dataset_train, dataset_val = dataset_train.split_train_val(self.val_split)
        else:
            dataset_val = None

        ch_names_train = dataset_train.ch_names
        ch_names_val = dataset_val.ch_names if dataset_val is not None else None

        bs = 1 if self.chunk_len_s is None else self.batch_size

        train_loader = DataLoader(
            dataset_train, batch_size=bs, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )
        val_loader = (
            DataLoader(dataset_val, batch_size=bs, shuffle=False, num_workers=self.num_workers, pin_memory=True)
            if dataset_val is not None
            else None
        )

        # only train head (and anything unfrozen)
        optim_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(optim_params, lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state = None
        patience_ctr = 0

        coords_train = self._coords(ch_names_train)

        coords_val = self._coords(ch_names_val) if ch_names_val is not None else None

        for _epoch in range(1, self.max_epochs + 1):
            # ---- train
            self.model.train()
            if self.freeze_encoder:
                self.model.backbone.eval()

            for batch in tqdm(train_loader, desc="LeJEPA clinical train", leave=True):
                x, yb, channels = batch  # matches your LaBraM clinical dataset contract

                yb = yb.to(self.device)
                if channels != -1 and channels[0] != -1:
                    chs = [arr[0] for arr in channels]
                    coords_b = self._coords(chs)
                else:
                    coords_b = coords_train

                x = x.to(self.device)
                cb = coords_b.unsqueeze(0).expand(x.size(0), -1, -1)

                optimizer.zero_grad(set_to_none=True)
                print("x.shape:", tuple(x.shape), "dtype:", x.dtype, "device:", x.device)
                if torch.is_tensor(cb):
                    print("cb.shape:", tuple(cb.shape), "cb min/max:", cb.min().item(), cb.max().item(), "dtype:", cb.dtype)
                else:
                    print("cb:", type(cb), cb)

                logits = self.model(x, cb)
                loss = _ce_loss(self.model, logits, yb)
                loss.backward()
                optimizer.step()

            # ---- val
            if val_loader is None:
                continue

            self.model.eval()
            total_loss = 0.0
            n = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="LeJEPA clinical val", leave=True):
                    x, yb, channels = batch
                    yb = yb.to(self.device)

                    if channels != -1 and channels[0] != -1:
                        chs = [arr[0] for arr in channels]
                        coords_b = self._coords(chs)
                    else:
                        coords_b = cast(torch.Tensor, coords_val)

                    x = x.to(self.device)
                    cb = coords_b.unsqueeze(0).expand(x.size(0), -1, -1)

                    logits = self.model(x, cb)
                    loss = _ce_loss(self.model, logits, yb)

                    total_loss += loss.item() * x.size(0)
                    n += x.size(0)

            val_loss = total_loss / max(n, 1)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state, strict=True)

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        task_name = meta[0]["task_name"]

        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name, self.chunk_len_s, is_train=False, use_cache=True
        )
        if len(dataset_test) == 0:
            return np.array([])

        ch_names = dataset_test.ch_names
        coords_default = self._coords(ch_names)

        bs = 1 if self.chunk_len_s is None else self.batch_size
        test_loader = DataLoader(
            dataset_test, batch_size=bs, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )

        self.model.eval()
        pred_idx_all = []
        idx_map_all = []

        for batch in tqdm(test_loader, desc="LeJEPA clinical test", leave=True):
            x, idx, channels = batch  # your dataset returns (x, idx, channels) at test time
            if channels != -1 and channels[0] != -1:
                chs = [arr[0] for arr in channels]
                coords_b = self._coords(chs)
            else:
                coords_b = coords_default

            x = x.to(self.device)
            cb = coords_b.unsqueeze(0).expand(x.size(0), -1, -1)

            logits = self.model(x, cb)

            if self.is_multilabel_task:
                # logits (B,K,L) -> per-window pred (B,L)
                pred = torch.argmax(logits, dim=1)  # (B,L)
                # keep raw multilabel predictions; dataset/benchmark expects these directly
                pred_idx_all.append(pred.detach().cpu().numpy())
                idx_map_all.append(idx.detach().cpu().numpy())
            else:
                pred = torch.argmax(logits, dim=1)  # (B,)
                pred_idx_all.append(pred.detach().cpu().numpy())
                idx_map_all.append(idx.detach().cpu().numpy())

        pred_idx = np.concatenate(pred_idx_all, axis=0)
        idx_map = np.concatenate(idx_map_all, axis=0)

        if self.chunk_len_s is not None and not self.is_multilabel_task:
            pred_idx = _infer_indices_majority(pred_idx, idx_map)

        # map to benchmark label space
        mapped = np.array([map_label_reverse(int(p), task_name) for p in np.asarray(pred_idx).reshape(-1)])
        return mapped
