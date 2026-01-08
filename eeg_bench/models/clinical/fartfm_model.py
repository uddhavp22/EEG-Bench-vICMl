# fartfm_clinical_model.py

from __future__ import annotations
from typing import List, Dict, Optional, cast
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import gc
from pathlib import Path
from ..abstract_model import AbstractModel
from ...config import get_config_value
from transformers import AutoModel
# LaBraM Clinical Utilities
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Fartfm Models
import sys
sys.path.append("/teamspace/studios/this_studio")
from eegfmchallenge.models.eeglejepa import EEGLEJEPAConfig
from eegfmchallenge.models.patch_embedder import ConvPatchEmbedderConfig
from eegfmchallenge.models.channel_mixer import DynamicChannelMixerConfig
from eegfmchallenge.models.common import EncoderConfig

from ...utils import wandb_utils
import pickle

class ConcreteFartfmClinical(nn.Module):
    def __init__(
        self,
        num_classes,
        num_labels_per_chunk,
        base_path=None,
        version=None,
        freeze_encoder=False,
    ):
        super().__init__()

        DIM = 384
        self.is_multilabel_task = num_labels_per_chunk is not None

        # ------------------------------------------------------------
        # Pretrained config / checkpoint resolution (SAFE)
        # ------------------------------------------------------------
        config_path = None
        pretrained_path = None

        if base_path is not None and version is not None:
            base_path = Path(base_path) / f"version_{version}"

            candidate_config = base_path / "config" / "config.pkl"
            candidate_ckpt = base_path / "checkpoints" / "last.ckpt"

            if candidate_config.exists():
                config_path = candidate_config
            else:
                print(f"[FartfmClinical] No config found at {candidate_config}. Using default config.")

            if candidate_ckpt.exists():
                pretrained_path = candidate_ckpt
            else:
                print(f"[FartfmClinical] No checkpoint found at {candidate_ckpt}. Training from scratch.")

        # ------------------------------------------------------------
        # Build model config
        # ------------------------------------------------------------
        if config_path is not None:
            with open(config_path, "rb") as f:
                pretrain_config = pickle.load(f)
            cfg = EEGLEJEPAConfig(**pretrain_config["model"])
        else:
            cfg = EEGLEJEPAConfig(
                name="EEGLEJEPA",
                dim=384,
                proj_dim=16,
                patch_size=25,
                n_channels=128,
                max_time=1500,
                patch_embedder=ConvPatchEmbedderConfig(
                    name="ConvPatchEmbedder",
                    preserve_channels=False,
                ),
                channel_mixer_config=DynamicChannelMixerConfig(
                    name="DynamicChannelMixer",
                    coord_dim=3,
                    output_channels=64,
                ),
                encoder_config=EncoderConfig(
                    dim=384,
                    depth=12,
                    heads=6,
                    use_flash_attn=True,
                ),
                predictor_config=EncoderConfig(
                    dim=128,
                    depth=4,
                    heads=4,
                    use_flash_attn=True,
                ),
                masking={
                    "mask_ratio": 0.5,
                    "block_size_range": [5, 10],
                    "strategy_probs": [1.0, 0.0, 0.0],
                },
            
            )

        # ------------------------------------------------------------
        # Build backbone
        # ------------------------------------------------------------
        self.backbone = cfg.build()

        # ------------------------------------------------------------
        # Load pretrained weights (if available)
        # ------------------------------------------------------------
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("model.", ""): v for k, v in state.items()}
            self.backbone.load_state_dict(state, strict=False)
            print("[FartfmClinical] Loaded pretrained weights")

        # ------------------------------------------------------------
        # Freeze encoder if requested
        # ------------------------------------------------------------
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.train()

        out_dim = num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1)
        self.head = nn.Linear(DIM, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes


    def forward(self, x, coords):
        # Uses your downstream forward
        outputs = self.backbone.forward_downstream(x=x, channel_locations=coords)
        cls = outputs["cls_token"]
        #print("cls_token shape",cls.shape)
        if cls.dim() == 3:
            cls = cls.mean(dim=1)
        
        logits = self.head(cls)
        if self.is_multilabel_task:
            logits = logits.reshape(x.shape[0], self.num_classes, -1)
        return logits

class FartfmClinicalModel(AbstractModel):
    def __init__(self, num_classes=2, num_labels_per_chunk=None, base_path="/teamspace/studios/this_studio/pretrained_lejepas",version=0):
        super().__init__("fartfmClinical")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_len_s = None if num_labels_per_chunk is None else 16
        self.num_labels_per_chunk = num_labels_per_chunk
        
        # Positions bank

        # Get HuggingFace token from environment variable for security
        hf_token = os.environ.get('HF_TOKEN', None)
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            torch_dtype="auto",
            token=hf_token
        ).to(self.device)

        self.model = ConcreteFartfmClinical(
            num_classes=num_classes, 
            num_labels_per_chunk=num_labels_per_chunk,
            base_path=base_path,
            version=version
        ).to(self.device)

    def _coords(self, ch_names):
        import sys
        from io import StringIO

        names = [c.replace("EEG", "").strip() for c in ch_names]
        #print(names)

        # Suppress stdout from pos_bank model
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            c = self.pos_bank(names)
        finally:
            sys.stdout = old_stdout

        if isinstance(c, dict):
            c = c.get("positions", c.get("coords", c.get("last_hidden_state")))

        c = c.squeeze(0) if c.dim() == 3 else c

        # Handle case where pos_bank returns fewer coords than requested channels
        if c.shape[0] < len(names):
            # Pad with mean of existing coords for missing channels
            mean_coord = c.mean(dim=0, keepdim=True)
            padding = mean_coord.repeat(len(names) - c.shape[0], 1)
            c = torch.cat([c, padding], dim=0)

        return c.to(self.device).float()

    def fit(self, X, y, meta) -> None:
        task_name = meta[0]["task_name"]
        
        # 1. Dataset Loading (matching LaBraM exact args)
        dataset_train = make_dataset_2(X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=False)
        
        # 2. Safety Check: If dataset is empty, the .h5 cache is likely bad
        if len(dataset_train) == 0:
            print("[Warning] Dataset empty. Retrying without cache...")
            dataset_train = make_dataset_2(X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=False)

        # 3. Validation Split
        val_split = 0.2
        dataset_train, dataset_val = dataset_train.split_train_val(val_split)

        # 4. DataLoader Setup
        bs = 64 if self.chunk_len_s else 1
        train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=bs, shuffle=False)
        
        # 5. Training Setup
        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=4e-4)
        
        # Note: coords are computed per batch from actual channel names
        # not pre-computed, to handle variable channels across samples

        num_epochs=30

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            total_acc_samples = 0
            for x, yb, channels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                x, yb = x.to(self.device), yb.to(self.device)

                # Get coordinates for this batch's actual channels
                if channels != -1 and channels[0] != -1:
                    batch_ch_names = [ch_arr[0] for ch_arr in channels]
                    cb = self._coords(batch_ch_names)
                else:
                    # Fallback to dataset-level channels if not available
                    cb = self._coords(dataset_train.ch_names)

                cb = cb.unsqueeze(0).expand(x.size(0), -1, -1)
                
                optimizer.zero_grad()
                logits = self.model(x, cb)
                loss = self.model.loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                if logits.dim() == 2:
                    preds = torch.argmax(logits, dim=1)
                    target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total_acc_samples += x.size(0)
                
                # Manual memory cleanup like LaBraM
                del x, yb, logits; torch.cuda.empty_cache()

            val_loss = 0.0
            val_samples = 0
            val_correct = 0
            val_acc_samples = 0
            self.model.eval()
            with torch.no_grad():
                for x, yb, channels in tqdm(val_loader, desc=f"Val {epoch}"):
                    x, yb = x.to(self.device), yb.to(self.device)

                    # Get coordinates for this batch's actual channels
                    if channels != -1 and channels[0] != -1:
                        batch_ch_names = [ch_arr[0] for ch_arr in channels]
                        cb = self._coords(batch_ch_names)
                    else:
                        cb = self._coords(dataset_val.ch_names)

                    cb = cb.unsqueeze(0).expand(x.size(0), -1, -1)
                    logits = self.model(x, cb)
                    loss = self.model.loss_fn(logits, yb)
                    val_loss += loss.item() * x.size(0)
                    val_samples += x.size(0)
                    if logits.dim() == 2:
                        preds = torch.argmax(logits, dim=1)
                        target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                        val_correct += (preds == target).sum().item()
                        val_acc_samples += x.size(0)
                    del x, yb, logits; torch.cuda.empty_cache()

            if self.wandb_run and total_samples:
                metrics = {f"{self.name}/train_loss": total_loss / total_samples}
                if total_acc_samples:
                    metrics[f"{self.name}/train_acc"] = correct / total_acc_samples
                if val_samples:
                    metrics[f"{self.name}/val_loss"] = val_loss / val_samples
                    if val_acc_samples:
                        metrics[f"{self.name}/val_acc"] = val_correct / val_acc_samples
                wandb_utils.log(metrics, step=epoch + 1)

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        task_name = meta[0]["task_name"]

        #

        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name,
            chunk_len_s=self.chunk_len_s, 
            is_train=False,
            use_cache=False
        )
        
        if len(dataset_test) == 0:
            return np.array([])

        bs = 64 if self.chunk_len_s else 1
        loader = DataLoader(dataset_test, batch_size=bs, shuffle=False)
        self.model.eval()

        preds_all = []
        idx_map_all = []

        for batch in tqdm(loader, desc="Predicting"):
            x, idx, channels = batch
            x = x.to(self.device)

            # Get coordinates for this batch's actual channels
            if channels != -1 and channels[0] != -1:
                batch_ch_names = [ch_arr[0] for ch_arr in channels]
                cb = self._coords(batch_ch_names)
            else:
                cb = self._coords(dataset_test.ch_names)

            cb = cb.unsqueeze(0).expand(x.size(0), -1, -1)

            logits = self.model(x, cb)
            
            # Get window-level predictions
            pred = torch.argmax(logits, dim=1)
            preds_all.append(pred.cpu().numpy())
            idx_map_all.append(idx.cpu().numpy())

        preds = np.concatenate(preds_all)
        idx_map = np.concatenate(idx_map_all)

        # Majority voting: Combine windows back into 1 patient prediction
        unique_indices = np.unique(idx_map)
        final_predictions = []
        for i in unique_indices:
            patient_votes = preds[idx_map == i]
            final_predictions.append(Counter(patient_votes).most_common(1)[0][0])

        return np.array([map_label_reverse(p, task_name) for p in final_predictions])
