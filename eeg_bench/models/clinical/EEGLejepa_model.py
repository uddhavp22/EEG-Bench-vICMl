# lejepa_clinical_model.py

from __future__ import annotations
from typing import List, Dict, Optional, cast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import gc

from ..abstract_model import AbstractModel
from ...config import get_config_value

# LaBraM Clinical Utilities
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from .LaBraM import utils


# LeJEPA Models
import sys
sys.path.append("/teamspace/studios/this_studio")
from eegfmchallenge.models.eeglejepa import EEGLEJEPAConfig

from eegfmchallenge.models.patch_embedder import ConvPatchEmbedderConfig
from eegfmchallenge.models.channel_mixer import DynamicChannelMixerConfig
from eegfmchallenge.models.common import EncoderConfig
from transformers import AutoModel

class ConcreteLeJEPAClinical(nn.Module):
    def __init__(self, num_classes, num_labels_per_chunk, pretrained_path=None):
        super().__init__()
        DIM = 384
        self.is_multilabel_task = num_labels_per_chunk is not None
        
        cfg = EEGLEJEPAConfig(
            name="EEGLEJEPA",
            dim=384,
            proj_dim=16,
            patch_size=25,
            n_channels=128,
            max_time=2000,
            # ADD THESE THREE REQUIRED FIELDS:
            patch_embedder=ConvPatchEmbedderConfig(name="ConvPatchEmbedder", preserve_channels=False),
            channel_mixer_config=DynamicChannelMixerConfig(name="DynamicChannelMixer", coord_dim=3, output_channels=64),
            encoder_config=EncoderConfig(dim=384, depth=12, heads=6, use_flash_attn=True),
            # Optional but good to include:
            predictor_config=EncoderConfig(dim=128, depth=4, heads=4, use_flash_attn=True),
            masking={"mask_ratio": 0.5, "block_size_range": [5, 10], "strategy_probs": [1.0, 0.0, 0.0]},
            use_scaler=False,
        )
        self.backbone = cfg.build()

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("model.", ""): v for k, v in state.items()}
            self.backbone.load_state_dict(state, strict=False)

        # Head follows LaBraM clinical style: Linear mapping to classes
        out_dim = num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1)
        self.head = nn.Linear(DIM, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, coords):
        # Uses your downstream forward
        outputs = self.backbone.forward_downstream(x=x, channel_locations=coords)
        cls = outputs["cls_token"]
        if cls.dim() == 3:
            cls = cls.mean(dim=1)
        
        logits = self.head(cls)
        if self.is_multilabel_task:
            logits = logits.reshape(x.shape[0], self.num_classes, -1)
        return logits

class EEGLeJEPAClinicalModel(AbstractModel):
    def __init__(self, num_classes=2, num_labels_per_chunk=None, pretrained_path=None):
        super().__init__("LeJEPAClinical")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_len_s = None if num_labels_per_chunk is None else 16
        self.num_labels_per_chunk = num_labels_per_chunk
        
        # Positions bank
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", trust_remote_code=True
        ).to(self.device)

        self.model = ConcreteLeJEPAClinical(
            num_classes=num_classes, 
            num_labels_per_chunk=num_labels_per_chunk,
            pretrained_path=pretrained_path
        ).to(self.device)

    def _coords(self, ch_names):
        names = [c.replace("EEG", "").strip() for c in ch_names]
        c = self.pos_bank(names)
        if isinstance(c, dict):
            c = c.get("positions", c.get("coords", c.get("last_hidden_state")))
        return c.squeeze(0).to(self.device).float() if c.dim() == 3 else c.to(self.device).float()

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
        
        coords_train = self._coords(dataset_train.ch_names)
        coords_val = self._coords(dataset_val.ch_names)

        for epoch in range(1, 4):
            self.model.train()
            for x, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
                x, yb = x.to(self.device), yb.to(self.device)
                cb = coords_train.unsqueeze(0).expand(x.size(0), -1, -1)
                
                optimizer.zero_grad()
                logits = self.model(x, cb)
                loss = self.model.loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                
                # Manual memory cleanup like LaBraM
                del x, yb, logits; torch.cuda.empty_cache()

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        task_name = meta[0]["task_name"]
        
        # FIX: Ensure chunk_len_s is 16, NOT None
        # This forces the test set to be broken into 16s windows
        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name, 
            chunk_len_s=16, # Match training!
            is_train=False, 
            use_cache=True
        )
        
        if len(dataset_test) == 0:
            return np.array([])
            
        loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
        coords = self._coords(dataset_test.ch_names)
        self.model.eval()

        preds_all = []
        idx_map_all = []

        for batch in tqdm(loader, desc="Predicting"):
            x, idx, _ = batch
            x = x.to(self.device)
            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)

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
