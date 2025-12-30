# lejepa_bci_model.py

from __future__ import annotations
from typing import List, Dict, Optional, cast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset import make_dataset  # BCI specific loader
from .LaBraM.utils_2 import reverse_map_label, n_unique_labels
from .LaBraM import utils

import sys
sys.path.append("/teamspace/studios/this_studio")
from eegfmchallenge.models.eeglejepa import EEGLEJEPAConfig
from transformers import AutoModel

class ConcreteLeJEPABCI(nn.Module):
    def __init__(self, num_classes: int, pretrained_path: str = None):
        super().__init__()
        
        DIM = 384
        # Same config as clinical to ensure weights load correctly
        cfg = EEGLEJEPAConfig(
            name="EEGLEJEPA",
            dim=DIM,
            proj_dim=16,
            patch_size=25,
            max_time=1500, # BCI trials (4s) are only ~160 patches, so 1500 is plenty
            n_channels=128,
            # ... (matching your clinical/pretrain configs)
        )
        self.backbone = cfg.build()

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("model.", ""): v for k, v in state.items()}
            self.backbone.load_state_dict(state, strict=False)

        # Freeze Backbone (Standard for BCI benchmarks to test feature quality)
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        self.head = nn.Sequential(
            nn.LayerNorm(DIM),
            nn.Linear(DIM, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, coords):
        # Uses your new downstream method
        outputs = self.backbone.forward_downstream(x=x, channel_locations=coords)
        
        # Use the 384-dim CLS token
        cls = outputs["cls_token"]
        if cls.dim() == 3:
            cls = cls.mean(dim=1)
            
        return self.head(cls)

class EEGLeJEPABCIModel(AbstractModel):
    def __init__(self, pretrained_path: str = None):
        super().__init__("LeJEPABCI")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_path = pretrained_path
        
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", 
            trust_remote_code=True,
            token="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"
        ).to(self.device)

    def _get_coords(self, ch_names):
        clean_names = [c.replace("EEG", "").strip() for c in ch_names]
        c = self.pos_bank(clean_names=clean_names)
        if isinstance(c, dict):
            c = c.get("positions", c.get("coords", c.get("last_hidden_state")))
        if c.dim() == 3: c = c.squeeze(0)
        return c.float().to(self.device)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        task_name = meta[0]["task_name"]
        num_classes = n_unique_labels(task_name)
        self.model = ConcreteLeJEPABCI(num_classes, self.pretrained_path).to(self.device)

        # BCI datasets use a specific make_dataset that expects split_size
        # No 'use_cache' required here usually, but check your version's signature
        datasets = [make_dataset(X_, y_, task_name, m_["sampling_frequency"], m_["channel_names"], train=True, split_size=0.15)
                    for X_, y_, m_ in zip(X, y, meta)]
        
        # Accessing the train/val split from the first dataset in the list
        ds_train, ds_val = datasets[0]
        
        train_loader = DataLoader(ds_train, batch_size=64, shuffle=True, pin_memory=True)
        coords = self._get_coords(ds_train.ch_names)

        optimizer = optim.AdamW(self.model.head.parameters(), lr=4e-4)
        
        self.model.train()
        for epoch in range(10): # BCI tasks converge quickly
            for x, y_batch in tqdm(train_loader, desc=f"BCI Fit Epoch {epoch}"):
                x, y_batch = x.to(self.device), y_batch.to(self.device).argmax(dim=1)
                cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
                
                optimizer.zero_grad()
                logits = self.model(x, cb)
                loss = self.model.loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        task_name = meta[0]["task_name"]
        self.model.eval()
        
        # Test dataset for BCI: train=False, split_size=0
        ds_test = make_dataset(X[0], None, task_name, meta[0]["sampling_frequency"], meta[0]["channel_names"], train=False, split_size=0)
        loader = DataLoader(ds_test, batch_size=64, shuffle=False)
        coords = self._get_coords(ds_test.ch_names)

        preds_all = []
        for x in tqdm(loader, desc="BCI Predicting"):
            x = x.to(self.device)
            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
            logits = self.model(x, cb)
            preds_all.append(torch.argmax(logits, dim=1).cpu().numpy())

        predictions = np.concatenate(preds_all)
        return np.array([reverse_map_label(idx, task_name) for idx in predictions])