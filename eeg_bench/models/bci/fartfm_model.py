# fartfm_bci_model.py

from __future__ import annotations
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset import make_dataset  # BCI specific loader
from .LaBraM.utils_2 import calc_class_weights, reverse_map_label, n_unique_labels
from joblib import Memory
from ...config import get_config_value
from ...utils import wandb_utils

import os
import sys
sys.path.append("/teamspace/studios/this_studio")
from eegfmchallenge.models.eeglejepa import EEGLEJEPAConfig
from eegfmchallenge.models.patch_embedder import ConvPatchEmbedderConfig
from eegfmchallenge.models.channel_mixer import DynamicChannelMixerConfig
from eegfmchallenge.models.common import EncoderConfig
from transformers import AutoModel
import random
import math

class ConcreteFartfmBCI(nn.Module):
    def __init__(self, num_classes: int, pretrained_path: str | None = None, freeze_encoder: bool = True):
        super().__init__()
        
        DIM = 384
        cfg = EEGLEJEPAConfig(
            name="fartfm",
            dim=DIM,
            proj_dim=16,
            patch_size=25,
            n_channels=128,
            max_time=1500,
            patch_embedder=ConvPatchEmbedderConfig(name="ConvPatchEmbedder", preserve_channels=False),
            channel_mixer_config=DynamicChannelMixerConfig(name="DynamicChannelMixer", coord_dim=3, output_channels=64),
            encoder_config=EncoderConfig(dim=384, depth=12, heads=6, use_flash_attn=True),
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

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            
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

class FartfmBCIModel(AbstractModel):
    def __init__(self, pretrained_path: str | None = None, freeze_encoder: bool = True):
        super().__init__("fartfmBCI")
        assert torch.cuda.is_available(), "CUDA is not available"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_path = pretrained_path
        self.freeze_encoder = freeze_encoder
        self.cache = Memory(location=get_config_value("cache"), verbose=0)
        
        # Get HuggingFace token from environment variable for security
        hf_token = os.environ.get('HF_TOKEN', None)
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            token=hf_token
        ).to(self.device)

    def _get_coords(self, ch_names):
        clean_names = [c.replace("EEG", "").strip() for c in ch_names]
        c = self.pos_bank(clean_names)
        if isinstance(c, dict):
            c = c.get("positions", c.get("coords", c.get("last_hidden_state")))
        if c.dim() == 3:
            c = c.squeeze(0)
        return c.float().to(self.device)

    def _train_epoch(self, dataloader, optimizer, scheduler, coords):
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        for x, y_batch in tqdm(dataloader, desc="Training", leave=False):
            x, y_batch = x.to(self.device), y_batch.to(self.device).argmax(dim=1)
            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)

            optimizer.zero_grad()
            logits = self.model(x, cb)
            loss = self.model.loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += (preds == y_batch).sum().item()
            total_samples += x.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def _validate_epoch(self, dataloader, coords):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        with torch.no_grad():
            for x, y_batch in tqdm(dataloader, desc="Validation", leave=False):
                x, y_batch = x.to(self.device), y_batch.to(self.device).argmax(dim=1)
                cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
                logits = self.model(x, cb)
                loss = self.model.loss_fn(logits, y_batch)

                running_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                running_corrects += (preds == y_batch).sum().item()
                total_samples += x.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        task_name = meta[0]["task_name"]
        num_classes = n_unique_labels(task_name)
        self.model = ConcreteFartfmBCI(num_classes, self.pretrained_path, freeze_encoder=self.freeze_encoder).to(self.device)

        # Pass model_name to enable FARTFM-specific preprocessing (250 Hz + defosse scaling)
        datasets = [self.cache.cache(make_dataset)(X_, y_, task_name, m_["sampling_frequency"], m_["channel_names"], train=True, split_size=0.15, model_name="fartfmBCI")
                    for X_, y_, m_ in zip(X, y, meta)]
        
        dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_val_list = [dataset[1] for dataset in datasets]
        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]
        dataset_val_list = [dataset for dataset in dataset_val_list if len(dataset) > 0]
        ch_names_list_train = [dataset.ch_names for dataset in dataset_train_list]
        ch_names_list_val = [dataset.ch_names for dataset in dataset_val_list]

        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        del X, y, meta
        torch.cuda.empty_cache()

        batch_size = 64
        num_workers = 8
        train_loader_list = [
            DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
            for train_dataset in dataset_train_list
        ]
        valid_loader_list = [
            DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
            for valid_dataset in dataset_val_list
        ]

        max_epochs = 30
        steps_per_epoch = math.ceil(sum(len(train_loader) for train_loader in train_loader_list))
        max_lr = 4e-4

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            pct_start=0.2,
        )

        patience = 10
        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(1, max_epochs + 1):
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            num_train_batches = 0
            train_pairs = list(zip(train_loader_list, ch_names_list_train))
            random.shuffle(train_pairs)

            for train_loader, ch_names in train_pairs:
                coords = self._get_coords(ch_names)
                train_loss, train_acc = self._train_epoch(train_loader, optimizer, scheduler, coords)
                epoch_train_loss += train_loss
                epoch_train_acc += train_acc
                num_train_batches += 1

            avg_train_loss = epoch_train_loss / num_train_batches
            avg_train_acc = epoch_train_acc / num_train_batches

            if valid_loader_list:
                epoch_val_loss = 0.0
                epoch_val_acc = 0.0
                num_val_batches = 0
                for valid_loader, ch_names in zip(valid_loader_list, ch_names_list_val):
                    coords = self._get_coords(ch_names)
                    val_loss, val_acc = self._validate_epoch(valid_loader, coords)
                    epoch_val_loss += val_loss
                    epoch_val_acc += val_acc
                    num_val_batches += 1

                avg_val_loss = epoch_val_loss / num_val_batches
                avg_val_acc = epoch_val_acc / num_val_batches

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.wandb_run:
                    wandb_utils.log(
                        {
                            f"{self.name}/train_loss": avg_train_loss,
                            f"{self.name}/train_acc": avg_train_acc,
                            f"{self.name}/val_loss": avg_val_loss,
                            f"{self.name}/val_acc": avg_val_acc,
                        },
                        step=epoch,
                    )

                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} (Patience: {patience}) due to no improvement in validation loss."
                    )
                    break
            else:
                if self.wandb_run:
                    wandb_utils.log(
                        {
                            f"{self.name}/train_loss": avg_train_loss,
                            f"{self.name}/train_acc": avg_train_acc,
                        },
                        step=epoch,
                    )

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        task_name = meta[0]["task_name"]
        self.model.eval()

        # Pass model_name to enable FARTFM-specific preprocessing (250 Hz + defosse scaling)
        dataset_test_list = [self.cache.cache(make_dataset)(X_, None, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=False, split_size=0, model_name="fartfmBCI")
                             for X_, meta_ in zip(X, meta)]
        dataset_test_list = [dataset for dataset in dataset_test_list if len(dataset) > 0]
        ch_names_list = [dataset.ch_names for dataset in dataset_test_list]

        batch_size = 64
        test_loader_list = [DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for test_dataset in dataset_test_list]

        predictions = []
        for test_loader, ch_names in zip(test_loader_list, ch_names_list):
            coords = self._get_coords(ch_names)
            preds_all = []
            for x in tqdm(test_loader, desc="BCI Predicting", leave=False):
                x = x.to(self.device)
                cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
                logits = self.model(x, cb)
                preds_all.append(torch.argmax(logits, dim=1).cpu())
            predictions.append(torch.cat(preds_all, dim=0))

        predictions = torch.cat(predictions, dim=0).numpy()
        mapped_pred = np.array([reverse_map_label(idx, task_name) for idx in predictions])
        return mapped_pred
