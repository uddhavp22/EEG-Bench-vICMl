from __future__ import annotations

from typing import List, Dict, Optional
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from ...utils import wandb_utils
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
from transformers import AutoModel
from collections import Counter



class REVEClinicalWrapper(nn.Module):
    """
    Wraps the HuggingFace REVE model with a classification head.
    """
    from torch.backends.cuda import SDPBackend
    try:
        from torch.nn.attention import sdpa_kernel
    except (ImportError, ModuleNotFoundError):
        # Fallback: create a simple context manager that does nothing
        from contextlib import contextmanager
        @contextmanager
        def sdpa_kernel(backends):
            yield

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        num_classes: int,
        num_labels_per_chunk: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.is_multilabel_task = num_labels_per_chunk is not None
        self.num_classes = num_classes

        self.backbone = AutoModel.from_pretrained(
            "brain-bzh/reve-base",
            trust_remote_code=True,
            dtype="auto",
            token='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'
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        if hidden_dim is None:
            config = getattr(self.backbone, "config", None)
            hidden_dim = (
                getattr(config, "hidden_size", None)
                or getattr(config, "hidden_dim", None)
                or getattr(config, "d_model", None)
                or 512
            )



        out_dim = num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1)
    
        self.head = nn.Sequential(torch.nn.Flatten(start_dim=1),torch.nn.Linear(n_channels*hidden_dim,out_dim))


        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, pos):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = self.backbone(x, pos)

            #print(features.shape)
        features = features.mean(dim=2) 
        logits = self.head(features)
        #print(logits.shape)
        if self.is_multilabel_task:
            logits = logits.view(x.shape[0], self.num_classes, -1)
            print(logits.shape)
        return logits


class REVEClinicalModel(AbstractModel):
    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        chunk_len_s: Optional[int] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__("REVEModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk
        if chunk_len_s is None and num_labels_per_chunk:
            self.chunk_len_s = 16
        else:
            self.chunk_len_s = chunk_len_s
        self.freeze_backbone = freeze_backbone

        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", trust_remote_code=True,token='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'
        ).to(self.device)
        self.model: Optional[REVEClinicalWrapper] = None

    def _coords(self, ch_names: List[str]) -> torch.Tensor:
        clean_names = [c.replace("EEG", "").strip() for c in ch_names]
        positions = self.pos_bank(clean_names)
        if isinstance(positions, dict):
            positions = positions.get(
                "positions", positions.get("coords", positions.get("last_hidden_state"))
            )
        if positions.dim() == 3:
            positions = positions.squeeze(0)
        return positions.float().to(self.device)

    def _init_model(self, sample: np.ndarray) -> None:
        n_channels, n_timepoints = sample.shape[0], sample.shape[1]
        self.model = REVEClinicalWrapper(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            num_classes=self.num_classes,
            num_labels_per_chunk=self.num_labels_per_chunk,
            freeze_backbone=self.freeze_backbone,
        ).to(self.device)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        task_name = meta[0]["task_name"]

        dataset_train = make_dataset_2(
            X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=True
        )
        if len(dataset_train) == 0:
            print("[Warning] Dataset empty. Retrying without cache...")
            dataset_train = make_dataset_2(
                X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=True
            )
        if len(dataset_train) == 0:
            print("[Warning] Dataset empty after retries. Skipping training.")
            return

        dataset_train, dataset_val = dataset_train.split_train_val(0.2)
        if len(dataset_train) == 0:
            print("[Warning] Training split is empty. Skipping training.")
            return

        sample_data, _, _ = dataset_train[0]
        print("Input shape",sample_data.shape)
        if self.model is None:
            self._init_model(sample_data)
        
        #print(self.model.backbone.config)

        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        train_loader = DataLoader(
            dataset_train, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            dataset_val, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
        )

        coords_train = self._coords(dataset_train.ch_names)
        coords_val = self._coords(dataset_val.ch_names)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=1e-3)

        num_epochs = 30
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            total_acc_samples = 0

            for x, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
                x, yb = x.to(self.device), yb.to(self.device)
                if not self.model.is_multilabel_task and yb.dim() > 1:
                    yb = yb.argmax(dim=1)
                cb = coords_train.unsqueeze(0).expand(x.size(0), -1, -1)

            

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

            val_loss = 0.0
            val_samples = 0
            val_correct = 0
            val_acc_samples = 0
            self.model.eval()
            with torch.no_grad():
                for x, yb, _ in tqdm(val_loader, desc=f"Val {epoch}"):
                    x, yb = x.to(self.device), yb.to(self.device)
                    if not self.model.is_multilabel_task and yb.dim() > 1:
                        yb = yb.argmax(dim=1)
                    cb = coords_val.unsqueeze(0).expand(x.size(0), -1, -1)

                    logits = self.model(x, cb)
                    loss = self.model.loss_fn(logits, yb)
                    val_loss += loss.item() * x.size(0)
                    val_samples += x.size(0)

                    if logits.dim() == 2:
                        preds = torch.argmax(logits, dim=1)
                        target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                        val_correct += (preds == target).sum().item()
                        val_acc_samples += x.size(0)

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
        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name, self.chunk_len_s, is_train=False, use_cache=True
        )

        if len(dataset_test) == 0:
            return np.array([])

        test_loader = DataLoader(
            dataset_test, batch_size=64 if self.chunk_len_s else 1, shuffle=False, num_workers=0
        )
        coords = self._coords(dataset_test.ch_names)
        self.model.eval()

        predictions = []
        indices = []
        for x, idx, _ in tqdm(test_loader, desc="Predicting"):
            x = x.to(self.device)
            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
            logits = self.model(x, cb)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred.cpu())
            indices.append(idx)

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        indices = torch.cat(indices, dim=0).cpu().numpy()

        if self.chunk_len_s is not None and not self.model.is_multilabel_task:
            unique_indices = np.unique(indices)
            aggregated_predictions = []
            for idx in unique_indices:
                idx_predictions = predictions[indices == idx]
                most_common_prediction = Counter(idx_predictions).most_common(1)[0][0]
                aggregated_predictions.append(most_common_prediction)
            predictions = np.array(aggregated_predictions)

        mapped_pred = np.array([map_label_reverse(pred, task_name) for pred in predictions])
        return mapped_pred
