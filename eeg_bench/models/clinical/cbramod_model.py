"""CBraMod Clinical Model Wrapper for EEG-Bench.

This module provides a wrapper for CBraMod (Criss-Cross Brain Foundation Model)
for clinical EEG classification tasks.

codebase:
https://github.com/wjq-learning/CBraMod
model weights:
https://huggingface.co/weighting666/CBraMod

Reference:
    Wang et al. (2025). CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. ICLR 2025.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from ...utils import wandb_utils
from ...utils.utils import CachedArrayDataset, create_temp_cache_dir, cleanup_temp_cache_dir

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global imports for CBraMod modules (set up dynamically)
CBraMod = None


def _setup_cbramod_imports(cbramod_path: Optional[str] = None):
    """Setup CBraMod imports by adding path to sys.path if needed."""
    global CBraMod

    if cbramod_path and cbramod_path not in sys.path:
        sys.path.insert(0, cbramod_path)
        logger.info(f"Added CBraMod path to sys.path: {cbramod_path}")

    try:
        from models.cbramod import CBraMod as _CBraMod
        CBraMod = _CBraMod
        logger.info("Successfully imported CBraMod modules")
    except ImportError as e:
        logger.error(f"Failed to import CBraMod: {e}")
        raise ImportError(
            "Could not import CBraMod. Please ensure CBraMod path is correct."
        )


class CBraModClinicalWrapper(nn.Module):
    """Wraps CBraMod for clinical classification tasks."""

    def __init__(
        self,
        num_classes: int,
        num_labels_per_chunk: Optional[int] = None,
        n_channels: int = 22,
        patch_size: int = 200,
        num_patches: int = 4,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.is_multilabel_task = num_labels_per_chunk is not None
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Build CBraMod backbone
        self.backbone = CBraMod(
            in_dim=patch_size,
            out_dim=patch_size,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=30,  # Not used for our patching scheme
            n_layer=n_layer,
            nhead=nhead,
        ).to(device)

        # Load pretrained weights if available
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

        # Replace projection head with identity and add custom classifier
        self.backbone.proj_out = nn.Identity()

        # Classifier: use adaptive pooling to handle variable number of patches
        # Output shape from backbone: (batch, channels, patches, d_model)
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b d c s'),  # [B, d_model, channels, patches]
            nn.AdaptiveAvgPool2d((1, 1)),      # [B, d_model, 1, 1]
            nn.Flatten(),                       # [B, d_model]
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes if not self.is_multilabel_task else num_classes * num_labels_per_chunk),
        ).to(device)

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

        self.loss_fn = nn.CrossEntropyLoss()

    def _load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained weights from checkpoint."""
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists():
            logger.warning(f"Pretrained weights not found at {pretrained_path}. Training from scratch.")
            return

        try:
            state_dict = torch.load(str(pretrained_path), map_location=device)
            # Load only backbone weights (not classifier)
            self.backbone.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            logger.error(f"Error loading pretrained weights: {e}")
            raise

    def _freeze_backbone(self):
        """Freeze all backbone parameters except classifier."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info("Froze backbone parameters, keeping classifier trainable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CBraMod.

        Args:
            x: Input signal [B, C, T] or [B, C, num_patches, patch_size]

        Returns:
            logits: Classification logits
        """
        x = x.to(device)

        # If input is 3D [B, C, T], reshape to 4D [B, C, num_patches, patch_size]
        if x.ndim == 3:
            batch_size, n_channels, total_samples = x.shape
            # Calculate how many complete patches we can extract
            num_patches = total_samples // self.patch_size
            # Trim to multiple of patch_size
            trimmed_samples = num_patches * self.patch_size
            x = x[:, :, :trimmed_samples]
            # Reshape to [B, C, num_patches, patch_size]
            x = x.reshape(batch_size, n_channels, num_patches, self.patch_size)

        # Backbone forward
        feats = self.backbone(x)  # (B, C, num_patches, d_model)
        # Classify
        logits = self.classifier(feats)

        if self.is_multilabel_task:
            logits = logits.reshape(x.shape[0], self.num_classes, -1)

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        if x.ndim == 3:
            batch_size, n_channels, total_samples = x.shape
            num_patches = total_samples // self.patch_size
            trimmed_samples = num_patches * self.patch_size
            x = x[:, :, :trimmed_samples]
            x = x.reshape(batch_size, n_channels, num_patches, self.patch_size)
        with torch.no_grad():
            return self.backbone(x)

    def classify_features(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(feats.to(device))
        if self.is_multilabel_task:
            logits = logits.reshape(feats.shape[0], self.num_classes, -1)
        return logits


class CBraModClinicalModel(AbstractModel):
    """CBraMod wrapper for clinical EEG classification tasks."""

    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        chunk_len_s: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        cbramod_path: Optional[str] = None,
        patch_size: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        freeze_backbone: bool = True,
    ):
        """Initialize CBraMod clinical model.

        Args:
            num_classes: Number of output classes
            num_labels_per_chunk: For multilabel tasks
            chunk_len_s: Chunk length in seconds
            pretrained_path: Path to pretrained CBraMod checkpoint (.pth)
            cbramod_path: Path to CBraMod repository for imports
            patch_size: Size of each patch (default 200, i.e., 1 second at 200Hz)
            d_model: Model embedding dimension
            dim_feedforward: Feedforward dimension
            n_layer: Number of transformer layers
            nhead: Number of attention heads
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__("CBraModModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk
        self.chunk_len_s = chunk_len_s if chunk_len_s is not None else (16 if num_labels_per_chunk else None)
        self.freeze_backbone = freeze_backbone

        # Model architecture parameters
        self.patch_size = patch_size
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.n_layer = n_layer
        self.nhead = nhead

        # Setup CBraMod imports
        _setup_cbramod_imports(cbramod_path)

        # Store pretrained path for model initialization
        self.pretrained_path = pretrained_path

        self.model: Optional[CBraModClinicalWrapper] = None

    def _init_model(self, sample: np.ndarray, sfreq: int) -> None:
        """Initialize the CBraMod model based on sample data shape.

        Args:
            sample: Sample data array with shape [C, T]
            sfreq: Sampling frequency
        """
        n_channels = sample.shape[0]

        # CBraMod was pretrained with patch_size=200, in_dim=200, out_dim=200, d_model=200
        # We keep these fixed to match the pretrained checkpoint architecture
        # The number of patches is adjusted based on chunk length and sampling frequency
        patch_size = 200

        # Number of patches per chunk
        if self.chunk_len_s:
            num_patches = int(self.chunk_len_s * sfreq / patch_size)
        else:
            # For full recordings, use 4 patches as default
            num_patches = 4

        self.model = CBraModClinicalWrapper(
            num_classes=self.num_classes,
            num_labels_per_chunk=self.num_labels_per_chunk,
            n_channels=n_channels,
            patch_size=patch_size,
            num_patches=num_patches,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            n_layer=self.n_layer,
            nhead=self.nhead,
            pretrained_path=self.pretrained_path,
            freeze_backbone=self.freeze_backbone,
        ).to(self.device)

        logger.info(f"Initialized CBraMod model with {n_channels} channels, {num_patches} patches of size {patch_size}, d_model={self.d_model}, sfreq={sfreq}Hz")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        epoch: int,
    ) -> tuple[float, float]:
        """Train for one epoch."""
        assert self.model is not None
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        total_acc_samples = 0

        for x, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, yb = x.to(self.device), yb.to(self.device)

            # Handle one-hot encoded labels
            if not self.model.is_multilabel_task and yb.dim() > 1:
                yb = yb.argmax(dim=1)

            optimizer.zero_grad()
            logits = self.model(x)
            loss = self.model.loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            if logits.dim() == 2:
                preds = torch.argmax(logits, dim=1)
                target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                correct += (preds == target).sum().item()
                total_acc_samples += x.size(0)

            del x, yb, logits, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_acc_samples if total_acc_samples > 0 else 0.0
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> tuple[float, float]:
        """Validate for one epoch."""
        assert self.model is not None
        self.model.eval()
        val_loss = 0.0
        val_samples = 0
        val_correct = 0
        val_acc_samples = 0

        for x, yb, _ in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
            x, yb = x.to(self.device), yb.to(self.device)

            if not self.model.is_multilabel_task and yb.dim() > 1:
                yb = yb.argmax(dim=1)

            logits = self.model(x)
            loss = self.model.loss_fn(logits, yb)
            val_loss += loss.item() * x.size(0)
            val_samples += x.size(0)

            if logits.dim() == 2:
                preds = torch.argmax(logits, dim=1)
                target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                val_correct += (preds == target).sum().item()
                val_acc_samples += x.size(0)

            del x, yb, logits, loss
            torch.cuda.empty_cache()

        avg_loss = val_loss / val_samples if val_samples > 0 else 0.0
        accuracy = val_correct / val_acc_samples if val_acc_samples > 0 else 0.0
        return avg_loss, accuracy

    def _fit_linear_probe_cached(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        assert self.model is not None

        cache_dir = create_temp_cache_dir("cbramod_clinical_lp_")
        try:
            sample_x, _, _ = train_loader.dataset[0]
            sample_features = self.model.extract_features(sample_x.unsqueeze(0)).cpu().numpy()
            feature_shape = sample_features.shape[1:]
            label_shape = (self.num_labels_per_chunk,) if self.num_labels_per_chunk is not None else ()

            train_features = np.memmap(
                os.path.join(cache_dir, "train_features.dat"),
                dtype=np.float32,
                mode="w+",
                shape=(len(train_loader.dataset), *feature_shape),
            )
            train_labels = np.memmap(
                os.path.join(cache_dir, "train_labels.dat"),
                dtype=np.int64,
                mode="w+",
                shape=(len(train_loader.dataset),) if not label_shape else (len(train_loader.dataset), *label_shape),
            )

            idx = 0
            self.model.eval()
            for x, yb, _ in tqdm(train_loader, desc="Cache CBraMod clinical train", leave=False):
                x, yb = x.to(self.device), yb.to(self.device)
                if not self.model.is_multilabel_task and yb.dim() > 1:
                    yb = yb.argmax(dim=1)
                feats = self.model.extract_features(x).cpu().numpy()
                labels = yb.cpu().numpy()
                bsz = feats.shape[0]
                train_features[idx:idx + bsz] = feats
                train_labels[idx:idx + bsz] = labels
                idx += bsz

            train_features.flush()
            train_labels.flush()

            val_features = np.memmap(
                os.path.join(cache_dir, "val_features.dat"),
                dtype=np.float32,
                mode="w+",
                shape=(len(val_loader.dataset), *feature_shape),
            )
            val_labels = np.memmap(
                os.path.join(cache_dir, "val_labels.dat"),
                dtype=np.int64,
                mode="w+",
                shape=(len(val_loader.dataset),) if not label_shape else (len(val_loader.dataset), *label_shape),
            )

            idx = 0
            for x, yb, _ in tqdm(val_loader, desc="Cache CBraMod clinical val", leave=False):
                x, yb = x.to(self.device), yb.to(self.device)
                if not self.model.is_multilabel_task and yb.dim() > 1:
                    yb = yb.argmax(dim=1)
                feats = self.model.extract_features(x).cpu().numpy()
                labels = yb.cpu().numpy()
                bsz = feats.shape[0]
                val_features[idx:idx + bsz] = feats
                val_labels[idx:idx + bsz] = labels
                idx += bsz

            val_features.flush()
            val_labels.flush()

            train_feat_loader = DataLoader(
                CachedArrayDataset(train_features, train_labels),
                batch_size=256,
                shuffle=True,
                num_workers=0,
            )
            val_feat_loader = DataLoader(
                CachedArrayDataset(val_features, val_labels),
                batch_size=256,
                shuffle=False,
                num_workers=0,
            )

            max_epochs = 30
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=4e-4, steps_per_epoch=max(1, len(train_feat_loader)), epochs=max_epochs, pct_start=0.2
            )

            patience = 10
            patience_counter = 0
            best_val_loss = float("inf")
            best_model_state = None

            for epoch in range(1, max_epochs + 1):
                self.model.classifier.train()
                total_loss = 0.0
                total_samples = 0
                correct = 0
                total_acc_samples = 0
                for feats, yb in tqdm(train_feat_loader, desc=f"Epoch {epoch}", leave=False):
                    feats, yb = feats.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model.classify_features(feats)
                    loss = self.model.loss_fn(logits, yb)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item() * feats.size(0)
                    total_samples += feats.size(0)
                    if logits.dim() == 2:
                        preds = torch.argmax(logits, dim=1)
                        target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                        correct += (preds == target).sum().item()
                        total_acc_samples += feats.size(0)

                val_loss = 0.0
                val_samples = 0
                val_correct = 0
                val_acc_samples = 0
                self.model.classifier.eval()
                with torch.no_grad():
                    for feats, yb in tqdm(val_feat_loader, desc=f"Val {epoch}", leave=False):
                        feats, yb = feats.to(self.device), yb.to(self.device)
                        logits = self.model.classify_features(feats)
                        loss = self.model.loss_fn(logits, yb)
                        val_loss += loss.item() * feats.size(0)
                        val_samples += feats.size(0)
                        if logits.dim() == 2:
                            preds = torch.argmax(logits, dim=1)
                            target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                            val_correct += (preds == target).sum().item()
                            val_acc_samples += feats.size(0)

                train_loss = total_loss / total_samples if total_samples > 0 else 0.0
                train_acc = correct / total_acc_samples if total_acc_samples > 0 else 0.0
                val_loss_avg = val_loss / val_samples if val_samples > 0 else 0.0
                val_acc = val_correct / val_acc_samples if val_acc_samples > 0 else 0.0

                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                current_lr = scheduler.get_last_lr()[0]
                metrics = {
                    f"{self.name}/train_loss": train_loss,
                    f"{self.name}/train_acc": train_acc,
                    f"{self.name}/val_loss": val_loss_avg,
                    f"{self.name}/val_acc": val_acc,
                    f"{self.name}/lr": current_lr,
                }
                if self.wandb_run:
                    wandb_utils.log(metrics, step=epoch)

                print(
                    f"[Epoch {epoch:02d}/{max_epochs}] "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f} | "
                    f"lr={current_lr:.2e} patience={patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
        finally:
            cleanup_temp_cache_dir(cache_dir)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        """Train the CBraMod model."""
        task_name = meta[0]["task_name"]
        sfreq = meta[0]["sampling_frequency"]

        # Create training dataset
        dataset_train = make_dataset_2(
            X, y, meta, task_name, self.name, self.chunk_len_s,
            is_train=True, use_cache=False
        )

        if len(dataset_train) == 0:
            logger.warning("Dataset empty. Retrying without cache...")
            dataset_train = make_dataset_2(
                X, y, meta, task_name, self.name, self.chunk_len_s,
                is_train=True, use_cache=False
            )

        if len(dataset_train) == 0:
            logger.warning("Dataset empty after retries. Skipping training.")
            return

        # Split into train/val
        val_split = 0.15
        dataset_train, dataset_val = dataset_train.split_train_val(val_split)

        # Initialize model if needed
        sample_data, _, _ = dataset_train[0]
        if self.model is None:
            self._init_model(sample_data, sfreq)

        # Setup loss function with class weights
        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Create data loaders
        batch_size = 64 if self.chunk_len_s else 1
        train_loader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True
        )

        if self.freeze_backbone:
            self._fit_linear_probe_cached(train_loader, val_loader)
            return

        # Setup optimizer and scheduler
        max_epochs = 30
        steps_per_epoch = len(train_loader)
        max_lr = 4e-4

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
            epochs=max_epochs, pct_start=0.2,
        )

        # Early stopping setup
        patience = 10
        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None

        # Training loop
        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, scheduler, epoch)
            val_loss, val_acc = self._validate_epoch(val_loader, epoch)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Log metrics
            current_lr = scheduler.get_last_lr()[0]
            metrics = {
                f"{self.name}/train_loss": train_loss,
                f"{self.name}/train_acc": train_acc,
                f"{self.name}/val_loss": val_loss,
                f"{self.name}/val_acc": val_acc,
                f"{self.name}/lr": current_lr,
            }

            if self.wandb_run:
                wandb_utils.log(metrics, step=epoch)

            print(
                f"[Epoch {epoch:02d}/{max_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e} patience={patience_counter}/{patience}"
            )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with val_loss={best_val_loss:.4f}")

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        """Generate predictions for the input data."""
        assert self.model is not None, "Model must be trained before prediction"

        task_name = meta[0]["task_name"]

        # Create test dataset
        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name, self.chunk_len_s,
            is_train=False, use_cache=False
        )

        if len(dataset_test) == 0:
            return np.array([])

        batch_size = 64 if self.chunk_len_s else 1
        test_loader = DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False, num_workers=0
        )

        self.model.eval()
        predictions = []
        indices = []

        for x, idx, _ in tqdm(test_loader, desc="Predicting"):
            x = x.to(self.device)
            logits = self.model(x)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred.cpu())
            indices.append(idx)

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        indices = torch.cat(indices, dim=0).cpu().numpy()

        # Aggregate predictions if using chunks (majority voting)
        if self.chunk_len_s is not None and not self.model.is_multilabel_task:
            unique_indices = np.unique(indices)
            aggregated_predictions = []
            for idx in unique_indices:
                idx_predictions = predictions[indices == idx]
                most_common = Counter(idx_predictions).most_common(1)[0][0]
                aggregated_predictions.append(most_common)
            predictions = np.array(aggregated_predictions)

        # Map predictions back to original labels
        mapped_pred = np.array([map_label_reverse(pred, task_name) for pred in predictions])
        return mapped_pred
