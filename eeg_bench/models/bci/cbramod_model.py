"""CBraMod BCI Model Wrapper for EEG-Bench.

This module provides a wrapper for CBraMod (Criss-Cross Brain Foundation Model)
for BCI (Motor Imagery) EEG classification tasks.

Reference:
    Wang et al. (2025). CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. ICLR 2025.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..abstract_model import AbstractModel
from ...utils import wandb_utils
from ...utils.utils import CachedArrayDataset, create_temp_cache_dir, cleanup_temp_cache_dir
from .LaBraM.make_dataset import make_dataset_cbramod, _ordered_target_channels
from .LaBraM.utils_2 import n_unique_labels, calc_class_weights

logger = logging.getLogger(__name__)

# Global imports for CBraMod modules
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
        raise ImportError("Could not import CBraMod. Please ensure CBraMod path is correct.")


class SimpleDataset(Dataset):
    """Simple wrapper to convert List[np.ndarray] into a Torch Dataset."""

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return {"data": self.X[idx], "labels": self.y[idx]}
        return {"data": self.X[idx]}


class CBraModBCIWrapper(nn.Module):
    """Wraps CBraMod for BCI classification tasks."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        patch_size: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = True,
        linear_probe: bool = False,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_probe = linear_probe
        self.feature_dim = d_model

        # Build CBraMod backbone
        self.backbone = CBraMod(
            in_dim=patch_size,
            out_dim=patch_size,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            seq_len=30,
            n_layer=n_layer,
            nhead=nhead,
        ).to(self.device)

        # Load pretrained weights if available
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

        # Replace projection head with identity and add custom classifier
        self.backbone.proj_out = nn.Identity()

        if linear_probe:
            # Linear probe: single linear layer on pooled features (B, d_model)
            self.classifier = nn.Linear(d_model, n_classes).to(self.device)
        else:
            # MLP classifier: mean-pool over patches so num_patches can vary across datasets
            self.classifier = nn.Sequential(
                nn.Linear(n_channels * d_model, d_model),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, n_classes),
            ).to(self.device)

        # Freeze backbone if requested
        if freeze_backbone or linear_probe:
            self._freeze_backbone()

    def _load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained weights from checkpoint."""
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists():
            logger.warning(f"Pretrained weights not found at {pretrained_path}. Training from scratch.")
            return

        try:
            state_dict = torch.load(str(pretrained_path), map_location=self.device)
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
            x: Input signal [B, C, num_patches, patch_size]

        Returns:
            logits: Classification logits [B, n_classes]
        """
        x = x.to(self.device)
        feats = self.backbone(x)  # (B, C, num_patches, d_model)
        if self.linear_probe:
            feats = feats.mean(dim=2).mean(dim=1)  # (B, d_model)
        else:
            feats = feats.mean(dim=2)  # (B, C, d_model) — pool over patches
            feats = feats.reshape(feats.shape[0], -1)  # (B, C * d_model)
        logits = self.classifier(feats)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x.to(self.device))  # (B, C, num_patches, d_model)
            if self.linear_probe:
                return feats.mean(dim=2).mean(dim=1)  # (B, d_model)
            else:
                feats = feats.mean(dim=2)  # (B, C, d_model) — pool over patches
                return feats.reshape(feats.shape[0], -1)  # (B, C * d_model)

    def classify_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(feats.to(self.device))


class CBraModBCIModel(AbstractModel):
    """CBraMod wrapper for BCI (Motor Imagery) EEG classification tasks."""

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        cbramod_path: Optional[str] = None,
        patch_size: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        freeze_backbone: bool = True,
        linear_probe: bool = True,
    ):
        """Initialize CBraMod BCI model.

        Args:
            pretrained_path: Path to pretrained CBraMod checkpoint (.pth)
            cbramod_path: Path to CBraMod repository for imports
            patch_size: Size of each patch (default 200 samples)
            d_model: Model embedding dimension
            dim_feedforward: Feedforward dimension
            n_layer: Number of transformer layers
            nhead: Number of attention heads
            freeze_backbone: Whether to freeze backbone weights
            linear_probe: Whether to use a linear probe (single nn.Linear) instead of MLP
        """
        super().__init__("CBraModModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_backbone = freeze_backbone
        self.linear_probe = linear_probe

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

        self.model: Optional[CBraModBCIWrapper] = None
        self.label_encoder: Optional[LabelEncoder] = None

    def _reshape_to_patches(self, X: np.ndarray, sfreq: int) -> np.ndarray:
        """Reshape continuous EEG data into patches.

        Args:
            X: Input data [N, C, T]
            sfreq: Sampling frequency

        Returns:
            Patched data [N, C, num_patches, patch_size]
        """
        N, C, T = X.shape

        # Calculate patch size based on sampling frequency
        # Aim for ~1 second patches
        if sfreq == 250:
            patch_size = 250
        elif sfreq == 200:
            patch_size = 200
        else:
            patch_size = self.patch_size

        # Calculate number of patches
        num_patches = T // patch_size

        if num_patches == 0:
            logger.warning(f"Signal too short ({T} samples). Padding to create at least one patch.")
            # Pad to create at least one patch
            pad_size = patch_size - T
            X = np.pad(X, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
            T = patch_size
            num_patches = 1

        # Truncate to fit exact patches
        truncated_T = num_patches * patch_size
        X = X[:, :, :truncated_T]

        # Reshape to patches: [N, C, num_patches, patch_size]
        X_patched = X.reshape(N, C, num_patches, patch_size)

        self.patch_size = patch_size
        self.num_patches = num_patches

        return X_patched

    def _fit_linear_probe_on_features(self, all_feats: np.ndarray, all_labels: np.ndarray, n_epochs: int) -> None:
        """Train only the classifier head on pre-extracted features."""
        assert self.model is not None
        cache_dir = create_temp_cache_dir("cbramod_bci_lp_")
        try:
            features = np.memmap(
                os.path.join(cache_dir, "train_features.dat"),
                dtype=np.float32, mode="w+", shape=all_feats.shape,
            )
            labels = np.memmap(
                os.path.join(cache_dir, "train_labels.dat"),
                dtype=np.int64, mode="w+", shape=all_labels.shape,
            )
            features[:] = all_feats
            labels[:] = all_labels
            features.flush()
            labels.flush()

            feat_loader = DataLoader(
                CachedArrayDataset(features, labels),
                batch_size=256, shuffle=True, num_workers=0,
            )
            optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(n_epochs):
                total_loss = 0.0
                correct = 0
                total = 0
                self.model.classifier.train()
                for feats, target in tqdm(feat_loader, desc=f"Epoch {epoch+1}", leave=False):
                    feats = feats.to(self.device)
                    target = target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model.classify_features(feats)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * target.size(0)
                    correct += (output.argmax(dim=1) == target).sum().item()
                    total += target.size(0)

                epoch_loss = total_loss / total if total else 0.0
                epoch_acc = 100.0 * correct / total if total else 0.0
                metrics = {
                    f"{self.name}/train_loss": epoch_loss,
                    f"{self.name}/train_acc": epoch_acc / 100.0,
                }
                if self.wandb_run:
                    wandb_utils.log(metrics, step=epoch + 1)
                print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        finally:
            cleanup_temp_cache_dir(cache_dir)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        """Train the CBraMod model on BCI data."""
        logger.info("Initializing CBraMod BCI Fit...")

        # 1. Get metadata
        meta_data = meta[0]
        task_name = meta_data["task_name"]
        n_classes = n_unique_labels(task_name)

        # 2. Preprocess data using CBraMod-specific pipeline (200 Hz, 0.3-75 Hz bandpass, z-score)
        logger.info("[CBraMod] Applying CBraMod-specific preprocessing...")
        # Compute common channels across all datasets so dimensions match
        channel_sets = [set(ch.upper() for ch in m_["channel_names"]) for m_ in meta]
        common_channels = channel_sets[0]
        for cs in channel_sets[1:]:
            common_channels = common_channels & cs
        target_channels = _ordered_target_channels(list(common_channels))
        self._target_channels = target_channels  # Store for predict()
        logger.info(f"[CBraMod] Using {len(target_channels)} common channels across {len(meta)} datasets")

        datasets = [
            make_dataset_cbramod(
                X_, y_, task_name,
                m_["sampling_frequency"],
                m_["channel_names"],
                target_channels=target_channels,
                train=True,
                split_size=0.15
            )
            for X_, y_, m_ in zip(X, y, meta)
        ]

        # Get train datasets
        dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]

        # 3. Patch each dataset separately (they may have different time lengths)
        sfreq = 200  # CBraMod always resamples to 200 Hz
        patched_list = []
        labels_list = []
        for d in dataset_train_list:
            patched = self._reshape_to_patches(d.data, sfreq)
            patched_list.append(patched)
            lab = d.labels
            if lab.ndim > 1:
                lab = np.argmax(lab, axis=1)
            labels_list.append(lab)

        # Check if we need label encoding (string labels)
        if not np.issubdtype(labels_list[0].dtype, np.number):
            self.label_encoder = LabelEncoder()
            all_labels = np.concatenate(labels_list)
            self.label_encoder.fit(all_labels)
            labels_list = [self.label_encoder.transform(lab) for lab in labels_list]
        else:
            self.label_encoder = None

        _, n_channels, _, patch_size = patched_list[0].shape

        # 4. Initialize model if needed
        if self.model is None:
            self.model = CBraModBCIWrapper(
                n_channels=n_channels,
                n_classes=n_classes,
                patch_size=patch_size,
                d_model=self.d_model,
                dim_feedforward=self.dim_feedforward,
                n_layer=self.n_layer,
                nhead=self.nhead,
                pretrained_path=self.pretrained_path,
                freeze_backbone=self.freeze_backbone,
                linear_probe=self.linear_probe,
            ).to(self.device)
            logger.info(f"[CBraMod] Initialized with {n_channels} channels, pool over patches, patch_size={patch_size} (200 Hz)")

        # Training loop with per-dataset loaders (datasets may have different num_patches)
        n_epochs = 10
        logger.info(f"Starting training for {n_epochs} epochs...")

        if self.freeze_backbone:
            # For linear probe, cache features per-dataset then combine
            feat_list = []
            lab_list = []
            self.model.eval()
            with torch.no_grad():
                for X_p, y_p in zip(patched_list, labels_list):
                    loader = DataLoader(SimpleDataset(X_p, y_p), batch_size=64, shuffle=False, num_workers=0)
                    for batch in loader:
                        feats = self.model.extract_features(batch["data"].to(self.device))
                        feat_list.append(feats.cpu().numpy())
                        lab_list.append(batch["labels"].numpy())
            all_feats = np.concatenate(feat_list, axis=0)
            all_labels = np.concatenate(lab_list, axis=0)
            self._fit_linear_probe_on_features(all_feats, all_labels, n_epochs)
            logger.info("Training complete!")
            return

        train_loaders = [
            DataLoader(SimpleDataset(X_p, y_p), batch_size=64, shuffle=True, num_workers=0)
            for X_p, y_p in zip(patched_list, labels_list)
        ]

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        self.model.backbone.eval()

        for epoch in range(n_epochs):
            total_loss = 0
            correct = 0
            total = 0
            num_batches = 0

            loader_order = list(range(len(train_loaders)))
            random.shuffle(loader_order)

            for li in loader_order:
                pbar = tqdm(train_loaders[li], desc=f"Epoch {epoch+1} DS{li}", leave=False)
                for batch in pbar:
                    data = batch["data"].to(self.device)
                    target = batch["labels"].to(self.device)

                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })

            epoch_loss = total_loss / num_batches
            epoch_acc = 100. * correct / total

            # Log metrics
            metrics = {
                f"{self.name}/train_loss": epoch_loss,
                f"{self.name}/train_acc": epoch_acc / 100.0,
            }

            if self.wandb_run:
                wandb_utils.log(metrics, step=epoch + 1)

            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

        logger.info("Training complete!")

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        """Generate predictions for the input BCI data."""
        assert self.model is not None, "Model must be trained before prediction"

        logger.info("Generating predictions...")

        # Get metadata
        meta_data = meta[0]
        task_name = meta_data["task_name"]

        # Preprocess test data with same pipeline as training
        logger.info("[CBraMod] Preprocessing test data...")
        # Use the same channels that were used during training
        target_channels = self._target_channels

        datasets = [
            make_dataset_cbramod(
                X_, None, task_name,
                m_["sampling_frequency"],
                m_["channel_names"],
                target_channels=target_channels,
                train=False,
            )
            for X_, m_ in zip(X, meta)
        ]
        dataset_list = [d for d in datasets if len(d) > 0]

        # Patch each dataset separately (different time lengths), predict per-dataset
        sfreq = 200  # CBraMod always resamples to 200 Hz
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for d in dataset_list:
                X_patched = self._reshape_to_patches(d.data, sfreq)
                test_loader = DataLoader(
                    SimpleDataset(X_patched, y=None), batch_size=64, shuffle=False, num_workers=0
                )
                for batch in tqdm(test_loader, desc="Predicting", leave=False):
                    data = batch["data"].to(self.device)
                    output = self.model(data)
                    _, predicted = output.max(1)
                    predictions.extend(predicted.cpu().numpy())

        predictions = np.array(predictions)
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        return predictions
