"""CBraMod BCI Model Wrapper for EEG-Bench.

This module provides a wrapper for CBraMod (Criss-Cross Brain Foundation Model)
for BCI (Motor Imagery) EEG classification tasks.

Reference:
    Wang et al. (2025). CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding. ICLR 2025.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from ..abstract_model import AbstractModel
from ...utils import wandb_utils
from ...utils.utils import create_temp_cache_dir, cleanup_temp_cache_dir
from .LaBraM.make_dataset import make_dataset_cbramod
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
        num_patches: int = 4,
        d_model: int = 200,
        dim_feedforward: int = 800,
        n_layer: int = 12,
        nhead: int = 8,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # BCI classifier: similar to the quick_example.py pattern
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(n_channels * num_patches * d_model, num_patches * d_model),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(num_patches * d_model, d_model),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_classes),
        ).to(self.device)

        # Freeze backbone if requested
        if freeze_backbone:
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
        logits = self.classifier(feats)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x.to(self.device))

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
        """
        super().__init__("CBraModModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self.model: Optional[CBraModBCIWrapper] = None

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

    def _fit_linear_probe_cached(self, X_patched: np.ndarray, y_all: np.ndarray, n_epochs: int) -> None:
        assert self.model is not None

        cache_dir = create_temp_cache_dir("cbramod_bci_lp_")
        try:
            total_samples = X_patched.shape[0]
            sample_batch = torch.from_numpy(X_patched[:1]).float().to(self.device)
            sample_features = self.model.extract_features(sample_batch).cpu().numpy()
            feature_shape = sample_features.shape[1:]

            features = np.memmap(
                os.path.join(cache_dir, "train_features.dat"),
                dtype=np.float32,
                mode="w+",
                shape=(total_samples, *feature_shape),
            )
            labels = np.memmap(
                os.path.join(cache_dir, "train_labels.dat"),
                dtype=np.int64,
                mode="w+",
                shape=(total_samples,),
            )

            data_loader = DataLoader(SimpleDataset(X_patched, y_all), batch_size=64, shuffle=False, num_workers=0)

            idx = 0
            self.model.eval()
            for batch in tqdm(data_loader, desc="Cache CBraMod BCI embeddings", leave=False):
                data = batch["data"].to(self.device)
                target = batch["labels"].cpu().numpy()
                feats = self.model.extract_features(data).cpu().numpy()
                bsz = feats.shape[0]
                features[idx:idx + bsz] = feats
                labels[idx:idx + bsz] = target
                idx += bsz

            features.flush()
            labels.flush()

            feat_loader = DataLoader(
                TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)),
                batch_size=256,
                shuffle=True,
                num_workers=0,
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

        # 2. Preprocess data using CBraMod-specific pipeline (200 Hz, 0.5-40 Hz bandpass, z-score)
        logger.info("[CBraMod] Applying CBraMod-specific preprocessing...")
        datasets = [
            make_dataset_cbramod(
                X_, y_, task_name,
                m_["sampling_frequency"],
                m_["channel_names"],
                train=True,
                split_size=0.15
            )
            for X_, y_, m_ in zip(X, y, meta)
        ]

        # Get train datasets
        dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]

        # 3. Prepare preprocessed data
        X_all = np.concatenate([d.data for d in dataset_train_list], axis=0)
        y_all = np.concatenate([d.labels for d in dataset_train_list], axis=0)
        # Convert one-hot back to class indices
        if y_all.ndim > 1:
            y_all = np.argmax(y_all, axis=1)

        # Get sampling frequency from preprocessed data (always 200 Hz for CBraMod)
        sfreq = 200

        # 4. Reshape to patches (now with consistent 200 Hz data)
        X_patched = self._reshape_to_patches(X_all, sfreq)
        _, n_channels, num_patches, patch_size = X_patched.shape

        # 5. Initialize model if needed
        if self.model is None:
            self.model = CBraModBCIWrapper(
                n_channels=n_channels,
                n_classes=n_classes,
                patch_size=patch_size,
                num_patches=num_patches,
                d_model=self.d_model,
                dim_feedforward=self.dim_feedforward,
                n_layer=self.n_layer,
                nhead=self.nhead,
                pretrained_path=self.pretrained_path,
                freeze_backbone=self.freeze_backbone,
            ).to(self.device)
            logger.info(f"[CBraMod] Initialized with {n_channels} channels, {num_patches} patches of size {patch_size} (200 Hz)")

        # Training loop
        n_epochs = 10

        logger.info(f"Starting training for {n_epochs} epochs...")

        if self.freeze_backbone:
            self._fit_linear_probe_cached(X_patched, y_all, n_epochs)
            logger.info("Training complete!")
            return

        train_dataset = SimpleDataset(X_patched, y_all)
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=0
        )

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for epoch in range(n_epochs):
            total_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in pbar:
                data = batch["data"].to(self.device)
                target = batch["label"].to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

            epoch_loss = total_loss / len(train_loader)
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
        sfreq = meta_data["sampling_frequency"]

        # Concatenate and reshape to patches
        X_all = np.concatenate(X, axis=0)
        X_patched = self._reshape_to_patches(X_all, sfreq)

        test_dataset = SimpleDataset(X_patched, y=None)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

        # Predict
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                data = batch["data"].to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)
