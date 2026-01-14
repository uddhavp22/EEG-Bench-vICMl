"""LUNA BCI Model Wrapper for EEG-Bench.

This module provides a wrapper for the LUNA (Linear Universal Neural Architecture)
model for BCI (Motor Imagery) EEG classification tasks.

Reference:
    Döner et al. (2025). LUNA: Efficient and Topology-Agnostic Foundation Model
    for EEG Signal Analysis. NeurIPS 2025.
"""

from __future__ import annotations

import logging
import sys
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel
from safetensors.torch import load_file as load_safetensors
from pathlib import Path

from ..abstract_model import AbstractModel
from ...utils import wandb_utils
from .LaBraM.make_dataset import make_dataset_luna
from .LaBraM.utils_2 import n_unique_labels, calc_class_weights

logger = logging.getLogger(__name__)

# Global imports for LUNA modules (set up dynamically)
LUNA = None
CrossAttentionBlock = None
PatchEmbedNetwork = None
RotaryTransformerBlock = None
FrequencyFeatureEmbedder = None
ChannelEmbeddings = None


def _setup_luna_imports(biofoundation_path: Optional[str] = None):
    """Setup BioFoundation imports by adding path to sys.path if needed."""
    global LUNA, CrossAttentionBlock, PatchEmbedNetwork
    global RotaryTransformerBlock, FrequencyFeatureEmbedder, ChannelEmbeddings

    if biofoundation_path and biofoundation_path not in sys.path:
        sys.path.insert(0, biofoundation_path)
        logger.info(f"Added BioFoundation path to sys.path: {biofoundation_path}")

    try:
        # Import LUNA modules from BioFoundation
        from models.LUNA import LUNA as _LUNA
        from models.LUNA import CrossAttentionBlock as _CrossAttentionBlock
        from models.LUNA import PatchEmbedNetwork as _PatchEmbedNetwork
        from models.modules.rope_transformer_encoder_block import RotaryTransformerBlock as _RotaryTransformerBlock
        from models.modules.frequency_embedder import FrequencyFeatureEmbedder as _FrequencyFeatureEmbedder
        from models.modules.channel_embeddings import ChannelEmbeddings as _ChannelEmbeddings

        LUNA = _LUNA
        CrossAttentionBlock = _CrossAttentionBlock
        PatchEmbedNetwork = _PatchEmbedNetwork
        RotaryTransformerBlock = _RotaryTransformerBlock
        FrequencyFeatureEmbedder = _FrequencyFeatureEmbedder
        ChannelEmbeddings = _ChannelEmbeddings

        logger.info("Successfully imported LUNA modules from BioFoundation")
    except ImportError as e:
        logger.error(f"Failed to import LUNA modules: {e}")
        raise ImportError(
            "Could not import LUNA modules. Please ensure BioFoundation path is correct "
            "and all dependencies are installed."
        )


class SimpleDataset(Dataset):
    """Simple wrapper to convert List[np.ndarray] into a Torch Dataset.

    Assumes X is (N, C, T) and y is (N,)
    """
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


class LUNABCIWrapper(nn.Module):
    """Wraps the LUNA model for BCI classification tasks."""

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_classes: int,
        patch_size: int = 40,
        num_queries: int = 4,
        embed_dim: int = 64,
        depth: int = 8,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build LUNA model with classification head
        self.backbone = LUNA(
            patch_size=patch_size,
            num_queries=num_queries,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=nn.LayerNorm,
            drop_path=0.0,
            num_classes=n_classes,  # Classification mode
        ).to(self.device)

        # Load pretrained weights if available
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path)

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained weights from safetensors file."""
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists():
            logger.warning(f"Pretrained weights not found at {pretrained_path}. Training from scratch.")
            return

        try:
            # Load safetensors
            state_dict = load_safetensors(str(pretrained_path))

            # Remove classification head weights if present
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}

            # Load weights with strict=False to allow missing classifier weights
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)

            # Filter out expected missing keys (classifier head)
            missing_keys = [k for k in missing_keys if not k.startswith('classifier.')]

            if missing_keys:
                logger.warning(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

            logger.info(f"Successfully loaded pretrained weights from {pretrained_path}")

        except Exception as e:
            logger.error(f"Error loading pretrained weights: {e}")
            raise

    def _freeze_backbone(self):
        """Freeze all backbone parameters except classification head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith('classifier.'):
                param.requires_grad = False

        # Set backbone to eval mode
        self.backbone.eval()
        # But keep classifier in train mode
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier.train()

        logger.info("Froze backbone parameters, keeping classification head trainable")

    def forward(self, x: torch.Tensor, channel_locations: torch.Tensor) -> torch.Tensor:
        """Forward pass through LUNA model.

        Args:
            x: Input signal [B, C, T]
            channel_locations: Channel 3D coordinates [B, C, 3]

        Returns:
            logits: Classification logits [B, n_classes]
        """
        x = x.to(self.device)
        channel_locations = channel_locations.to(self.device)

        # LUNA expects mask=None for classification
        # Forward returns (x_classified, x_original) for classification mode
        logits, _ = self.backbone(
            x_signal=x,
            mask=None,
            channel_locations=channel_locations,
            channel_names=None  # Not used in classification mode
        )

        return logits


class LUNABCIModel(AbstractModel):
    """LUNA model wrapper for BCI (Motor Imagery) EEG classification tasks."""

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        biofoundation_path: Optional[str] = None,
        patch_size: int = 40,
        num_queries: int = 4,
        embed_dim: int = 64,
        depth: int = 8,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        freeze_backbone: bool = True,
    ):
        """Initialize LUNA BCI model.

        Args:
            pretrained_path: Path to pretrained LUNA checkpoint (.safetensors)
            biofoundation_path: Path to BioFoundation repository for imports
            patch_size: Patch size for LUNA
            num_queries: Number of queries for channel unification
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__("LUNAModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.freeze_backbone = freeze_backbone

        # Model architecture parameters
        self.patch_size = patch_size
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Setup LUNA imports from BioFoundation
        _setup_luna_imports(biofoundation_path)

        # Store pretrained path for model initialization
        self.pretrained_path = pretrained_path

        # Load position bank for channel coordinates (using REVE's position bank)
        try:
            self.pos_bank = AutoModel.from_pretrained(
                "brain-bzh/reve-positions",
                trust_remote_code=True,
                torch_dtype="auto",
            ).to(self.device)
            logger.info("Loaded position bank from HuggingFace Hub")
        except Exception as e:
            logger.error(f"Failed to load position bank: {e}")
            raise

        self.model: Optional[LUNABCIWrapper] = None

    def _get_channel_coords(self, channel_names: List[str]) -> torch.Tensor:
        """Get 3D channel coordinates from position bank.

        Args:
            channel_names: List of channel names

        Returns:
            positions: Tensor of shape [C, 3] with 3D coordinates
        """
        # Get positions from position bank
        raw_positions = self.pos_bank(channel_names)

        # Handle different output formats from position bank
        if isinstance(raw_positions, dict):
            raw_positions = raw_positions.get(
                "positions",
                raw_positions.get("coords", raw_positions.get("last_hidden_state"))
            )

        # Ensure correct shape: [C, 3]
        if raw_positions.dim() == 3:
            raw_positions = raw_positions.squeeze(0)

        return raw_positions.float().to(self.device)

    def _get_collate_fn(self, channel_names: List[str]):
        """Creates the collate function for LUNA that includes position embeddings.

        Args:
            channel_names: List of channel names for this task

        Returns:
            collate function that batches data with position embeddings
        """
        # Get embeddings for the specific channels of this task
        raw_positions = self._get_channel_coords(channel_names)

        def collate(batch, positions):
            # Stack data: [Batch, Channels, Time]
            x_data = torch.stack([x["data"] for x in batch])

            # Repeat positions for the batch: [Batch, Channels, 3]
            batch_positions = positions.repeat(len(batch), 1, 1)

            batch_dict = {
                "sample": x_data,
                "pos": batch_positions
            }

            if "labels" in batch[0]:
                y_label = torch.tensor([x["labels"] for x in batch])
                batch_dict["label"] = y_label.long()

            return batch_dict

        return partial(collate, positions=raw_positions)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        """Train the LUNA model on BCI data.

        Args:
            X: List of data arrays [n_samples, n_channels, n_timepoints]
            y: List of label arrays [n_samples]
            meta: List of metadata dicts containing channel names, etc.
        """
        print("Initializing LUNA BCI Fit...")

        # 1. Get metadata
        meta_data = meta[0]
        task_name = meta_data["task_name"]
        channel_names = meta_data["channel_names"]
        n_classes = n_unique_labels(task_name)

        # 2. Preprocess data using LUNA-specific pipeline (256 Hz, 0.1-75 Hz bandpass)
        print("[LUNA] Applying LUNA-specific preprocessing...")
        datasets = [
            make_dataset_luna(
                X_, y_, task_name,
                m_["sampling_frequency"],
                m_["channel_names"],
                train=True,
                split_size=0.15
            )
            for X_, y_, m_ in zip(X, y, meta)
        ]

        # Get train datasets and channel names
        dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_val_list = [dataset[1] for dataset in datasets]
        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]

        # Get processed channel names from first dataset
        if dataset_train_list:
            channel_names = dataset_train_list[0].ch_names

        # Get shape from processed data
        sample_data = dataset_train_list[0].data
        n_channels = sample_data.shape[1]
        n_timepoints = sample_data.shape[2]

        # 3. Initialize model with processed dimensions
        if self.model is None:
            self.model = LUNABCIWrapper(
                n_channels=n_channels,
                n_timepoints=n_timepoints,
                n_classes=n_classes,
                patch_size=self.patch_size,
                num_queries=self.num_queries,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                pretrained_path=self.pretrained_path,
                freeze_backbone=self.freeze_backbone,
            ).to(self.device)
            logger.info(f"Initialized LUNA BCI model with {n_channels} channels and {n_timepoints} timepoints (256 Hz)")

        # 4. Prepare DataLoaders from preprocessed datasets
        X_all = np.concatenate([d.data for d in dataset_train_list], axis=0)
        y_all = np.concatenate([d.labels for d in dataset_train_list], axis=0)
        # Convert one-hot back to class indices
        if y_all.ndim > 1:
            y_all = np.argmax(y_all, axis=1)

        train_dataset = SimpleDataset(X_all, y_all)
        collate_fn = self._get_collate_fn(channel_names)

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        # 4. Optimizer
        # Only optimize trainable parameters (classifier head if backbone is frozen)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # 5. Training loop
        self.model.train()
        n_epochs = 10

        print(f"Starting training for {n_epochs} epochs on {self.device}...")

        for epoch in range(n_epochs):
            total_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in pbar:
                data = batch["sample"].to(self.device)
                pos = batch["pos"].to(self.device)
                target = batch["label"].to(self.device)

                optimizer.zero_grad()

                # LUNA forward pass requires (data, pos)
                output = self.model(data, pos)

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

        print("Training complete!")

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        """Generate predictions for the input BCI data.

        Args:
            X: List of data arrays [n_samples, n_channels, n_timepoints]
            meta: List of metadata dicts

        Returns:
            predictions: Array of predicted labels
        """
        assert self.model is not None, "Model must be trained before prediction"

        print("Generating predictions...")

        # Get metadata
        meta_data = meta[0]
        channel_names = meta_data["channel_names"]

        # Prepare data
        X_all = np.concatenate(X, axis=0)
        test_dataset = SimpleDataset(X_all, y=None)
        collate_fn = self._get_collate_fn(channel_names)

        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Predict
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                data = batch["sample"].to(self.device)
                pos = batch["pos"].to(self.device)

                output = self.model(data, pos)
                _, predicted = output.max(1)
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)
