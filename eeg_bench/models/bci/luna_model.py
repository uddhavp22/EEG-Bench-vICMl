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
import os
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from tqdm import tqdm
from transformers import AutoModel
from safetensors.torch import load_file as load_safetensors
from pathlib import Path
from joblib import Memory

from ..abstract_model import AbstractModel
from ...utils import wandb_utils
from ...utils.utils import configure_torch_backend_for_speed, create_temp_cache_dir, cleanup_temp_cache_dir
from ...config import get_config_value
from .LaBraM.make_dataset import make_dataset_luna, standard_1020
from .LaBraM.utils_2 import n_unique_labels, calc_class_weights, reverse_map_label

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
        linear_probe: bool = False,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_probe = linear_probe
        self.feature_dim = embed_dim * num_queries

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

        self.linear_probe_head = None
        if linear_probe:
            self.linear_probe_head = nn.Linear(self.feature_dim, n_classes).to(self.device)
            self._freeze_backbone(keep_classifier=False)
        elif freeze_backbone:
            self._freeze_backbone(keep_classifier=True)

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

    def _freeze_backbone(self, keep_classifier: bool = True):
        """Freeze backbone parameters, optionally keeping classifier trainable."""
        for name, param in self.backbone.named_parameters():
            if keep_classifier and name.startswith('classifier.'):
                continue
            param.requires_grad = False

        self.backbone.eval()
        if keep_classifier and hasattr(self.backbone, 'classifier'):
            self.backbone.classifier.train()

        if keep_classifier:
            logger.info("Froze backbone parameters, keeping classification head trainable")
        else:
            logger.info("Froze backbone parameters for linear probe")

    def _extract_features(self, x: torch.Tensor, channel_locations: torch.Tensor) -> torch.Tensor:
        """Run backbone up to latent tokens and return pooled features."""
        with torch.no_grad():
            B = x.shape[0]
            x_tokens, _ = self.backbone.prepare_tokens(x, channel_locations, mask=None)
            x_tokens, _ = self.backbone.cross_attn(x_tokens)
            num_patches = x_tokens.shape[0] // B
            x_tokens = x_tokens.reshape(B, num_patches, -1)
            for blk in self.backbone.blocks:
                x_tokens = blk(x_tokens)
            x_latent = self.backbone.norm(x_tokens)
            return x_latent.mean(dim=1)

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

        if self.linear_probe:
            features = self._extract_features(x, channel_locations)
            logits = self.linear_probe_head(features)
        else:
            logits, _ = self.backbone(
                x_signal=x,
                mask=None,
                channel_locations=channel_locations,
                channel_names=None
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
        linear_probe: bool = False,
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
        self.linear_probe = linear_probe
        self.freeze_backbone = freeze_backbone or linear_probe
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

        # Model architecture parameters
        self.patch_size = patch_size
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        biofoundation_path = '/raid/spanchavati/BioFoundation/'
        _setup_luna_imports(biofoundation_path=biofoundation_path)

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
        self.channel_names: List[str] = []
        self.target_timepoints: Optional[int] = None

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

        # Important: keep positions on CPU. This collate_fn runs inside
        # DataLoader worker processes, and touching CUDA there can trigger
        # "CUDA error: initialization error".
        return raw_positions.detach().float().cpu()

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

    def _derive_target_layout(
        self,
        datasets: List[Dataset],
        fallback_channels: List[str]
    ) -> tuple[List[str], int]:
        """Infer canonical channel ordering and timepoints shared across datasets."""
        observed_channels: List[str] = []
        max_timepoints = 0

        for dataset in datasets:
            data = getattr(dataset, "data", None)
            if data is None or data.size == 0:
                continue

            max_timepoints = max(max_timepoints, data.shape[2])
            for ch in getattr(dataset, "ch_names", []):
                ch_name = ch.upper()
                if ch_name not in observed_channels:
                    observed_channels.append(ch_name)

        if not observed_channels:
            observed_channels = fallback_channels.copy()

        if not observed_channels:
            raise ValueError("Unable to determine channel ordering for LUNA BCI.")

        standard_order = [ch.upper() for ch in standard_1020]
        ordered_channels = [ch for ch in standard_order if ch in observed_channels]
        for ch in observed_channels:
            if ch not in ordered_channels:
                ordered_channels.append(ch)

        if max_timepoints == 0:
            raise ValueError("Unable to determine timepoints for LUNA BCI.")

        if max_timepoints % self.patch_size != 0:
            max_timepoints = ((max_timepoints // self.patch_size) + 1) * self.patch_size

        return ordered_channels, max_timepoints

    def _align_dataset_shape(
        self,
        dataset: Dataset,
        target_channels: List[str],
        target_timepoints: int
    ) -> None:
        """Align channel/time axes of a dataset to the desired layout."""
        if dataset is None or len(dataset) == 0:
            return

        data = dataset.data
        if data.size == 0:
            dataset.ch_names = target_channels
            dataset.data = np.zeros((0, len(target_channels), target_timepoints), dtype=np.float32)
            return

        n_trials = data.shape[0]
        aligned = np.zeros((n_trials, len(target_channels), target_timepoints), dtype=data.dtype)
        channel_lookup = {ch.upper(): idx for idx, ch in enumerate(dataset.ch_names)}

        for target_idx, ch in enumerate(target_channels):
            src_idx = channel_lookup.get(ch.upper())
            if src_idx is None:
                continue

            copy_len = min(data.shape[2], target_timepoints)
            aligned[:, target_idx, :copy_len] = data[:, src_idx, :copy_len]

        dataset.data = aligned
        dataset.ch_names = target_channels

    def _concat_datasets(
        self,
        datasets: List[Dataset],
        require_labels: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Concatenate multiple LaBraM datasets into numpy arrays."""
        if not datasets:
            raise ValueError("No datasets available to combine.")

        X = np.concatenate([d.data for d in datasets], axis=0)
        y = None

        if require_labels and datasets[0].labels is not None:
            y = np.concatenate([d.labels for d in datasets], axis=0)
            if y.ndim > 1:
                y = np.argmax(y, axis=1)

        return X, y

    def _fit_linear_probe_cached(
        self,
        train_loaders: List[DataLoader],
        val_loaders: List[DataLoader],
        criterion: nn.Module,
        n_epochs: int,
    ) -> None:
        assert self.model is not None and self.model.linear_probe_head is not None

        cache_dir = create_temp_cache_dir("luna_bci_lp_")
        try:
            feature_dim = self.model.feature_dim
            train_datasets = []
            val_datasets = []

            self.model.eval()
            for i, loader in enumerate(train_loaders):
                train_count = len(loader.dataset)
                train_features_path = os.path.join(cache_dir, f"train_features_{i}.dat")
                train_labels_path = os.path.join(cache_dir, f"train_labels_{i}.dat")
                train_features = np.memmap(
                    train_features_path, dtype=np.float32, mode="w+", shape=(train_count, feature_dim)
                )
                train_labels = np.memmap(
                    train_labels_path, dtype=np.int64, mode="w+", shape=(train_count,)
                )

                idx = 0
                for batch in tqdm(loader, desc=f"Cache train embeddings {i+1}/{len(train_loaders)}", leave=False):
                    data = batch["sample"].to(self.device)
                    pos = batch["pos"].to(self.device)
                    labels = batch["label"].cpu().numpy()

                    feats = self.model._extract_features(data, pos).cpu().numpy()
                    bsz = feats.shape[0]
                    train_features[idx:idx + bsz] = feats
                    train_labels[idx:idx + bsz] = labels
                    idx += bsz

                train_features.flush()
                train_labels.flush()
                train_datasets.append(
                    TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
                )

            for i, loader in enumerate(val_loaders):
                val_count = len(loader.dataset)
                val_features_path = os.path.join(cache_dir, f"val_features_{i}.dat")
                val_labels_path = os.path.join(cache_dir, f"val_labels_{i}.dat")
                val_features = np.memmap(
                    val_features_path, dtype=np.float32, mode="w+", shape=(val_count, feature_dim)
                )
                val_labels = np.memmap(
                    val_labels_path, dtype=np.int64, mode="w+", shape=(val_count,)
                )

                idx = 0
                for batch in tqdm(loader, desc=f"Cache val embeddings {i+1}/{len(val_loaders)}", leave=False):
                    data = batch["sample"].to(self.device)
                    pos = batch["pos"].to(self.device)
                    labels = batch["label"].cpu().numpy()

                    feats = self.model._extract_features(data, pos).cpu().numpy()
                    bsz = feats.shape[0]
                    val_features[idx:idx + bsz] = feats
                    val_labels[idx:idx + bsz] = labels
                    idx += bsz

                val_features.flush()
                val_labels.flush()
                val_datasets.append(
                    TensorDataset(torch.from_numpy(val_features), torch.from_numpy(val_labels))
                )

            train_feat_loader = DataLoader(
                ConcatDataset(train_datasets),
                batch_size=256,
                shuffle=True,
                num_workers=0
            )

            val_feat_loader = None
            if val_datasets:
                val_feat_loader = DataLoader(
                    ConcatDataset(val_datasets),
                    batch_size=256,
                    shuffle=False,
                    num_workers=0
                )

            optimizer = torch.optim.AdamW(self.model.linear_probe_head.parameters(), lr=1e-3)

            print(f"Starting linear probe training for {n_epochs} epochs on {self.device}...")

            for epoch in range(n_epochs):
                self.model.linear_probe_head.train()
                total_loss = 0.0
                total_samples = 0
                correct = 0

                pbar = tqdm(train_feat_loader, desc=f"Epoch {epoch+1}", leave=False)
                for feats, target in pbar:
                    feats = feats.to(self.device)
                    target = target.to(self.device)

                    optimizer.zero_grad()
                    output = self.model.linear_probe_head(feats)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    batch_size = target.size(0)
                    total_loss += loss.item() * batch_size
                    _, predicted = output.max(1)
                    total_samples += batch_size
                    correct += predicted.eq(target).sum().item()

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total_samples if total_samples > 0 else 0.0:.2f}%'
                    })

                epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
                epoch_acc = 100. * correct / total_samples if total_samples > 0 else 0.0

                val_loss = None
                val_acc = None
                if val_feat_loader is not None:
                    self.model.linear_probe_head.eval()
                    v_loss = 0.0
                    v_samples = 0
                    v_correct = 0
                    with torch.no_grad():
                        for feats, target in val_feat_loader:
                            feats = feats.to(self.device)
                            target = target.to(self.device)
                            output = self.model.linear_probe_head(feats)
                            loss = criterion(output, target)
                            batch_size = target.size(0)
                            v_loss += loss.item() * batch_size
                            v_samples += batch_size
                            _, predicted = output.max(1)
                            v_correct += predicted.eq(target).sum().item()
                    val_loss = v_loss / v_samples if v_samples > 0 else 0.0
                    val_acc = v_correct / v_samples if v_samples > 0 else 0.0

                metrics = {
                    f"{self.name}/train_loss": epoch_loss,
                    f"{self.name}/train_acc": epoch_acc / 100.0,
                }
                if val_loss is not None and val_acc is not None:
                    metrics[f"{self.name}/val_loss"] = val_loss
                    metrics[f"{self.name}/val_acc"] = val_acc

                if self.wandb_run:
                    wandb_utils.log(metrics, step=epoch + 1)

                if val_loss is not None and val_acc is not None:
                    print(
                        f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}, "
                        f"Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%"
                    )
                else:
                    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        finally:
            cleanup_temp_cache_dir(cache_dir)

    def _evaluate_loader(self, data_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        """Evaluate the model on a dataloader and return loss/accuracy."""
        assert self.model is not None

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Val", leave=False):
                data = batch["sample"].to(self.device)
                pos = batch["pos"].to(self.device)
                target = batch["label"].to(self.device)

                output = self.model(data, pos)
                loss = criterion(output, target)

                batch_size = target.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        return avg_loss, accuracy

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
        n_classes = n_unique_labels(task_name)
        meta_channel_names = [ch.upper() for ch in meta_data.get("channel_names", [])]

        # 2. Preprocess data using LUNA-specific pipeline (256 Hz, 0.1-75 Hz bandpass)
        print("[LUNA] Applying LUNA-specific preprocessing...")
        cached_make_dataset = self.cache.cache(make_dataset_luna)
        datasets = [
            cached_make_dataset(
                X_, y_, task_name,
                m_["sampling_frequency"],
                m_["channel_names"],
                train=True,
                split_size=0.15,
                patch_size=self.patch_size
            )
            for X_, y_, m_ in zip(X, y, meta)
        ]

        # Separate train/val splits and drop empty datasets
        dataset_train_list = [dataset[0] for dataset in datasets if len(dataset[0]) > 0]
        dataset_val_list = [dataset[1] for dataset in datasets if len(dataset[1]) > 0]

        if not dataset_train_list:
            logger.warning("No training samples available after preprocessing. Skipping fit.")
            return

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
                linear_probe=self.linear_probe,
            ).to(self.device)
            logger.info(f"Initialized LUNA BCI model with {n_channels} channels and {n_timepoints} timepoints (256 Hz)")

        # 4. Prepare DataLoaders from preprocessed datasets
        configure_torch_backend_for_speed()

        num_workers = 2
        loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

        train_loader_list = []
        for dataset in dataset_train_list:
            collate_fn = self._get_collate_fn(dataset.ch_names)
            X_train = dataset.data
            y_train = dataset.labels
            if y_train is None:
                raise ValueError("Training labels are missing after preprocessing.")
            if y_train.ndim > 1:
                y_train = np.argmax(y_train, axis=1)
            train_dataset = SimpleDataset(X_train, y_train)
            train_loader_list.append(DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True,
                collate_fn=collate_fn,
                **loader_kwargs
            ))

        val_loader_list = []
        for dataset in dataset_val_list:
            collate_fn = self._get_collate_fn(dataset.ch_names)
            X_val = dataset.data
            y_val = dataset.labels
            if y_val is None or len(y_val) == 0:
                continue
            if y_val.ndim > 1:
                y_val = np.argmax(y_val, axis=1)
            val_dataset = SimpleDataset(X_val, y_val)
            val_loader_list.append(DataLoader(
                val_dataset,
                batch_size=64,
                shuffle=False,
                collate_fn=collate_fn,
                **loader_kwargs
            ))

        # 5. Optimizer and loss
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        class_weights = torch.tensor(calc_class_weights(y, task_name), dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 6. Training loop
        n_epochs = 10

        if self.linear_probe:
            cached_train_loaders = [
                DataLoader(
                    loader.dataset,
                    batch_size=64,
                    shuffle=False,
                    collate_fn=loader.collate_fn,
                    **loader_kwargs
                )
                for loader in train_loader_list
            ]
            cached_val_loaders = [
                DataLoader(
                    loader.dataset,
                    batch_size=64,
                    shuffle=False,
                    collate_fn=loader.collate_fn,
                    **loader_kwargs
                )
                for loader in val_loader_list
            ]
            self._fit_linear_probe_cached(cached_train_loaders, cached_val_loaders, criterion, n_epochs)
            return

        print(f"Starting training for {n_epochs} epochs on {self.device}...")

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0

            for train_loader in train_loader_list:
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

                    batch_size = target.size(0)
                    total_loss += loss.item() * batch_size
                    _, predicted = output.max(1)
                    total_samples += batch_size
                    correct += predicted.eq(target).sum().item()

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total_samples if total_samples > 0 else 0.0:.2f}%'
                    })

            epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
            epoch_acc = 100. * correct / total_samples if total_samples > 0 else 0.0

            val_loss = None
            val_acc = None
            if val_loader_list:
                v_loss = 0.0
                v_acc = 0.0
                v_batches = 0
                for val_loader in val_loader_list:
                    l, a = self._evaluate_loader(val_loader, criterion)
                    v_loss += l
                    v_acc += a
                    v_batches += 1
                if v_batches > 0:
                    val_loss = v_loss / v_batches
                    val_acc = v_acc / v_batches

            # Log metrics
            metrics = {
                f"{self.name}/train_loss": epoch_loss,
                f"{self.name}/train_acc": epoch_acc / 100.0,
            }
            if val_loss is not None and val_acc is not None:
                metrics[f"{self.name}/val_loss"] = val_loss
                metrics[f"{self.name}/val_acc"] = val_acc

            if self.wandb_run:
                wandb_utils.log(metrics, step=epoch + 1)

            if val_loss is not None and val_acc is not None:
                print(
                    f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}, "
                    f"Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%"
                )
            else:
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
        task_name = meta_data["task_name"]
        print("[LUNA] Preprocessing evaluation data...")
        cached_make_dataset = self.cache.cache(make_dataset_luna)
        datasets = [
            cached_make_dataset(
                X_, None, task_name,
                m_["sampling_frequency"],
                m_["channel_names"],
                train=False,
                patch_size=self.patch_size
            )
            for X_, m_ in zip(X, meta)
        ]

        dataset_list = [dataset for dataset in datasets if len(dataset) > 0]
        if not dataset_list:
            logger.warning("No evaluation samples available after preprocessing.")
            return np.array([])

        num_workers = 2
        loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for dataset in dataset_list:
                X_all = dataset.data
                test_dataset = SimpleDataset(X_all, y=None)
                collate_fn = self._get_collate_fn(dataset.ch_names)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=64,
                    shuffle=False,
                    collate_fn=collate_fn,
                    **loader_kwargs
                )

                for batch in tqdm(test_loader, desc="Predicting"):
                    data = batch["sample"].to(self.device)
                    pos = batch["pos"].to(self.device)

                    output = self.model(data, pos)
                    _, predicted = output.max(1)
                    predictions.extend(predicted.cpu().numpy())

        mapped_predictions = np.array([reverse_map_label(idx, task_name) for idx in predictions])
        return mapped_predictions
