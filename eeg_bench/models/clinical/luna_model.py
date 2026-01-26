"""LUNA Clinical Model Wrapper for EEG-Bench.

This module provides a wrapper for the LUNA (Linear Universal Neural Architecture)
model for clinical EEG classification tasks. LUNA is a topology-agnostic foundation
model that uses query-based channel unification.

Reference:
    Döner et al. (2025). LUNA: Efficient and Topology-Agnostic Foundation Model
    for EEG Signal Analysis. NeurIPS 2025.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from safetensors.torch import load_file as load_safetensors

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from ...utils import wandb_utils

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global imports for LUNA modules (set up dynamically)
LUNA = None
CrossAttentionBlock = None
PatchEmbedNetwork = None
RotaryTransformerBlock = None
FrequencyFeatureEmbedder = None
ChannelEmbeddings = None

DEFAULT_SINGLE_LABEL_CHUNK_LEN_S = 10
DEFAULT_MULTILABEL_CHUNK_LEN_S = 16
TARGET_SAMPLING_FREQ = 256


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


class LUNAClinicalWrapper(nn.Module):
    """Wraps the LUNA model for clinical classification tasks."""

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        num_classes: int,
        num_labels_per_chunk: Optional[int] = None,
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
        self.is_multilabel_task = num_labels_per_chunk is not None
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.n_timepoints = n_timepoints

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
            num_classes=num_classes,  # Classification mode
        ).to(self.device)

        # Load pretrained weights if available
        if pretrained_path is not None:
            self._load_pretrained_weights(pretrained_path, freeze_backbone)

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

        self.loss_fn = nn.CrossEntropyLoss()

    def _load_pretrained_weights(self, pretrained_path: str, freeze_backbone: bool):
        """Load pretrained weights from safetensors file."""
        pretrained_path = Path(pretrained_path)

        if not pretrained_path.exists():
            logger.warning(f"Pretrained weights not found at {pretrained_path}. Training from scratch.")
            return

        try:
            # Load safetensors
            state_dict = load_safetensors(str(pretrained_path))

            # Remove classification head weights if present (we have different num_classes)
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
            logits: Classification logits [B, num_classes]
        """
        x = x.to(self.device)
        channel_locations = channel_locations.to(self.device)

        # LUNA expects mask=None for classification (no masking during inference)
        # Forward returns (x_classified, x_original) for classification mode
        logits, _ = self.backbone(
            x_signal=x,
            mask=None,
            channel_locations=channel_locations,
            channel_names=None  # Not used in classification mode
        )

        return logits


class LUNAClinicalModel(AbstractModel):
    """LUNA model wrapper for clinical EEG classification tasks."""

    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        chunk_len_s: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        biofoundation_path: Optional[str] = "/raid/spanchavati/",
        patch_size: int = 40,
        num_queries: int = 4,
        embed_dim: int = 64,
        depth: int = 8,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        freeze_backbone: bool = True,
    ):
        """Initialize LUNA clinical model.

        Args:
            num_classes: Number of output classes
            num_labels_per_chunk: For multilabel tasks, number of labels per chunk
            chunk_len_s: Chunk length in seconds (None for no chunking)
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
        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk
        self.freeze_backbone = freeze_backbone
        self.target_sampling_freq = TARGET_SAMPLING_FREQ

        if self.num_labels_per_chunk is not None:
            self.chunk_len_s = chunk_len_s if chunk_len_s is not None else DEFAULT_MULTILABEL_CHUNK_LEN_S
            self.internal_chunk_len_s = None
            self.use_internal_chunking = False
        else:
            # Handle chunking inside the model so we don't depend on dataset-side chunking.
            self.chunk_len_s = None
            self.internal_chunk_len_s = chunk_len_s if chunk_len_s is not None else DEFAULT_SINGLE_LABEL_CHUNK_LEN_S
            self.use_internal_chunking = True
            logger.info(
                "Single-label task detected. Using internal chunking with %s-second windows.",
                self.internal_chunk_len_s,
            )

        # Model architecture parameters
        self.patch_size = patch_size
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Setup LUNA imports from BioFoundation
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

        self.model: Optional[LUNAClinicalWrapper] = None

    def _get_channel_coords(self, ch_names: List[str]) -> torch.Tensor:
        """Get 3D channel coordinates from position bank.

        Args:
            ch_names: List of channel names

        Returns:
            positions: Tensor of shape [C, 3] with 3D coordinates
        """
        # Clean channel names (remove 'EEG' prefix if present)
        clean_names = [c.replace("EEG", "").strip() for c in ch_names]

        # Get positions from position bank
        positions = self.pos_bank(clean_names)

        # Handle different output formats from position bank
        if isinstance(positions, dict):
            positions = positions.get(
                "positions",
                positions.get("coords", positions.get("last_hidden_state"))
            )

        # Ensure correct shape: [C, 3]
        if positions.dim() == 3:
            positions = positions.squeeze(0)

        return positions.float().to(self.device)

    def _init_model(self, sample: np.ndarray) -> None:
        """Initialize the LUNA model based on sample data shape.

        Args:
            sample: Sample data array with shape [C, T]
        """
        n_channels, n_timepoints = sample.shape[0], sample.shape[1]

        self.model = LUNAClinicalWrapper(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            num_classes=self.num_classes,
            num_labels_per_chunk=self.num_labels_per_chunk,
            patch_size=self.patch_size,
            num_queries=self.num_queries,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            pretrained_path=self.pretrained_path,
            freeze_backbone=self.freeze_backbone,
        ).to(self.device)

        logger.info(f"Initialized LUNA model with {n_channels} channels and {n_timepoints} timepoints")

    def _internal_chunk_len_samples(self) -> Optional[int]:
        if not self.use_internal_chunking or self.internal_chunk_len_s is None:
            return None
        chunk_len = int(round(self.internal_chunk_len_s * self.target_sampling_freq))
        chunk_len = max(self.patch_size, (chunk_len // self.patch_size) * self.patch_size)
        return chunk_len

    def _split_signal_into_chunks(self, signal: torch.Tensor, chunk_len: int) -> torch.Tensor:
        """Split a single recording (C, T) into chunks of length chunk_len."""
        assert signal.dim() == 2, "Signal must have shape [C, T]"
        total_len = (signal.shape[-1] // self.patch_size) * self.patch_size
        if total_len == 0:
            total_len = signal.shape[-1]
        signal = signal[..., :total_len].contiguous()
        chunk_len = min(chunk_len, signal.shape[-1])
        chunk_len = max(self.patch_size, (chunk_len // self.patch_size) * self.patch_size)
        n_chunks = max(1, signal.shape[-1] // chunk_len)
        trimmed = n_chunks * chunk_len
        signal = signal[..., :trimmed]
        chunks = signal.view(signal.shape[0], n_chunks, chunk_len).permute(1, 0, 2).contiguous()
        return chunks

    def _expand_label_for_chunks(self, label: torch.Tensor, num_chunks: int) -> torch.Tensor:
        if label.dim() == 0:
            return label.repeat(num_chunks)
        repeat_dims = [num_chunks] + [1] * label.dim()
        return label.unsqueeze(0).repeat(*repeat_dims)

    def _prepare_train_batch(
        self,
        x: torch.Tensor,
        yb: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_internal_chunking:
            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
            return x, yb, cb

        chunk_len = self._internal_chunk_len_samples()
        assert chunk_len is not None

        chunked_x, chunked_y, chunked_cb = [], [], []
        for i in range(x.size(0)):
            sample_chunks = self._split_signal_into_chunks(x[i], chunk_len)
            chunked_x.append(sample_chunks)
            chunked_y.append(self._expand_label_for_chunks(yb[i], sample_chunks.size(0)))
            chunked_cb.append(coords.unsqueeze(0).expand(sample_chunks.size(0), -1, -1))

        return (
            torch.cat(chunked_x, dim=0),
            torch.cat(chunked_y, dim=0),
            torch.cat(chunked_cb, dim=0),
        )

    def _prepare_inference_batch(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.use_internal_chunking:
            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
            return x, indices, cb

        chunk_len = self._internal_chunk_len_samples()
        assert chunk_len is not None

        chunked_x, chunked_idx, chunked_cb = [], [], []
        for i in range(x.size(0)):
            sample_chunks = self._split_signal_into_chunks(x[i], chunk_len)
            chunked_x.append(sample_chunks)
            chunked_idx.append(indices[i].repeat(sample_chunks.size(0)))
            chunked_cb.append(coords.unsqueeze(0).expand(sample_chunks.size(0), -1, -1))

        return (
            torch.cat(chunked_x, dim=0),
            torch.cat(chunked_idx, dim=0),
            torch.cat(chunked_cb, dim=0),
        )

    def _train_epoch(
        self,
        train_loader: DataLoader,
        coords: torch.Tensor,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        epoch: int,
    ) -> tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            coords: Channel coordinates [C, 3]
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number

        Returns:
            avg_loss: Average training loss
            accuracy: Training accuracy
        """
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

            x_chunks, y_chunks, cb = self._prepare_train_batch(x, yb, coords)

            optimizer.zero_grad()
            logits = self.model(x_chunks, cb)
            loss = self.model.loss_fn(logits, y_chunks)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            batch_samples = x_chunks.size(0)
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

            if logits.dim() == 2:
                preds = torch.argmax(logits, dim=1)
                target = y_chunks if y_chunks.dim() == 1 else y_chunks.argmax(dim=1)
                correct += (preds == target).sum().item()
                total_acc_samples += batch_samples

            # Memory cleanup
            del x, yb, x_chunks, y_chunks, cb, logits, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_acc_samples if total_acc_samples > 0 else 0.0
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        coords: torch.Tensor,
        epoch: int
    ) -> tuple[float, float]:
        """Validate for one epoch.

        Args:
            val_loader: Validation data loader
            coords: Channel coordinates [C, 3]
            epoch: Current epoch number

        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
        """
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

            x_chunks, y_chunks, cb = self._prepare_train_batch(x, yb, coords)

            logits = self.model(x_chunks, cb)
            loss = self.model.loss_fn(logits, y_chunks)
            batch_samples = x_chunks.size(0)
            val_loss += loss.item() * batch_samples
            val_samples += batch_samples

            if logits.dim() == 2:
                preds = torch.argmax(logits, dim=1)
                target = y_chunks if y_chunks.dim() == 1 else y_chunks.argmax(dim=1)
                val_correct += (preds == target).sum().item()
                val_acc_samples += batch_samples

            # Memory cleanup
            del x, yb, x_chunks, y_chunks, cb, logits, loss
            torch.cuda.empty_cache()

        avg_loss = val_loss / val_samples if val_samples > 0 else 0.0
        accuracy = val_correct / val_acc_samples if val_acc_samples > 0 else 0.0
        return avg_loss, accuracy

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        """Train the LUNA model.

        Args:
            X: List of data arrays [n_samples, n_channels, n_timepoints]
            y: List of label arrays [n_samples]
            meta: List of metadata dicts containing task info, channels, etc.
        """
        task_name = meta[0]["task_name"]

        # Create training dataset using make_dataset_2
        dataset_train = make_dataset_2(
            X, y, meta, task_name, self.name, self.chunk_len_s,
            is_train=True, use_cache=True
        )

        # Safety check: retry without cache if empty
        if len(dataset_train) == 0:
            logger.warning("Dataset empty. Retrying without cache...")
            dataset_train = make_dataset_2(
                X, y, meta, task_name, self.name, self.chunk_len_s,
                is_train=True, use_cache=False
            )

        if len(dataset_train) == 0:
            logger.warning("Dataset empty after retries. Skipping training.")
            return

        # Split into train/val (15% validation like EEGLejepa)
        val_split = 0.15
        dataset_train, dataset_val = dataset_train.split_train_val(val_split)

        if len(dataset_train) == 0:
            logger.warning("Training split is empty. Skipping training.")
            return

        # Initialize model if needed
        sample_data, _, _ = dataset_train[0]
        if self.model is None:
            self._init_model(sample_data)

        # Setup loss function with class weights
        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Create data loaders
        batch_size = 64 if self.chunk_len_s else 1
        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        # Get channel coordinates
        coords_train = self._get_channel_coords(dataset_train.ch_names)
        coords_val = self._get_channel_coords(dataset_val.ch_names)

        # Setup optimizer and scheduler (following EEGLejepa pattern)
        max_epochs = 30
        steps_per_epoch = len(train_loader)
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

        # Early stopping setup
        patience = 10
        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None

        # Training loop
        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self._train_epoch(
                train_loader, coords_train, optimizer, scheduler, epoch
            )
            val_loss, val_acc = self._validate_epoch(val_loader, coords_val, epoch)

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

            # Console logging
            print(
                f"[Epoch {epoch:02d}/{max_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e} patience={patience_counter}/{patience}"
            )

            # Early stopping trigger
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch with val_loss={best_val_loss:.4f}")

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        """Generate predictions for the input data.

        Args:
            X: List of data arrays [n_samples, n_channels, n_timepoints]
            meta: List of metadata dicts

        Returns:
            predictions: Array of predicted labels
        """
        assert self.model is not None, "Model must be trained before prediction"

        task_name = meta[0]["task_name"]

        # Create test dataset
        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name, self.chunk_len_s,
            is_train=False, use_cache=True
        )

        if len(dataset_test) == 0:
            return np.array([])

        # Create test loader
        batch_size = 64 if self.chunk_len_s else 1
        test_loader = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # Get channel coordinates
        coords = self._get_channel_coords(dataset_test.ch_names)
        self.model.eval()

        # Collect predictions
        predictions = []
        indices = []

        for x, idx, _ in tqdm(test_loader, desc="Predicting"):
            x = x.to(self.device)
            idx = idx.to(self.device)
            batch_x, batch_idx, cb = self._prepare_inference_batch(x, idx, coords)

            logits = self.model(batch_x, cb)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred.cpu())
            indices.append(batch_idx.cpu())

            del x, idx, batch_x, batch_idx, cb, logits, pred
            torch.cuda.empty_cache()

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        indices = torch.cat(indices, dim=0).cpu().numpy()

        # Aggregate predictions if using chunks (majority voting)
        if self.chunk_len_s is not None and not self.model.is_multilabel_task:
            unique_indices = np.unique(indices)
            aggregated_predictions = []
            for idx in unique_indices:
                idx_predictions = predictions[indices == idx]
                most_common_prediction = Counter(idx_predictions).most_common(1)[0][0]
                aggregated_predictions.append(most_common_prediction)
            predictions = np.array(aggregated_predictions)
        elif self.use_internal_chunking and not self.model.is_multilabel_task:
            unique_indices = np.unique(indices)
            aggregated_predictions = []
            for idx in unique_indices:
                idx_predictions = predictions[indices == idx]
                most_common_prediction = Counter(idx_predictions).most_common(1)[0][0]
                aggregated_predictions.append(most_common_prediction)
            predictions = np.array(aggregated_predictions)

        # Map predictions back to original labels
        mapped_pred = np.array([map_label_reverse(pred, task_name) for pred in predictions])
        return mapped_pred
