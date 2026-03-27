# lejepa_bci_model.py

from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys
import logging

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset import make_dataset_lejepa  # LeJEPA-specific loader with Defossez scaling
from .LaBraM.utils_2 import calc_class_weights, reverse_map_label, n_unique_labels
from joblib import Memory
from ...config import get_config_value, LeJEPAConfig
from ...utils import wandb_utils

from transformers import AutoModel
import random
import math
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# eegfm imports are done dynamically in _setup_eegfm_imports()
EEGLEJEPAConfig = None
ConvPatchEmbedderConfig = None
DynamicChannelMixerConfig = None
EncoderConfig = None


def _setup_eegfm_imports(eegfm_path: Optional[str] = None):
    """Setup eegfm imports by adding path to sys.path if needed."""
    global EEGLEJEPAConfig, ConvPatchEmbedderConfig, DynamicChannelMixerConfig, EncoderConfig

    if eegfm_path and eegfm_path not in sys.path:
        sys.path.insert(0, eegfm_path)
        logger.info(f"Added eegfm path to sys.path: {eegfm_path}")

    # Import eegfm modules
    from eegfmchallenge.models.eeglejepa import EEGLEJEPAConfig as _EEGLEJEPAConfig
    from eegfmchallenge.models.patch_embedder import ConvPatchEmbedderConfig as _ConvPatchEmbedderConfig
    from eegfmchallenge.models.channel_mixer import DynamicChannelMixerConfig as _DynamicChannelMixerConfig
    from eegfmchallenge.models.common import EncoderConfig as _EncoderConfig

    EEGLEJEPAConfig = _EEGLEJEPAConfig
    ConvPatchEmbedderConfig = _ConvPatchEmbedderConfig
    DynamicChannelMixerConfig = _DynamicChannelMixerConfig
    EncoderConfig = _EncoderConfig

class ConcreteLeJEPABCI(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_path: str | None = None,
        version: int | None = None,
        freeze_encoder: bool = True,
        config_path: Path | None = None,
        pretrained_path: Path | None = None,
    ):
        super().__init__()


        # ------------------------------------------------------------
        # Pretrained config / checkpoint resolution (matching clinical)
        # ------------------------------------------------------------
        if (config_path is None or pretrained_path is None) and base_path is not None and version is not None:
            base_path_resolved = Path(base_path) / f"version_{version}"

            candidate_config = base_path_resolved / "config" / "config.pkl"
            candidate_ckpt = base_path_resolved / "checkpoints" / "last.ckpt"

            if config_path is None and candidate_config.exists():
                config_path = candidate_config
            elif config_path is None:
                print(f"[LeJEPABCI] No config found at {candidate_config}. Using default config.")

            if pretrained_path is None and candidate_ckpt.exists():
                pretrained_path = candidate_ckpt
            elif pretrained_path is None:
                print(f"[LeJEPABCI] No checkpoint found at {candidate_ckpt}. Training from scratch.")

        # ------------------------------------------------------------
        # Build model config
        # ------------------------------------------------------------
        if config_path is not None:
            with open(config_path, "rb") as f:
                pretrain_config = pickle.load(f)
            cfg = EEGLEJEPAConfig(**pretrain_config["model"])
            print("[LeJEPABCI] Loaded Config!")
        else:
            cfg = EEGLEJEPAConfig(
                name="EEGLEJEPA",
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

        # ------------------------------------------------------------
        # Build backbone
        # ------------------------------------------------------------
        self.backbone = cfg.build()
        DIM = self.backbone.dim

        # ------------------------------------------------------------
        # Load pretrained weights (if available)
        # ------------------------------------------------------------
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("model.", ""): v for k, v in state.items()}

            # DEBUG: Verify checkpoint keys match model keys
            model_keys = set(self.backbone.state_dict().keys())
            ckpt_keys = set(state.keys())
            print(f"[LeJEPABCI] Checkpoint keys (first 5): {list(ckpt_keys)[:5]}")
            print(f"[LeJEPABCI] Model keys (first 5): {list(model_keys)[:5]}")

            # Load with strict=False but capture missing/unexpected
            load_result = self.backbone.load_state_dict(state, strict=False)
            missing_keys = load_result.missing_keys
            unexpected_keys = load_result.unexpected_keys

            matched_keys = model_keys & ckpt_keys
            print(f"[LeJEPABCI] Matched keys: {len(matched_keys)}/{len(model_keys)}")
            print(f"[LeJEPABCI] Missing keys: {len(missing_keys)}")
            print(f"[LeJEPABCI] Unexpected keys: {len(unexpected_keys)}")

            if len(missing_keys) > 0:
                print(f"[LeJEPABCI] WARNING: Missing keys (first 5): {missing_keys[:5]}")
            if len(unexpected_keys) > 0:
                print(f"[LeJEPABCI] WARNING: Unexpected keys (first 5): {unexpected_keys[:5]}")
            if len(matched_keys) == 0:
                print(f"[LeJEPABCI] CRITICAL: No keys matched! Checkpoint may have wrong format.")

            print(f"[LeJEPABCI] Loaded pretrained weights from {pretrained_path}")

        # ------------------------------------------------------------
        # Freeze encoder if requested
        # ------------------------------------------------------------
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.train()

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
    def __init__(
        self,
        config: Optional[LeJEPAConfig] = None,
        pretrained_path: Optional[str] = None,
        base_path: Optional[str] = None,
        version: Optional[int] = None,
        freeze_encoder: bool = True
    ):
        super().__init__("LeJEPABCI")
        assert torch.cuda.is_available(), "CUDA is not available"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Handle config vs legacy parameters
        if config is not None:
            # Use checkpoint path from config (get_checkpoint_path resolves full_path vs base_path+version)
            checkpoint_path = config.get_checkpoint_path()
            if checkpoint_path:
                # Extract base_path and version from full path for config discovery
                ckpt_path = Path(checkpoint_path)
                self.pretrained_path = ckpt_path
                self.config_path = None
                if ckpt_path.parent.name == "checkpoints":
                    version_dir = ckpt_path.parent.parent
                    if version_dir.name.startswith("version_"):
                        self.base_path = str(version_dir.parent)
                        self.version = int(version_dir.name.replace("version_", ""))
                        candidate_config = version_dir / "config" / "config.pkl"
                        if candidate_config.exists():
                            self.config_path = candidate_config
                    else:
                        self.base_path = None
                        self.version = None
                else:
                    self.base_path = None
                    self.version = None
            else:
                self.base_path = config.checkpoint_base_path
                self.version = config.checkpoint_version
                self.config_path = None
                self.pretrained_path = None
            self.freeze_encoder = config.freeze_encoder
            pos_bank_path = config.pos_bank_path
            eegfm_path = config.eegfm_path
        else:
            # Legacy mode - use parameters directly
            self.pretrained_path = Path(pretrained_path) if pretrained_path else None
            self.base_path = base_path
            self.version = version
            self.config_path = None
            self.freeze_encoder = freeze_encoder
            pos_bank_path = get_config_value("lejepa", {}).get("pos_bank_path", "./REVE_posbank")
            eegfm_path = get_config_value("lejepa", {}).get("eegfm_path")

        # Setup eegfm imports
        _setup_eegfm_imports(eegfm_path)

        self.cache = Memory(location=get_config_value("cache"), verbose=0)

        # Load position bank with HuggingFace fallback
        self.pos_bank = self._load_position_bank(pos_bank_path)

    def _load_position_bank(self, local_fallback_path: str):
        """Load REVE position bank - try local first, fall back to HuggingFace."""
        try:
            logger.info(f"Attempting to load position bank from local path: {local_fallback_path}")
            pos_bank = AutoModel.from_pretrained(
                local_fallback_path,
                trust_remote_code=True
            ).to(self.device)
            logger.info("Successfully loaded position bank from local storage.")
            return pos_bank
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            logger.info("Falling back to HuggingFace Hub (brain-bzh/reve-positions)...")
            pos_bank = AutoModel.from_pretrained(
                "brain-bzh/reve-positions",
                trust_remote_code=True
            ).to(self.device)
            return pos_bank



    def _get_coords_and_valid_mask(self, ch_names):
        """
        Get coordinates and a mask indicating which channels have valid positions.
        Returns: (coords, valid_indices) where coords only contains valid channels.
        """
        clean_names = [c.replace("EEG", "").strip() for c in ch_names]
        c = self.pos_bank(clean_names)
        if isinstance(c, dict):
            c = c.get("positions", c.get("coords", c.get("last_hidden_state")))
        if c.dim() == 3:
            c = c.squeeze(0)
        c = c.float().to(self.device)

        # Detect invalid positions (NaN or all zeros)
        valid_mask = ~(torch.isnan(c).any(dim=-1) | (c.abs().sum(dim=-1) == 0))
        valid_indices = torch.where(valid_mask)[0]

        if valid_indices.numel() < len(ch_names):
            print(f"[LeJEPABCI] Found {valid_indices.numel()} valid positions out of {len(ch_names)} channels")

        return c[valid_indices], valid_indices

    def _get_coords(self, ch_names):
        """Legacy method - returns all coords (may include invalid ones)."""
        clean_names = [c.replace("EEG", "").strip() for c in ch_names]
        c = self.pos_bank(clean_names)
        if isinstance(c, dict):
            c = c.get("positions", c.get("coords", c.get("last_hidden_state")))
        if c.dim() == 3:
            c = c.squeeze(0)
        return c.float().to(self.device)

    def _train_epoch(self, dataloader, optimizer, scheduler, coords, valid_indices=None):
        # Only set head to train mode; preserve backbone eval mode if frozen
        self.model.head.train()
        if self.freeze_encoder:
            self.model.backbone.eval()  # Explicitly keep frozen encoder in eval mode
        else:
            self.model.backbone.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        for x, y_batch in tqdm(dataloader, desc="Training", leave=False):
            x, y_batch = x.to(self.device), y_batch.to(self.device).argmax(dim=1)

            # Filter to only valid channels if needed
            if valid_indices is not None:
                x = x[:, valid_indices, :]

            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)

            optimizer.zero_grad()
            logits = self.model(x, cb)
            loss = self.model.loss_fn(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += (preds == y_batch).sum().item()
            total_samples += x.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def _validate_epoch(self, dataloader, coords, valid_indices=None):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        with torch.no_grad():
            for x, y_batch in tqdm(dataloader, desc="Validation", leave=False):
                x, y_batch = x.to(self.device), y_batch.to(self.device).argmax(dim=1)

                # Filter to only valid channels if needed
                if valid_indices is not None:
                    x = x[:, valid_indices, :]

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

    @torch.no_grad()
    def _extract_embeddings(self, dataloader, coords, valid_indices=None):
        """Extract CLS token embeddings from frozen encoder in a single pass."""
        self.model.backbone.eval()
        embeddings_list = []
        labels_list = []

        for x, y_batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            x = x.to(self.device)
            y_batch = y_batch.to(self.device).argmax(dim=1)

            # Filter to only valid channels if needed
            if valid_indices is not None:
                x = x[:, valid_indices, :]

            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)

            # Get CLS token from backbone
            outputs = self.model.backbone.forward_downstream(x=x, channel_locations=cb)
            cls = outputs["cls_token"]
            if cls.dim() == 3:
                cls = cls.mean(dim=1)

            embeddings_list.append(cls.cpu())
            labels_list.append(y_batch.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return embeddings, labels

    def _train_epoch_cached(self, dataloader, optimizer, scheduler):
        """Train only the head on cached embeddings."""
        self.model.head.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for embeddings, y_batch in dataloader:
            embeddings = embeddings.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            logits = self.model.head(embeddings)
            loss = self.model.loss_fn(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * embeddings.size(0)
            preds = torch.argmax(logits, dim=1)
            running_corrects += (preds == y_batch).sum().item()
            total_samples += embeddings.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def _validate_epoch_cached(self, dataloader):
        """Validate on cached embeddings."""
        self.model.head.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        with torch.no_grad():
            for embeddings, y_batch in dataloader:
                embeddings = embeddings.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model.head(embeddings)
                loss = self.model.loss_fn(logits, y_batch)

                running_loss += loss.item() * embeddings.size(0)
                preds = torch.argmax(logits, dim=1)
                running_corrects += (preds == y_batch).sum().item()
                total_samples += embeddings.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        return epoch_loss, epoch_acc

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        task_name = meta[0]["task_name"]
        num_classes = n_unique_labels(task_name)
        self.model = ConcreteLeJEPABCI(
            num_classes=num_classes,
            base_path=self.base_path,
            version=self.version,
            freeze_encoder=self.freeze_encoder,
            config_path=self.config_path,
            pretrained_path=self.pretrained_path,
        ).to(self.device)

        datasets = [self.cache.cache(make_dataset_lejepa)(X_, y_, task_name, m_["sampling_frequency"], m_["channel_names"], train=True, split_size=0.15)
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
        max_epochs = 30
        patience = 10

        if self.freeze_encoder:
            # =============================================
            # CACHED EMBEDDINGS PATH (frozen encoder)
            # =============================================
            print("[LeJEPABCI] Using cached embeddings (freeze_encoder=True)")
            
            # Extract embeddings from all datasets in a single pass
            train_embeddings_list = []
            train_labels_list = []
            for dataset, ch_names in zip(dataset_train_list, ch_names_list_train):
                loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
                coords, valid_indices = self._get_coords_and_valid_mask(ch_names)
                emb, lbl = self._extract_embeddings(loader, coords, valid_indices)
                train_embeddings_list.append(emb)
                train_labels_list.append(lbl)
            
            val_embeddings_list = []
            val_labels_list = []
            for dataset, ch_names in zip(dataset_val_list, ch_names_list_val):
                loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
                coords, valid_indices = self._get_coords_and_valid_mask(ch_names)
                emb, lbl = self._extract_embeddings(loader, coords, valid_indices)
                val_embeddings_list.append(emb)
                val_labels_list.append(lbl)
            
            # Combine all embeddings into single TensorDatasets
            all_train_emb = torch.cat(train_embeddings_list, dim=0)
            all_train_lbl = torch.cat(train_labels_list, dim=0)
            cached_train_dataset = TensorDataset(all_train_emb, all_train_lbl)
            
            all_val_emb = torch.cat(val_embeddings_list, dim=0)
            all_val_lbl = torch.cat(val_labels_list, dim=0)
            cached_val_dataset = TensorDataset(all_val_emb, all_val_lbl)
            
            print(f"[LeJEPABCI] Cached {len(cached_train_dataset)} train and {len(cached_val_dataset)} val embeddings")
            
            # Use larger batch size for cached training (no encoder memory needed)
            cached_batch_size = batch_size * 4  # 256
            cached_train_loader = DataLoader(cached_train_dataset, batch_size=cached_batch_size, shuffle=True, pin_memory=True)
            cached_val_loader = DataLoader(cached_val_dataset, batch_size=cached_batch_size, shuffle=False, pin_memory=True)
            
            # Setup optimizer for head only
            steps_per_epoch = math.ceil(len(cached_train_loader))
            max_lr = 1e-4
            
            optimizer = optim.AdamW(self.model.head.parameters(), lr=1e-6, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=max_epochs,
                pct_start=0.2,
            )
            
            patience_counter = 0
            best_val_loss = float("inf")
            best_model_state = None
            
            for epoch in range(1, max_epochs + 1):
                train_loss, train_acc = self._train_epoch_cached(cached_train_loader, optimizer, scheduler)
                val_loss, val_acc = self._validate_epoch_cached(cached_val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                current_lr = scheduler.get_last_lr()[0]
                
                if self.wandb_run:
                    wandb_utils.log(
                        {
                            f"{self.name}/train_loss": train_loss,
                            f"{self.name}/train_acc": train_acc,
                            f"{self.name}/val_loss": val_loss,
                            f"{self.name}/val_acc": val_acc,
                            f"{self.name}/lr": current_lr,
                        },
                        step=epoch,
                    )
                
                print(f"[Epoch {epoch:02d}/{max_epochs}] "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                      f"lr={current_lr:.2e} patience={patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    break
            
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
        
        else:
            # =============================================
            # FULL FORWARD PASS PATH (fine-tuning encoder)
            # =============================================
            print("[LeJEPABCI] Using full forward pass (freeze_encoder=False)")
            
            train_loader_list = [
                DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
                for train_dataset in dataset_train_list
            ]
            valid_loader_list = [
                DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
                for valid_dataset in dataset_val_list
            ]

            steps_per_epoch = math.ceil(sum(len(train_loader) for train_loader in train_loader_list))
            max_lr = 1e-4

            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=max_epochs,
                pct_start=0.2,
            )

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
                    coords, valid_indices = self._get_coords_and_valid_mask(ch_names)
                    train_loss, train_acc = self._train_epoch(train_loader, optimizer, scheduler, coords, valid_indices)
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
                        coords, valid_indices = self._get_coords_and_valid_mask(ch_names)
                        val_loss, val_acc = self._validate_epoch(valid_loader, coords, valid_indices)
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

                    current_lr = scheduler.get_last_lr()[0]

                    if self.wandb_run:
                        wandb_utils.log(
                            {
                                f"{self.name}/train_loss": avg_train_loss,
                                f"{self.name}/train_acc": avg_train_acc,
                                f"{self.name}/val_loss": avg_val_loss,
                                f"{self.name}/val_acc": avg_val_acc,
                                f"{self.name}/lr": current_lr,
                            },
                            step=epoch,
                        )

                    print(f"[Epoch {epoch:02d}/{max_epochs}] "
                          f"train_loss={avg_train_loss:.4f} train_acc={avg_train_acc:.4f} | "
                          f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f} | "
                          f"lr={current_lr:.2e} patience={patience_counter}/{patience}")

                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
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
                    print(f"[Epoch {epoch:02d}/{max_epochs}] "
                          f"train_loss={avg_train_loss:.4f} train_acc={avg_train_acc:.4f}")

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        task_name = meta[0]["task_name"]
        self.model.eval()
        
        dataset_test_list = [self.cache.cache(make_dataset_lejepa)(X_, None, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=False, split_size=0)
                             for X_, meta_ in zip(X, meta)]
        dataset_test_list = [dataset for dataset in dataset_test_list if len(dataset) > 0]
        ch_names_list = [dataset.ch_names for dataset in dataset_test_list]

        batch_size = 64
        test_loader_list = [DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for test_dataset in dataset_test_list]

        predictions = []
        for test_loader, ch_names in zip(test_loader_list, ch_names_list):
            coords, valid_indices = self._get_coords_and_valid_mask(ch_names)
            preds_all = []
            for x in tqdm(test_loader, desc="BCI Predicting", leave=False):
                x = x.to(self.device)
                # Filter to only valid channels if needed
                if valid_indices is not None:
                    x = x[:, valid_indices, :]
                cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
                logits = self.model(x, cb)
                preds_all.append(torch.argmax(logits, dim=1).cpu())
            predictions.append(torch.cat(preds_all, dim=0))

        predictions = torch.cat(predictions, dim=0).numpy()
        mapped_pred = np.array([reverse_map_label(idx, task_name) for idx in predictions])
        return mapped_pred
