# lejepa_clinical_model.py

from __future__ import annotations
from typing import List, Dict, Optional, cast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from collections import Counter
import gc
import math
import sys
import logging
from pathlib import Path
from ..abstract_model import AbstractModel
from ...config import get_config_value, LeJEPAConfig

# LaBraM Clinical Utilities
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from .LaBraM import utils

from transformers import AutoModel
from ...utils import wandb_utils

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
    from eegfm.models.eeglejepa import EEGLEJEPAConfig as _EEGLEJEPAConfig
    from eegfm.models.patch_embedder import ConvPatchEmbedderConfig as _ConvPatchEmbedderConfig
    from eegfm.models.channel_mixer import DynamicChannelMixerConfig as _DynamicChannelMixerConfig
    from eegfm.models.common import EncoderConfig as _EncoderConfig

    EEGLEJEPAConfig = _EEGLEJEPAConfig
    ConvPatchEmbedderConfig = _ConvPatchEmbedderConfig
    DynamicChannelMixerConfig = _DynamicChannelMixerConfig
    EncoderConfig = _EncoderConfig

class ConcreteLeJEPAClinical(nn.Module):
    def __init__(
        self,
        num_classes,
        num_labels_per_chunk,
        base_path=None,
        version=None,
        freeze_encoder=True,
    ):
        super().__init__()

        DIM = 384
        self.is_multilabel_task = num_labels_per_chunk is not None

        # ------------------------------------------------------------
        # Pretrained config / checkpoint resolution (SAFE)
        # ------------------------------------------------------------
        config_path = None
        pretrained_path = None

        if base_path is not None and version is not None:
            base_path = Path(base_path) / f"version_{version}"

            candidate_config = base_path / "config" / "config.pkl"
            candidate_ckpt = base_path / "checkpoints" / "last.ckpt"

            if candidate_config.exists():
                config_path = candidate_config
            else:
                print(f"[LeJEPAClinical] No config found at {candidate_config}. Using default config.")

            if candidate_ckpt.exists():
                pretrained_path = candidate_ckpt
            else:
                print(f"[LeJEPAClinical] No checkpoint found at {candidate_ckpt}. Training from scratch.")

        # ------------------------------------------------------------
        # Build model config
        # ------------------------------------------------------------
        if config_path is not None:
            with open(config_path, "rb") as f:
                pretrain_config = pickle.load(f)
            cfg = EEGLEJEPAConfig(**pretrain_config["model"])
        else:
            cfg = EEGLEJEPAConfig(
                name="EEGLEJEPA",
                dim=384,
                proj_dim=16,
                patch_size=25,
                n_channels=128,
                max_time=1500,
                patch_embedder=ConvPatchEmbedderConfig(
                    name="ConvPatchEmbedder",
                    preserve_channels=False,
                ),
                channel_mixer_config=DynamicChannelMixerConfig(
                    name="DynamicChannelMixer",
                    coord_dim=3,
                    output_channels=64,
                ),
                encoder_config=EncoderConfig(
                    dim=384,
                    depth=12,
                    heads=6,
                    use_flash_attn=True,
                ),
                predictor_config=EncoderConfig(
                    dim=128,
                    depth=4,
                    heads=4,
                    use_flash_attn=True,
                ),
                masking={
                    "mask_ratio": 0.5,
                    "block_size_range": [5, 10],
                    "strategy_probs": [1.0, 0.0, 0.0],
                },
            
            )

        # ------------------------------------------------------------
        # Build backbone
        # ------------------------------------------------------------
        self.backbone = cfg.build()
        self.chunk_length = 5000 #20s chunks!

        # ------------------------------------------------------------
        # Load pretrained weights (if available)
        # ------------------------------------------------------------
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("model.", ""): v for k, v in state.items()}
            self.backbone.load_state_dict(state, strict=False)
            print("[LeJEPAClinical] Loaded pretrained weights")

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

        out_dim = num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1)
        self.head = nn.Linear(DIM, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes


    def forward(self, x, coords):

        B, C, T = x.shape
        n_chunks = T // self.chunk_length
        chunk_trunc = n_chunks * self.chunk_length if n_chunks else T
        x = x[:, :, :chunk_trunc]

        # Reshape into segments:
        x = x.view(B, C, n_chunks, self.chunk_length)
        # Permute to (batch_size, num_chunks, n_channels, chunk_length)
        x = x.permute(0, 2, 1, 3)
        # Merge batch and chunk dimensions for efficient processing:
        x = x.reshape(B * n_chunks, C, self.chunk_length)


        outputs = self.backbone.forward_downstream(x=x, channel_locations=coords)
        cls = outputs["cls_token"]

        # Restore the batch and chunk dimensions:
        embedding_dim = cls.shape[1]
        cls = cls.view(B, n_chunks, embedding_dim)
        
        # Simple aggregation: mean pooling over segments
        if cls.dim() == 3:
            cls = cls.mean(dim=1)
        logits = self.head(cls)
        if self.is_multilabel_task:
            logits = logits.reshape(x.shape[0], self.num_classes, -1)
        return logits

class EEGLeJEPAClinicalModel(AbstractModel):
    def __init__(
        self,
        config: Optional[LeJEPAConfig] = None,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        base_path: Optional[str] = None,
        version: Optional[int] = None,
        freeze_encoder: bool = True
    ):
        super().__init__("LeJEPAClinical")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_len_s = None if num_labels_per_chunk is None else 16
        self.num_labels_per_chunk = num_labels_per_chunk

        # Handle config vs legacy parameters
        if config is not None:
            # Use checkpoint path from config (get_checkpoint_path resolves full_path vs base_path+version)
            checkpoint_path = config.get_checkpoint_path()
            if checkpoint_path:
                # Extract base_path and version from full path for ConcreteLeJEPAClinical
                # which expects base_path/version_X/checkpoints/last.ckpt structure
                ckpt_path = Path(checkpoint_path)
                if ckpt_path.name == "last.ckpt" and ckpt_path.parent.name == "checkpoints":
                    version_dir = ckpt_path.parent.parent
                    if version_dir.name.startswith("version_"):
                        base_path = str(version_dir.parent)
                        version = int(version_dir.name.replace("version_", ""))
                    else:
                        # full_path mode - pass checkpoint directly
                        base_path = None
                        version = None
                else:
                    base_path = None
                    version = None
            else:
                base_path = config.checkpoint_base_path
                version = config.checkpoint_version
            freeze_encoder = config.freeze_encoder
            pos_bank_path = config.pos_bank_path
            eegfm_path = config.eegfm_path
        else:
            # Legacy mode - use parameters directly (with old defaults if not provided)
            pos_bank_path = get_config_value("lejepa", {}).get("pos_bank_path", "./REVE_posbank")
            eegfm_path = get_config_value("lejepa", {}).get("eegfm_path")

        # Setup eegfm imports
        #for uddhav set path directly
        _setup_eegfm_imports(eegfm_path)

        # Load position bank with HuggingFace fallback
        self.pos_bank = self._load_position_bank(pos_bank_path)

        self.model = ConcreteLeJEPAClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            base_path=base_path,
            version=version,
            freeze_encoder=freeze_encoder
        ).to(self.device)

    def _load_position_bank(self, local_fallback_path: str):
        """Load REVE position bank - try HuggingFace first, fall back to local."""
        try:
            logger.info("Attempting to load position bank from HuggingFace Hub...")
            pos_bank = AutoModel.from_pretrained(
                "brain-bzh/reve-positions",
                trust_remote_code=True
            ).to(self.device)
            logger.info("Successfully loaded position bank from HuggingFace Hub")
            return pos_bank
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace Hub: {e}")
            logger.info(f"Falling back to local path: {local_fallback_path}")
            pos_bank = AutoModel.from_pretrained(
                local_fallback_path,
                trust_remote_code=True
            ).to(self.device)
            return pos_bank

    def _coords(self, ch_names):
        names = [c.replace("EEG", "").strip() for c in ch_names]
        c = self.pos_bank(names)
        if isinstance(c, dict):
            c = c.get("positions", c.get("coords", c.get("last_hidden_state")))
        return c.squeeze(0).to(self.device).float() if c.dim() == 3 else c.to(self.device).float()

    def fit(self, X, y, meta) -> None:
        task_name = meta[0]["task_name"]

        # 1. Dataset Loading (matching LaBraM exact args)
        dataset_train = make_dataset_2(X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache = True)

        # 2. Safety Check: If dataset is empty, the .h5 cache is likely bad
        if len(dataset_train) == 0:
            print("[Warning] Dataset empty. Retrying without cache...")
            dataset_train = make_dataset_2(X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache = False)

        # 3. Validation Split (aligned with BCI: 15%)
        val_split = 0.15
        dataset_train, dataset_val = dataset_train.split_train_val(val_split)

        # 4. DataLoader Setup
        bs = 64 if self.chunk_len_s else 1
        train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=bs, shuffle=False)

        # 5. Training Setup (aligned with BCI)
        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer and Scheduler (matching BCI setup)
        max_epochs = 30
        steps_per_epoch = math.ceil(len(train_loader))
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

        # Early stopping setup (matching BCI)
        patience = 10
        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None

        coords_train = self._coords(dataset_train.ch_names)
        coords_val = self._coords(dataset_val.ch_names)

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            total_acc_samples = 0
            for x, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False):
                x, yb = x.to(self.device), yb.to(self.device)
                cb = coords_train.unsqueeze(0).expand(x.size(0), -1, -1)

                optimizer.zero_grad()
                logits = self.model(x, cb)
                loss = self.model.loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                if logits.dim() == 2:
                    preds = torch.argmax(logits, dim=1)
                    target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total_acc_samples += x.size(0)

                # Manual memory cleanup like LaBraM
                del x, yb, logits; torch.cuda.empty_cache()

            # Compute train metrics
            train_loss = total_loss / total_samples if total_samples else 0.0
            train_acc = correct / total_acc_samples if total_acc_samples else 0.0

            # Validation
            val_loss = 0.0
            val_samples = 0
            val_correct = 0
            val_acc_samples = 0
            self.model.eval()
            with torch.no_grad():
                for x, yb, _ in tqdm(val_loader, desc=f"Val {epoch}/{max_epochs}", leave=False):
                    x, yb = x.to(self.device), yb.to(self.device)
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
                    del x, yb, logits; torch.cuda.empty_cache()

            # Compute val metrics
            avg_val_loss = val_loss / val_samples if val_samples else 0.0
            val_acc = val_correct / val_acc_samples if val_acc_samples else 0.0

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging (wandb or console)
            current_lr = scheduler.get_last_lr()[0]
            metrics = {
                f"{self.name}/train_loss": train_loss,
                f"{self.name}/train_acc": train_acc,
                f"{self.name}/val_loss": avg_val_loss,
                f"{self.name}/val_acc": val_acc,
                f"{self.name}/lr": current_lr,
            }

            if self.wandb_run:
                wandb_utils.log(metrics, step=epoch)

            # Always print to console for visibility
            print(f"[Epoch {epoch:02d}/{max_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"lr={current_lr:.2e} patience={patience_counter}/{patience}")

            # Early stopping trigger
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            


    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        task_name = meta[0]["task_name"]
        
        
        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name, 
            chunk_len_s=self.chunk_len_s,
            is_train=False, 
            use_cache = True
        )
        
        if len(dataset_test) == 0:
            return np.array([])

        if self.chunk_len_s is None:
            batch_size = 1
        else: 
            batch_size = 64
            
        loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        coords = self._coords(dataset_test.ch_names)
        self.model.eval()

        preds_all = []
        idx_map_all = []

        for batch in tqdm(loader, desc="Predicting"):
            x, idx, _ = batch
            x = x.to(self.device)
            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
            
            logits = self.model(x, cb) #forward insteadhere?    
            # logits = self.
            
            # Get window-level predictions
            pred = torch.argmax(logits, dim=1)
            preds_all.append(pred.cpu().numpy())
            idx_map_all.append(idx.cpu().numpy())



        preds = np.concatenate(preds_all)
        idx_map = np.concatenate(idx_map_all)


        # Majority voting: Combine windows back into 1 patient prediction
        unique_indices = np.unique(idx_map)
        final_predictions = []
        for i in unique_indices:
            patient_votes = preds[idx_map == i]
            final_predictions.append(Counter(patient_votes).most_common(1)[0][0])

        return np.array([map_label_reverse(p, task_name) for p in final_predictions])
