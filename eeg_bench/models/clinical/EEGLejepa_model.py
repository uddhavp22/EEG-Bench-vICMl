# lejepa_clinical_model.py

from __future__ import annotations
from typing import List, Dict, Optional, cast, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from tqdm import tqdm
import pickle
from collections import Counter
import gc
import math
import logging
import hashlib
import json
import bisect
from pathlib import Path
from ..abstract_model import AbstractModel
from .. import lejepa_utils as _lejepa_utils
from ..lejepa_utils import (
    EMBED_CACHE_VERSION,
    _setup_eegfm_imports,
    build_simple_probe_head,
    extract_layer_cls,
    probe_layer_suffix,
)
from ...config import get_config_value, LeJEPAConfig, _resolve_probe_layer_idx

# LaBraM Clinical Utilities
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from .LaBraM import utils

from transformers import AutoModel
from ...utils import wandb_utils
from ...utils.eeg_noise import apply_eeg_noise

logger = logging.getLogger(__name__)

    # def __init__(self, dim: int, out_dim: int):
    #     super().__init__()
    #     self.attn = nn.Linear(dim, 1, bias=False)
    #     self.norm = nn.LayerNorm(dim)
    #     self.fc = nn.Linear(dim, out_dim)

    # def _pool(self, x: torch.Tensor) -> torch.Tensor:
    #     # x: (B, S, D)
    #     dim = x.size(-1)
    #     scores = self.attn(x).squeeze(-1)
    #     weights = torch.softmax(scores / math.sqrt(dim), dim=1)

    #     return torch.einsum("bs,bsd->bd", weights, x)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # x: (B, S, D) or (B, n_chunks, S, D)
    #     if x.dim() == 4:
    #         bsz, n_chunks, seq_len, dim = x.shape
    #         x = x.reshape(bsz * n_chunks, seq_len, dim)
    #         pooled = self._pool(x).view(bsz, n_chunks, dim).mean(dim=1)
    #     elif x.dim() == 3:
    #         pooled = self._pool(x)
    #     else:
    #         raise ValueError(f"Unexpected attentive probe input shape: {tuple(x.shape)}")
    #     pooled = self.norm(pooled)
    #     return self.fc(pooled)

class AttentiveProbe(nn.Module):
    def __init__(self, dim: int, out_dim: int, use_mlp: bool = False, dropout: float = 0.4):
        super().__init__()

        # Single learned query
        self.query = nn.Parameter(torch.randn(1, 1, dim))

        # Value projection (linear here, but could be identity?)
        self.value_proj = nn.Linear(dim, dim)
        self.value_dropout = nn.Dropout(dropout)

        # Normalization after pooling
        self.norm = nn.LayerNorm(dim)
        self.out_dropout = nn.Dropout(dropout)

        # Output head
        if use_mlp:
            self.head = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, out_dim),
            )
        else:
            self.head = nn.Linear(dim, out_dim)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D)
        returns: (B, D)
        """
        B, S, D = x.shape

        # Expand query per batch
        q = self.query.expand(B, -1, -1)  # (B, 1, D)

        # Scaled dot-product attention
        scores = torch.einsum("bqd,bsd->bqs", q, x) / math.sqrt(D)
        weights = torch.softmax(scores, dim=-1)  # (B, 1, S)

        values = self.value_dropout(self.value_proj(x))
        pooled = torch.einsum("bqs,bsd->bqd", weights, values)

        return pooled.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D) or (B, n_chunks, S, D)
        """
        if x.dim() == 4:
            B, n_chunks, S, D = x.shape
            x = x.view(B * n_chunks, S, D)
            pooled = self._pool(x).view(B, n_chunks, D).mean(dim=1)
        elif x.dim() == 3:
            pooled = self._pool(x)
        else:
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

        pooled = self.norm(pooled)
        pooled = self.out_dropout(pooled)
        return self.head(pooled)


def build_probe_head(dim: int, out_dim: int, probe_head: str) -> nn.Module:
    head = build_simple_probe_head(dim, out_dim, probe_head)
    if head is None:  # probe_head == "attentive"
        return AttentiveProbe(dim, out_dim)
    return head

class ShardedEmbeddingDataset(Dataset):
    def __init__(self, index_path: Path):
        with open(index_path, "r") as f:
            index = json.load(f)

        self.shards = []
        self.cum_counts = []
        total = 0

        for shard in index.get("shards", []):
            emb = np.load(shard["embeddings"], mmap_mode="r")
            lbl = np.load(shard["labels"], mmap_mode="r")
            count = int(shard.get("count", len(emb)))
            self.shards.append(
                {
                    "emb": emb,
                    "lbl": lbl,
                    "count": count,
                    "sequence_ids": shard.get("sequence_ids"),
                }
            )
            total += count
            self.cum_counts.append(total)

    def __len__(self) -> int:
        return self.cum_counts[-1] if self.cum_counts else 0

    def __getitem__(self, idx: int):
        shard_idx = bisect.bisect_left(self.cum_counts, idx + 1)
        prev = 0 if shard_idx == 0 else self.cum_counts[shard_idx - 1]
        local_idx = idx - prev
        shard = self.shards[shard_idx]
        emb = shard["emb"][local_idx]
        lbl = shard["lbl"][local_idx]
        return torch.from_numpy(emb), torch.from_numpy(lbl)

    def get_sequence_id(self, idx: int) -> Optional[Any]:
        shard_idx = bisect.bisect_left(self.cum_counts, idx + 1)
        prev = 0 if shard_idx == 0 else self.cum_counts[shard_idx - 1]
        local_idx = idx - prev
        seq_ids = self.shards[shard_idx].get("sequence_ids")
        if seq_ids is None:
            return None
        return seq_ids[local_idx]

class MemmapEmbeddingDataset(Dataset):
    def __init__(self, emb_path: Path, lbl_path: Path, sequence_ids: Optional[List[Any]] = None):
        self.emb = np.load(emb_path, mmap_mode="r")
        self.lbl = np.load(lbl_path, mmap_mode="r")
        self.sequence_ids = sequence_ids

    def __len__(self) -> int:
        return len(self.emb)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.emb[idx]), torch.from_numpy(self.lbl[idx])

    def get_sequence_id(self, idx: int) -> Optional[Any]:
        if self.sequence_ids is None:
            return None
        return self.sequence_ids[idx]

class ConcreteLeJEPAClinical(nn.Module):
    def __init__(
        self,
        num_classes,
        num_labels_per_chunk,
        base_path=None,
        version=None,
        freeze_encoder=True,
        config_path: Optional[Path] = None,
        pretrained_path: Optional[Path] = None,
        probe_head: str = "linear",
    ):
        super().__init__()

        self.is_multilabel_task = num_labels_per_chunk is not None
        self.probe_head = probe_head
        self.attentive_probe = probe_head == "attentive"

        # ------------------------------------------------------------
        # Pretrained config / checkpoint resolution (SAFE)
        # ------------------------------------------------------------
        if (config_path is None or pretrained_path is None) and base_path is not None and version is not None:
            base_path = Path(base_path) / f"version_{version}"

            candidate_config = base_path / "config" / "config.pkl"
            candidate_ckpt = base_path / "checkpoints" / "last.ckpt"

            if config_path is None and candidate_config.exists():
                config_path = candidate_config
            elif config_path is None:
                print(f"[LeJEPAClinical] No config found at {candidate_config}. Using default config.")

            if pretrained_path is None and candidate_ckpt.exists():
                pretrained_path = candidate_ckpt
            elif pretrained_path is None:
                print(f"[LeJEPAClinical] No checkpoint found at {candidate_ckpt}. Training from scratch.")

        # ------------------------------------------------------------
        # Build model config
        # ------------------------------------------------------------
        if config_path is not None:
            with open(config_path, "rb") as f:
                pretrain_config = pickle.load(f)
                pretrain_config['model']['name'] = 'EEGLEJEPA' #force for MAE.
            cfg = _lejepa_utils.EEGLEJEPAConfig(**pretrain_config["model"])
            print("Loaded Config!")
        else:
            raise 

        # ------------------------------------------------------------
        # Build backbone
        # ------------------------------------------------------------
        self.backbone = cfg.build()
        self.chunk_length = 4000 #16s chunks!
        DIM = self.backbone.dim
        # DIM = self.backbone.proj_dim


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
            print(f"[LeJEPAClinical] Checkpoint keys (first 5): {list(ckpt_keys)[:5]}")
            print(f"[LeJEPAClinical] Model keys (first 5): {list(model_keys)[:5]}")

            # Load with strict=False but capture missing/unexpected
            load_result = self.backbone.load_state_dict(state, strict=False)
            missing_keys = load_result.missing_keys
            unexpected_keys = load_result.unexpected_keys

            matched_keys = model_keys & ckpt_keys
            print(f"[LeJEPAClinical] Matched keys: {len(matched_keys)}/{len(model_keys)}")
            print(f"[LeJEPAClinical] Missing keys: {len(missing_keys)}")
            print(f"[LeJEPAClinical] Unexpected keys: {len(unexpected_keys)}")

            if len(missing_keys) > 0:
                print(f"[LeJEPAClinical] WARNING: Missing keys (first 5): {missing_keys[:5]}")
            if len(unexpected_keys) > 0:
                print(f"[LeJEPAClinical] WARNING: Unexpected keys (first 5): {unexpected_keys[:5]}")
            if len(matched_keys) == 0:
                print(f"[LeJEPAClinical] CRITICAL: No keys matched! Checkpoint may have wrong format.")

            print(f"[LeJEPAClinical] Loaded pretrained weights from {pretrained_path}")

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
        self.head = build_probe_head(DIM, out_dim, self.probe_head)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def _get_logits_from_outputs(self, outputs: Dict[str, torch.Tensor], batch_size: int, n_chunks: int) -> torch.Tensor:
        if self.attentive_probe:
            seq = outputs["sequence_embeddings"]
            seq_len = seq.shape[1]
            embed_dim = seq.shape[2]
            seq = seq.view(batch_size, n_chunks, seq_len, embed_dim)
            logits = self.head(seq)
        else:
            cls = outputs["cls_token"]
            embedding_dim = cls.shape[1]
            cls = cls.view(batch_size, n_chunks, embedding_dim)
            if cls.dim() == 3:
                cls = cls.mean(dim=1)
            logits = self.head(cls)
        return logits

    def forward(self, x, coords, probe_layer_idx: Optional[int] = None):

        B, C, T = x.shape
        n_chunks = T // self.chunk_length

        # Handle case where data is shorter than one chunk
        if n_chunks == 0:
            # Pad to chunk_length if too short
            pad_length = self.chunk_length - T
            x = torch.nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
            n_chunks = 1
            T = self.chunk_length

        chunk_trunc = n_chunks * self.chunk_length
        x = x[:, :, :chunk_trunc]

        # Reshape into segments:
        x = x.view(B, C, n_chunks, self.chunk_length)
        # Permute to (batch_size, num_chunks, n_channels, chunk_length)
        x = x.permute(0, 2, 1, 3)
        # Merge batch and chunk dimensions for efficient processing:
        x = x.reshape(B * n_chunks, C, self.chunk_length)

        # FIX: Expand coords to match chunked batch dimension
        # coords shape: (B, C, 3) -> (B*n_chunks, C, 3)
        coords = coords.unsqueeze(1).expand(-1, n_chunks, -1, -1).reshape(B * n_chunks, C, 3)

        if probe_layer_idx is not None:
            cls = extract_layer_cls(self.backbone, x, coords, probe_layer_idx)
            embedding_dim = cls.shape[1]
            cls = cls.view(B, n_chunks, embedding_dim).mean(dim=1)
            logits = self.head(cls)
        else:
            outputs = self.backbone.forward_downstream(x=x, channel_locations=coords)
            logits = self._get_logits_from_outputs(outputs, B, n_chunks)
        if self.is_multilabel_task:
            logits = logits.reshape(B, self.num_classes, -1)
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
        self.freeze_encoder = freeze_encoder  # Store for use in fit()
        self.probe_head = "linear"
        self.attentive_probe = False
        self.probe_layer = None
        self.probe_layer_idx = None

        # Handle config vs legacy parameters
        if config is not None:
            # Use checkpoint path from config (get_checkpoint_path resolves full_path vs base_path+version)
            checkpoint_path = config.get_checkpoint_path()
            if checkpoint_path:
                # Extract base_path and version from full path for config discovery
                ckpt_path = Path(checkpoint_path)
                self.pretrained_path = ckpt_path  # Store as instance variable
                config_path = None
                if ckpt_path.parent.name == "checkpoints":
                    version_dir = ckpt_path.parent.parent
                    if version_dir.name.startswith("version_"):
                        base_path = str(version_dir.parent)
                        version = int(version_dir.name.replace("version_", ""))
                        candidate_config = version_dir / "config" / "config.pkl"
                        if candidate_config.exists():
                            config_path = candidate_config
                    else:
                        base_path = None
                        version = None
                else:
                    base_path = None
                    version = None
            else:
                base_path = config.checkpoint_base_path
                version = config.checkpoint_version
                config_path = None
                self.pretrained_path = None  # Store as instance variable
            freeze_encoder = config.freeze_encoder
            self.freeze_encoder = freeze_encoder
            self.probe_head = config.probe_head
            self.attentive_probe = self.probe_head == "attentive"
            self.probe_layer = config.probe_layer
            pos_bank_path = config.pos_bank_path
            eegfm_path = config.eegfm_path
        else:
            # Legacy mode - use parameters directly (with old defaults if not provided)
            lejepa_config = get_config_value("lejepa", {})
            pos_bank_path = lejepa_config.get("pos_bank_path", "./REVE_posbank")
            eegfm_path = lejepa_config.get("eegfm_path")
            self.probe_head = lejepa_config.get("probe_head", "linear")
            if "probe_head" not in lejepa_config and lejepa_config.get("attentive_probe", False):
                self.probe_head = "attentive"
            self.attentive_probe = self.probe_head == "attentive"
            config_path = None
            self.pretrained_path = None  # Store as instance variable

        # Setup eegfm imports
        _setup_eegfm_imports(eegfm_path)

        # Load position bank with HuggingFace fallback
        self.pos_bank = self._load_position_bank(pos_bank_path)

        self.model = ConcreteLeJEPAClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            base_path=base_path,
            version=version,
            freeze_encoder=freeze_encoder,
            config_path=config_path,
            pretrained_path=self.pretrained_path,
            probe_head=self.probe_head,
        ).to(self.device)
        depth = self.model.backbone.config.encoder_config.depth
        self.probe_layer_idx = _resolve_probe_layer_idx(self.probe_layer, depth)
        self._eval_noise_config: Optional[Dict] = None

    def _get_probe_optimizer_defaults(self) -> Dict[str, float]:
        if self.probe_head == "attentive":
            return {"max_lr": 3e-4, "weight_decay": 0.1, "patience": 5}
        if self.probe_head == "mlp":
            return {"max_lr": 1e-3, "weight_decay": 0.05, "patience": 6}
        return {"max_lr": 4e-4, "weight_decay": 0.01, "patience": 10}

    def set_eval_noise_config(self, config: Optional[Dict]) -> None:
        """Set evaluation-time noise configuration (opt-in)."""
        self._eval_noise_config = config

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
            logger.info("Successfully got hub position bank!")
            return pos_bank

    def _coords(self, ch_names):
        """
        Get 3D coordinates for channel names.
        Handles bipolar channels (e.g., 'C3-P3') by computing midpoint of the two electrodes.
        """
        names = [c.replace("EEG", "").strip() for c in ch_names]
        
        # Collect all unique electrode names (split bipolar channels)
        all_electrodes = set()
        for name in names:
            if '-' in name:
                parts = name.split('-')
                all_electrodes.update(parts)
            else:
                all_electrodes.add(name)
        
        # Query position bank once for all electrodes
        all_electrodes = list(all_electrodes)
        try:
            c = self.pos_bank(all_electrodes)
            if isinstance(c, dict):
                c = c.get("positions", c.get("coords", c.get("last_hidden_state")))
            if c.dim() == 3:
                c = c.squeeze(0)
            coords_dict = {name: c[i] for i, name in enumerate(all_electrodes)}
        except Exception as e:
            logger.warning(f"Position bank error: {e}. Using zeros.")
            return torch.zeros(len(names), 3, device=self.device, dtype=torch.float32)
        
        # Build output: single electrodes directly, bipolar as midpoints
        output = []
        for name in names:
            if '-' in name:
                e1, e2 = name.split('-')[:2]
                output.append((coords_dict.get(e1, torch.zeros(3)) + coords_dict.get(e2, torch.zeros(3))) / 2)
            else:
                output.append(coords_dict.get(name, torch.zeros(3)))
        
        return torch.stack(output).to(self.device).float()

    @torch.no_grad()
    def _extract_embeddings_clinical(self, dataloader, coords):
        """Extract averaged embeddings from frozen encoder (handles chunking internally)."""
        self.model.backbone.eval()
        embeddings_list = []
        labels_list = []

        chunk_length = self.model.chunk_length  # 4000 samples = 20s at 200Hz
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
                    x, yb, batch_coords = batch  # third element is channels, not needed
                    x = x.to(self.device)
                    B, C, T = x.shape

                    # Handle chunking (same logic as ConcreteLeJEPAClinical.forward)
                    n_chunks = T // chunk_length
                    if n_chunks == 0:
                        pad_length = chunk_length - T
                        x = torch.nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
                        n_chunks = 1
                        T = chunk_length

                    chunk_trunc = n_chunks * chunk_length
                    x = x[:, :, :chunk_trunc]

                    # Reshape into chunks
                    x = x.view(B, C, n_chunks, chunk_length)
                    x = x.permute(0, 2, 1, 3)
                    x = x.reshape(B * n_chunks, C, chunk_length)

                    #path for bipolar stuff
                    if x.shape[1] != coords.shape[0]: # mismatch due to bipolar channels
                        #stack batch_coords tuple to get batch channel names
                        batch_coords = [ch_name[0] for ch_name in batch_coords]
                        coords = self._coords(batch_coords).to(self.device)

                    # Expand coords for all chunks
                    cb = coords.unsqueeze(0).unsqueeze(1).expand(B, n_chunks, -1, -1)
                    cb = cb.reshape(B * n_chunks, C, 3)

                    # Forward through backbone
                    if self.probe_layer_idx is not None:
                        cls = extract_layer_cls(self.model.backbone, x, cb, self.probe_layer_idx)
                        embedding_dim = cls.shape[1]
                        cls = cls.view(B, n_chunks, embedding_dim).mean(dim=1)  # (B, dim)
                        embeddings_list.append(cls.cpu())
                    elif self.attentive_probe:
                        outputs = self.model.backbone.forward_downstream(x=x, channel_locations=cb)
                        seq = outputs["sequence_embeddings"]
                        seq_len = seq.shape[1]
                        embed_dim = seq.shape[2]
                        seq = seq.view(B, n_chunks, seq_len, embed_dim)
                        embeddings_list.append(seq.cpu())
                    else:
                        outputs = self.model.backbone.forward_downstream(x=x, channel_locations=cb)
                        cls = outputs["cls_token"]
                        embedding_dim = cls.shape[1]
                        cls = cls.view(B, n_chunks, embedding_dim).mean(dim=1)  # (B, dim)
                        embeddings_list.append(cls.cpu())
                    # Handle different label formats (match LaBraM behavior)
                    if not self.model.is_multilabel_task and yb.dim() > 1:
                        labels_list.append(yb.argmax(dim=1).cpu())
                    else:
                        labels_list.append(yb.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return embeddings, labels

    @torch.no_grad()
    def _extract_embeddings_clinical_stream(self, dataloader, coords, index_path: Path, shard_prefix: Path, shard_size: int = 1000) -> Dataset:
        """Stream embeddings/labels to sharded .npy files to avoid large in-memory concatenation."""
        self.model.backbone.eval()
        write_idx = 0
        shard_idx = 0
        shards = []
        rec_names = getattr(dataloader.dataset, "recording_names", None)
        emb_buffer = []
        lbl_buffer = []
        seq_buffer = []

        chunk_length = self.model.chunk_length
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
                    x, yb, batch_coords = batch
                    x = x.to(self.device)
                    B, C, T = x.shape

                    n_chunks = T // chunk_length
                    if n_chunks == 0:
                        pad_length = chunk_length - T
                        x = torch.nn.functional.pad(x, (0, pad_length), mode='constant', value=0)
                        n_chunks = 1
                        T = chunk_length

                    chunk_trunc = n_chunks * chunk_length
                    x = x[:, :, :chunk_trunc]

                    x = x.view(B, C, n_chunks, chunk_length)
                    x = x.permute(0, 2, 1, 3)
                    x = x.reshape(B * n_chunks, C, chunk_length)

                    if x.shape[1] != coords.shape[0]:
                        batch_coords = [ch_name[0] for ch_name in batch_coords]
                        coords = self._coords(batch_coords).to(self.device)

                    cb = coords.unsqueeze(0).unsqueeze(1).expand(B, n_chunks, -1, -1)
                    cb = cb.reshape(B * n_chunks, C, 3)

                    if self.probe_layer_idx is not None:
                        cls = extract_layer_cls(self.model.backbone, x, cb, self.probe_layer_idx)
                        embedding_dim = cls.shape[1]
                        cls = cls.view(B, n_chunks, embedding_dim).mean(dim=1)
                        embeddings = cls.cpu().numpy()
                    else:
                        outputs = self.model.backbone.forward_downstream(x=x, channel_locations=cb)
                        if self.attentive_probe:
                            seq = outputs["sequence_embeddings"]
                            seq_len = seq.shape[1]
                            embed_dim = seq.shape[2]
                            seq = seq.view(B, n_chunks, seq_len, embed_dim)
                            embeddings = seq.cpu().numpy()
                        else:
                            cls = outputs["cls_token"]
                            embedding_dim = cls.shape[1]
                            cls = cls.view(B, n_chunks, embedding_dim).mean(dim=1)
                            embeddings = cls.cpu().numpy()

                    if not self.model.is_multilabel_task and yb.dim() > 1:
                        labels = yb.argmax(dim=1).cpu().numpy()
                    else:
                        labels = yb.cpu().numpy()

                    end_idx = write_idx + embeddings.shape[0]
                    if rec_names is not None:
                        seq_ids = rec_names[write_idx:end_idx]
                    else:
                        seq_ids = [f"sample_{i:08d}" for i in range(write_idx, end_idx)]

                    emb_buffer.append(embeddings)
                    lbl_buffer.append(labels)
                    seq_buffer.extend(seq_ids)
                    write_idx = end_idx

                    buffer_count = sum(arr.shape[0] for arr in emb_buffer)
                    if buffer_count >= shard_size:
                        shard_emb_path = Path(f"{shard_prefix}.part{shard_idx:04d}.embeddings.npy")
                        shard_lbl_path = Path(f"{shard_prefix}.part{shard_idx:04d}.labels.npy")
                        shard_emb = np.concatenate(emb_buffer, axis=0)
                        shard_lbl = np.concatenate(lbl_buffer, axis=0)
                        np.save(shard_emb_path, shard_emb)
                        np.save(shard_lbl_path, shard_lbl)

                        shards.append(
                            {
                                "embeddings": str(shard_emb_path),
                                "labels": str(shard_lbl_path),
                                "count": int(shard_emb.shape[0]),
                                "sequence_ids": list(seq_buffer),
                            }
                        )
                        emb_buffer.clear()
                        lbl_buffer.clear()
                        seq_buffer.clear()
                        shard_idx += 1

        if emb_buffer:
            shard_emb_path = Path(f"{shard_prefix}.part{shard_idx:04d}.embeddings.npy")
            shard_lbl_path = Path(f"{shard_prefix}.part{shard_idx:04d}.labels.npy")
            shard_emb = np.concatenate(emb_buffer, axis=0)
            shard_lbl = np.concatenate(lbl_buffer, axis=0)
            np.save(shard_emb_path, shard_emb)
            np.save(shard_lbl_path, shard_lbl)
            shards.append(
                {
                    "embeddings": str(shard_emb_path),
                    "labels": str(shard_lbl_path),
                    "count": int(shard_emb.shape[0]),
                    "sequence_ids": list(seq_buffer),
                }
            )

        index = {
            "shards": shards,
            "meta": {
                "chunk_len_s": self.chunk_len_s,
                "num_labels_per_chunk": self.num_labels_per_chunk,
                "attentive_probe": True,
                "shard_size": shard_size,
            },
        }
        with open(index_path, "w") as f:
            json.dump(index, f)

        return ShardedEmbeddingDataset(index_path)

    def _train_epoch_cached(self, dataloader, optimizer):
        """Train only the head on cached embeddings."""
        self.model.head.train()
        running_loss = 0.0
        running_corrects = 0
        total_loss_samples = 0
        total_acc_samples = 0

        for embeddings, y_batch in dataloader:
            embeddings = embeddings.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            logits = self.model.head(embeddings)
            if self.model.is_multilabel_task:
                logits = logits.view(embeddings.size(0), self.model.num_classes, -1)
            y_batch = y_batch.long()
            loss = self.model.loss_fn(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.head.parameters(), max_norm=1.0)
            optimizer.step()
            # NOTE: Don't step scheduler here - CosineAnnealingLR is epoch-based

            running_loss += loss.item() * embeddings.size(0)
            total_loss_samples += embeddings.size(0)

            preds = torch.argmax(logits, dim=1)
            if self.model.is_multilabel_task:
                running_corrects += (preds == y_batch).sum().item()
                total_acc_samples += y_batch.numel()
            else:
                running_corrects += (preds == y_batch).sum().item()
                total_acc_samples += embeddings.size(0)

        epoch_loss = running_loss / total_loss_samples if total_loss_samples else 0.0
        epoch_acc = running_corrects / total_acc_samples if total_acc_samples else 0.0
        return epoch_loss, epoch_acc

    def _validate_epoch_cached(self, dataloader):
        """Validate on cached embeddings."""
        self.model.head.eval()
        running_loss = 0.0
        running_corrects = 0
        total_loss_samples = 0
        total_acc_samples = 0

        with torch.no_grad():
            for embeddings, y_batch in dataloader:
                embeddings = embeddings.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model.head(embeddings)
                if self.model.is_multilabel_task:
                    logits = logits.view(embeddings.size(0), self.model.num_classes, -1)
                y_batch = y_batch.long()
                loss = self.model.loss_fn(logits, y_batch)

                running_loss += loss.item() * embeddings.size(0)
                total_loss_samples += embeddings.size(0)

                preds = torch.argmax(logits, dim=1)
                if self.model.is_multilabel_task:
                    running_corrects += (preds == y_batch).sum().item()
                    total_acc_samples += y_batch.numel()
                else:
                    running_corrects += (preds == y_batch).sum().item()
                    total_acc_samples += embeddings.size(0)

        epoch_loss = running_loss / total_loss_samples if total_loss_samples else 0.0
        epoch_acc = running_corrects / total_acc_samples if total_acc_samples else 0.0
        return epoch_loss, epoch_acc

    def fit(self, X, y, meta, data_percentage: float = 1.0) -> None:
        task_name = meta[0]["task_name"]

        # 1. Dataset Loading (matching LaBraM exact args)
        dataset_train = make_dataset_2(
            X, y, meta, task_name, self.name, 
            chunk_len_s=self.chunk_len_s,
            is_train=True, 
            use_cache=True,
            sfreq=250
        )
        

        # 2. Safety Check: If dataset is empty, the .h5 cache is likely bad
        if len(dataset_train) == 0:
            print("[Warning] Dataset empty. Retrying without cache...")
            dataset_train = make_dataset_2(
                X, y, meta, task_name, self.name,
                chunk_len_s=self.chunk_len_s,
                is_train=True,
                use_cache=False,
                sfreq=250
            )

        # 3. Validation Split (aligned with BCI: 15%)
        val_split = 0.2
        dataset_train, dataset_val = dataset_train.split_train_val(val_split)



        # 4. DataLoader Setup
        bs = 64 if self.chunk_len_s else 1
        train_loader = DataLoader(dataset_train, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)  # shuffle=False for caching

        has_val = dataset_val is not None
        val_loader = DataLoader(dataset_val, batch_size=bs, shuffle=False) if has_val else None

        # 5. Training Setup (aligned with BCI)
        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        max_epochs = 30
        probe_hparams = self._get_probe_optimizer_defaults()
        patience = int(probe_hparams["patience"])

        coords_train = self._coords(dataset_train.ch_names)

        coords_val = self._coords(dataset_val.ch_names) if has_val else None

        if self.freeze_encoder:
            # =============================================
            # CACHED EMBEDDINGS PATH (frozen encoder)
            # =============================================
            print("[LeJEPAClinical] Using cached embeddings (freeze_encoder=True)")

            # Compute dataset hash for caching
            dataset_hash = self._compute_dataset_hash(X, meta)
            checkpoint_path = str(self.pretrained_path) if self.pretrained_path else "no_ckpt"

            # Load or extract embeddings (cached to disk)
            train_emb, train_lbl = self._load_or_extract_embeddings(
                train_loader, coords_train, checkpoint_path, task_name, dataset_hash, "train"
            )

            if has_val:
                val_emb, val_lbl = self._load_or_extract_embeddings(
                    val_loader, coords_val, checkpoint_path, task_name, dataset_hash, "val"
                )

            if self.attentive_probe and isinstance(train_emb, Dataset):
                cached_train_dataset = train_emb
                cached_val_dataset = val_emb if has_val else None

                if data_percentage < 1.0:
                    train_indices = self._subsample_indices(len(cached_train_dataset), data_percentage)
                    cached_train_dataset = Subset(cached_train_dataset, train_indices)
                    print(f"[LeJEPAClinical] Subsampled to {len(cached_train_dataset)} samples ({data_percentage*100:.0f}%)")
            else:
                # Apply data percentage subsampling (deterministic)
                if data_percentage < 1.0:
                    train_indices = self._subsample_indices(len(train_emb), data_percentage)
                    train_emb = train_emb[train_indices]
                    train_lbl = train_lbl[train_indices]
                    print(f"[LeJEPAClinical] Subsampled to {len(train_emb)} samples ({data_percentage*100:.0f}%)")

                cached_train_dataset = TensorDataset(train_emb, train_lbl)
                cached_val_dataset = TensorDataset(val_emb, val_lbl) if has_val else None

            val_count = len(cached_val_dataset) if cached_val_dataset is not None else 0
            print(f"[LeJEPAClinical] Using {len(cached_train_dataset)} train and {val_count} val embeddings")

            # Use larger batch size for cached training (no encoder memory needed)
            cached_batch_size = bs * 32  if not self.attentive_probe else 32 # 256 for chunked, 4 for full recordings
            cached_train_loader = DataLoader(cached_train_dataset, batch_size=cached_batch_size, shuffle=True, pin_memory=True, num_workers = 4)
            cached_val_loader = DataLoader(cached_val_dataset, batch_size=cached_batch_size, shuffle=False, pin_memory=True, num_workers = 4) if has_val else None

            steps_per_epoch = math.ceil(len(train_loader))

            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            
            max_lr = probe_hparams["max_lr"]
            weight_decay = probe_hparams["weight_decay"]
            optimizer = optim.AdamW(trainable_params, lr=max_lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=1e-6
            )
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     optimizer,
            #     max_lr=max_lr,
            #     steps_per_epoch=steps_per_epoch,
            #     epochs=max_epochs,
            #     pct_start=0.1,
            # )

            
            patience_counter = 0
            best_val_loss = float("inf")
            best_model_state = None
            for epoch in range(1, max_epochs + 1):
                train_loss, train_acc = self._train_epoch_cached(cached_train_loader, optimizer)
                if has_val:
                    val_loss, val_acc = self._validate_epoch_cached(cached_val_loader)
                else:
                    val_loss, val_acc = None, None
                
                # Step scheduler once per epoch (CosineAnnealingLR is epoch-based)
                scheduler.step()

                if has_val:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1

                current_lr = scheduler.get_last_lr()[0]

                if self.wandb_run:
                    metrics = {
                        f"{self.name}/train_loss": train_loss,
                        f"{self.name}/train_acc": train_acc,
                        f"{self.name}/lr": current_lr,
                    }
                    if has_val:
                        metrics.update({
                            f"{self.name}/val_loss": val_loss,
                            f"{self.name}/val_acc": val_acc,
                        })
                    wandb_utils.log(metrics, step=epoch)

                if has_val:
                    print(f"[Epoch {epoch:02d}/{max_epochs}] "
                          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                          f"lr={current_lr:.2e} patience={patience_counter}/{patience}")
                else:
                    print(f"[Epoch {epoch:02d}/{max_epochs}] "
                          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                          f"lr={current_lr:.2e}")

                if has_val and patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    break

            if has_val and best_model_state is not None:
                self.model.load_state_dict(best_model_state)

        else:
            # =============================================
            # FULL FORWARD PASS PATH (fine-tuning encoder)
            # =============================================
            print("[LeJEPAClinical] Using full forward pass (freeze_encoder=False)")

            steps_per_epoch = math.ceil(len(train_loader))
            max_lr = probe_hparams["max_lr"]
            weight_decay = probe_hparams["weight_decay"]

            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.AdamW(trainable_params, lr=1e-6, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=max_epochs,
                pct_start=0.1,
            )

            patience_counter = 0
            best_val_loss = float("inf")
            best_model_state = None

            for epoch in range(1, max_epochs + 1):
                # Only set head to train mode; preserve backbone eval mode if frozen
                self.model.head.train()
                self.model.backbone.train()
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

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

                if has_val:
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
                else:
                    avg_val_loss, val_acc = None, None

                # Logging (wandb or console)
                current_lr = scheduler.get_last_lr()[0]
                metrics = {
                    f"{self.name}/train_loss": train_loss,
                    f"{self.name}/train_acc": train_acc,
                    f"{self.name}/lr": current_lr,
                }
                if has_val:
                    metrics.update({
                        f"{self.name}/val_loss": avg_val_loss,
                        f"{self.name}/val_acc": val_acc,
                    })

                if self.wandb_run:
                    wandb_utils.log(metrics, step=epoch)

                # Always print to console for visibility
                if has_val:
                    print(f"[Epoch {epoch:02d}/{max_epochs}] "
                          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                          f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} | "
                          f"lr={current_lr:.2e} patience={patience_counter}/{patience}")
                else:
                    print(f"[Epoch {epoch:02d}/{max_epochs}] "
                          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                          f"lr={current_lr:.2e}")

                # Early stopping trigger
                if has_val and patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                    break

            # Restore best model
            if has_val and best_model_state is not None:
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

        noise_cfg = self._eval_noise_config or {}
        noise_types = noise_cfg.get("noise_types") or []
        snr_db = noise_cfg.get("snr_db")
        has_noise = bool(noise_types) and snr_db is not None

        if has_noise:
            sfreq = float(
                noise_cfg.get("sfreq")
                or meta[0].get("sampling_frequency")
                or getattr(dataset_test, "sfreq", 200)
            )
            channel_dropout_prob = float(noise_cfg.get("channel_dropout_prob", 0.0))
            one_over_f_band = tuple(noise_cfg.get("one_over_f_band", (0.5, 40.0)))
            emg_band = tuple(noise_cfg.get("emg_band", (30.0, 100.0)))
            noise_seed = noise_cfg.get("seed")
        else:
            sfreq = 200.0
            channel_dropout_prob = 0.0
            one_over_f_band = (0.5, 40.0)
            emg_band = (30.0, 100.0)
            noise_seed = None

        preds_all = []
        idx_map_all = []

        for batch_idx, batch in enumerate(tqdm(loader, desc="Predicting")):
            x, idx, batch_coords = batch
            x = x.to(self.device)
            B, C, T = x.shape

            if has_noise:
                batch_seed = int(noise_seed) + int(batch_idx) if noise_seed is not None else None
                x_clean = x.clone()
                x_noisy = apply_eeg_noise(
                    x,
                    sfreq=sfreq,
                    snr_db=snr_db,
                    noise_types=noise_types,
                    channel_dropout_prob=channel_dropout_prob,
                    one_over_f_band=one_over_f_band,
                    emg_band=emg_band,
                    seed=batch_seed,
                )
                noise = x_noisy - x_clean
                sig_rms = torch.sqrt((x_clean ** 2).mean(dim=-1) + 1e-8)
                noise_rms = torch.sqrt((noise ** 2).mean(dim=-1) + 1e-8)
                achieved_snr_db = 20 * torch.log10(sig_rms / noise_rms)

                print(
                    f"[NoiseDebug] target={snr_db}dB "
                    f"achieved_mean={achieved_snr_db.mean().item():.2f}dB "
                    f"achieved_std={achieved_snr_db.std().item():.2f}dB"
                )

                x = x_noisy

            # Handle bipolar/mismatch channels like training path
            if C != coords.shape[0]:
                batch_coords = [ch_name[0] for ch_name in batch_coords]
                coords = self._coords(batch_coords).to(self.device)

            cb = coords.unsqueeze(0).expand(B, -1, -1)

            logits = self.model(x, cb, probe_layer_idx=self.probe_layer_idx)
            # Get window-level predictions
            pred = torch.argmax(logits, dim=1)
            preds_all.append(pred.cpu().numpy())
            idx_map_all.append(idx.cpu().numpy())



        preds = np.concatenate(preds_all)
        idx_map = np.concatenate(idx_map_all)

        # Majority voting: Combine windows back into 1 patient prediction
        if self.chunk_len_s is not None and not self.model.is_multilabel_task:
            unique_indices = np.unique(idx_map)
            final_predictions = []
            for i in unique_indices:
                patient_votes = preds[idx_map == i]
                final_predictions.append(Counter(patient_votes).most_common(1)[0][0])
            return np.array([map_label_reverse(p, task_name) for p in final_predictions])

        return np.array([map_label_reverse(p, task_name) for p in preds])

    def _get_embedding_cache_path(self, checkpoint_path: str, task_name: str, dataset_hash: str, split: str) -> Path:
        """Generate cache path for embeddings.

        The probe-layer suffix takes precedence over the attentive ``_seq`` tag:
        layerwise probes always extract pooled CLS embeddings regardless of probe head.
        """
        cache_dir = Path(get_config_value("cache", ".cache")) / "lejepa_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_hash = hashlib.md5(str(checkpoint_path).encode()).hexdigest()[:12]
        if self.probe_layer is not None:
            tag = probe_layer_suffix(self.probe_layer)
        elif self.attentive_probe:
            tag = "_seq"
        else:
            tag = ""
        return cache_dir / f"{task_name}_{ckpt_hash}_{dataset_hash}_{split}{tag}_{EMBED_CACHE_VERSION}.npz"

    def _compute_dataset_hash(self, X: list, meta: list) -> str:
        """Compute a hash to identify the dataset (robust to missing fields)."""
        task = ""
        channels = []
        if meta and isinstance(meta, list) and isinstance(meta[0], dict):
            task = meta[0].get("task_name", "") or ""
            ch = meta[0].get("channel_names") or meta[0].get("ch_names") or []
            channels = list(ch)[:5] if ch is not None else []

        shapes = []
        for x in X[:5]:
            shapes.append(getattr(x, "shape", None))

        hash_data = {
            "n_samples": len(X),
            "shapes": shapes,
            "task": task,
            "channels": channels,
            **({"embedding_type": "sequence"} if self.attentive_probe else {}),
            "cache_version": EMBED_CACHE_VERSION,
        }
        return hashlib.md5(json.dumps(hash_data, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def _load_or_extract_embeddings(self, dataloader, coords, checkpoint_path: str, task_name: str,
                                    dataset_hash: str, split: str) -> tuple:
        """Load cached embeddings or extract and cache them."""
        cache_path = self._get_embedding_cache_path(checkpoint_path, task_name, dataset_hash, split)

        if self.attentive_probe:
            index_path = cache_path.with_suffix(".index.json")
            shard_prefix = cache_path.with_suffix("")
            if index_path.exists():
                try:
                    print(f"[LeJEPAClinical] Loading cached shards from {index_path}")
                    return ShardedEmbeddingDataset(index_path), None
                except Exception as e:
                    print(f"[LeJEPAClinical] Sharded cache load failed ({e}); re-extracting.")

            emb_path = cache_path.with_suffix(".embeddings.npy")
            lbl_path = cache_path.with_suffix(".labels.npy")
            if emb_path.exists() and lbl_path.exists():
                try:
                    print(f"[LeJEPAClinical] Loading cached embeddings from {emb_path}")
                    return MemmapEmbeddingDataset(emb_path, lbl_path), None
                except Exception as e:
                    print(f"[LeJEPAClinical] Cache load failed ({e}); re-extracting.")

            print(f"[LeJEPAClinical] Extracting embeddings (will cache to {index_path})")
            return self._extract_embeddings_clinical_stream(dataloader, coords, index_path, shard_prefix), None

        if cache_path.exists():
            try:
                print(f"[LeJEPAClinical] Loading cached embeddings from {cache_path}")
                data = np.load(cache_path)
                embeddings = torch.from_numpy(data["embeddings"])
                labels = torch.from_numpy(data["labels"])
                return embeddings, labels
            except Exception as e:
                print(f"[LeJEPAClinical] Cache load failed ({e}); re-extracting.")

        print(f"[LeJEPAClinical] Extracting embeddings (will cache to {cache_path})")
        embeddings, labels = self._extract_embeddings_clinical(dataloader, coords)

        np.savez_compressed(
            cache_path,
            embeddings=embeddings.numpy(),
            labels=labels.numpy()
        )
        return embeddings, labels

    def _subsample_indices(self, n_samples: int, percentage: float, seed: int = 42) -> np.ndarray:
        """Get deterministic subsample indices for data percentage sweeps."""
        rng = np.random.RandomState(seed)
        n_select = max(1, int(n_samples * percentage))
        indices = rng.permutation(n_samples)[:n_select]
        return np.sort(indices)
