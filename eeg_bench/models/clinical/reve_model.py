from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel

from ..abstract_model import AbstractModel
from ..reve_utils import (
    REVE_BACKBONE_ID,
    REVE_POSITIONS_ID,
    build_reve_cache_path,
    pool_reve_features,
)
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from ...utils import wandb_utils

logger = logging.getLogger(__name__)

DEFAULT_SINGLE_LABEL_CHUNK_LEN_S = 10
DEFAULT_MULTILABEL_CHUNK_LEN_S = 16


class REVEClinicalWrapper(nn.Module):
    """Pooled REVE backbone plus a clinical classification head."""

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        num_classes: int,
        num_labels_per_chunk: Optional[int] = None,
        freeze_backbone: bool = True,
        coords: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.is_multilabel_task = num_labels_per_chunk is not None
        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk

        self.backbone = AutoModel.from_pretrained(
            REVE_BACKBONE_ID,
            trust_remote_code=True,
            dtype="auto",
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        with torch.no_grad():
            backbone_device = next(self.backbone.parameters()).device
            dummy = torch.randn(1, n_channels, n_timepoints, device=backbone_device)
            if coords is not None:
                dummy_coords = coords.unsqueeze(0).to(backbone_device)
            else:
                dummy_coords = torch.zeros(1, n_channels, 3, device=backbone_device)
            pooled = pool_reve_features(self.backbone(dummy, dummy_coords))
            self.feature_dim = pooled.shape[1]

        if self.feature_dim <= 0:
            raise ValueError(f"Unexpected REVE pooled feature shape: {self.feature_dim}")
        logger.info("[REVE Clinical] Using pooled feature dimension %s", self.feature_dim)

        out_dim = num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1)
        self.classifier = nn.Linear(self.feature_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        pos = pos.to(x.device)
        features = pool_reve_features(self.backbone(x, pos))
        logits = self.classifier(features)
        if self.is_multilabel_task:
            logits = logits.view(x.shape[0], self.num_classes, self.num_labels_per_chunk)
        return logits

    def extract_features(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pos = pos.to(x.device)
            return pool_reve_features(self.backbone(x, pos))

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(features)
        if self.is_multilabel_task:
            logits = logits.view(features.shape[0], self.num_classes, self.num_labels_per_chunk)
        return logits


class REVEClinicalModel(AbstractModel):
    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        chunk_len_s: Optional[int] = None,
        freeze_backbone: bool = True,
        linear_probe: bool = False,
    ):
        super().__init__("REVEModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk
        if num_labels_per_chunk is not None:
            self.chunk_len_s = (
                DEFAULT_MULTILABEL_CHUNK_LEN_S if chunk_len_s is None else chunk_len_s
            )
            self.internal_chunk_len_s = None
            self.use_internal_chunking = False
        else:
            self.chunk_len_s = None
            self.internal_chunk_len_s = (
                DEFAULT_SINGLE_LABEL_CHUNK_LEN_S if chunk_len_s is None else chunk_len_s
            )
            self.use_internal_chunking = True
            logger.info(
                "[REVE Clinical] Single-label task detected. Using internal %s-second chunking.",
                self.internal_chunk_len_s,
            )
        self.linear_probe = linear_probe
        self.freeze_backbone = freeze_backbone or linear_probe

        self.pos_bank = AutoModel.from_pretrained(REVE_POSITIONS_ID, trust_remote_code=True).to(self.device)
        bank_names = self.pos_bank.get_all_positions()
        self._bank_vocab = set(bank_names)
        self._upper_to_bank = {name.upper(): name for name in bank_names}

        self.model: Optional[REVEClinicalWrapper] = None
        self._ch_keep: Optional[List[int]] = None
        self.supports_full_dataset_cache = self.freeze_backbone

    @staticmethod
    def _collate_labeled_batch(batch):
        xs = torch.stack([item[0] for item in batch])
        ys = [item[1] for item in batch]
        channels = [item[2] for item in batch]
        if ys[0] is None:
            yb = None
        elif isinstance(ys[0], torch.Tensor):
            yb = torch.stack(ys)
        else:
            yb = torch.as_tensor(ys)
        return xs, yb, channels

    @staticmethod
    def _collate_predict_batch(batch):
        xs = torch.stack([item[0] for item in batch])
        raw_indices = [item[1] for item in batch]
        if raw_indices[0] is None:
            indices = torch.arange(len(batch), dtype=torch.long)
            has_explicit_indices = False
        else:
            indices = torch.as_tensor(raw_indices, dtype=torch.long)
            has_explicit_indices = True
        return xs, indices, [item[2] for item in batch], has_explicit_indices

    def _normalize_ch_name(self, name: str) -> str:
        if name in self._bank_vocab:
            return name
        return self._upper_to_bank.get(name.upper(), name)

    def _get_channel_coords(self, ch_names: List[str]) -> tuple[torch.Tensor, Optional[List[int]]]:
        clean_names = [c.replace("EEG", "").strip() for c in ch_names]
        kept = []
        query_names = []

        for idx, name in enumerate(clean_names):
            if "-" in name:
                parts = [part.strip() for part in name.split("-", 1)]
                normed = [self._normalize_ch_name(part) for part in parts]
                if all(entry in self._bank_vocab for entry in normed):
                    kept.append((idx, normed))
                    query_names.extend(normed)
                else:
                    missing = [part for part, entry in zip(parts, normed) if entry not in self._bank_vocab]
                    logger.warning("[REVE Clinical] Dropping bipolar channel '%s' (missing: %s)", name, missing)
            else:
                normed = self._normalize_ch_name(name)
                if normed in self._bank_vocab:
                    kept.append((idx, [normed]))
                    query_names.append(normed)
                else:
                    logger.warning("[REVE Clinical] Dropping channel '%s' from position lookup", name)

        if not kept:
            raise ValueError("No channels could be resolved by the REVE position bank")

        unique_names = list(dict.fromkeys(query_names))
        raw_positions = self.pos_bank(unique_names)
        if isinstance(raw_positions, dict):
            raw_positions = raw_positions.get(
                "positions",
                raw_positions.get("coords", raw_positions.get("last_hidden_state")),
            )
        if raw_positions.dim() == 3:
            raw_positions = raw_positions.squeeze(0)

        elec_to_pos = {name: raw_positions[j].float().cpu() for j, name in enumerate(unique_names)}
        positions = torch.zeros(len(kept), 3, dtype=torch.float32)
        for out_idx, (_, electrodes) in enumerate(kept):
            if len(electrodes) == 2:
                positions[out_idx] = (elec_to_pos[electrodes[0]] + elec_to_pos[electrodes[1]]) / 2.0
            else:
                positions[out_idx] = elec_to_pos[electrodes[0]]

        kept_indices = [idx for idx, _ in kept] if len(kept) < len(clean_names) else None
        return positions.to(self.device), kept_indices

    @staticmethod
    def _actual_ch_names(dataset) -> List[str]:
        _, _, first_ch = dataset[0]
        if isinstance(first_ch, list) and len(first_ch) > 0:
            return first_ch
        return dataset.ch_names

    @staticmethod
    def _parse_recording_index(name: str) -> int:
        for token in name.split("_")[1:]:
            if token.isdigit():
                return int(token)
        raise ValueError(f"Unable to parse recording index from name '{name}'")

    def _filter_dataset_to_subset(self, dataset, X, subset_indices):
        global_indices = set()
        offset = 0
        for ds_idx, ds in enumerate(X):
            for idx in subset_indices[ds_idx]:
                global_indices.add(offset + idx)
            offset += len(ds)
        dataset.recording_names = [
            name for name in dataset.recording_names if self._parse_recording_index(name) in global_indices
        ]
        return dataset

    @staticmethod
    def _gather_subset_labels(labels, subset_indices):
        subset = []
        for ds_labels, idxs in zip(labels, subset_indices):
            if isinstance(ds_labels, np.ndarray):
                subset.append(ds_labels[idxs])
            else:
                subset.append([ds_labels[i] for i in idxs])
        return subset

    def _init_model(self, sample: np.ndarray, coords: torch.Tensor) -> None:
        if self.model is not None:
            return
        n_channels, n_timepoints = sample.shape
        self.model = REVEClinicalWrapper(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            num_classes=self.num_classes,
            num_labels_per_chunk=self.num_labels_per_chunk,
            freeze_backbone=self.freeze_backbone,
            coords=coords,
        ).to(self.device)

    def _internal_chunk_len_samples(self, sfreq: int) -> Optional[int]:
        if not self.use_internal_chunking or self.internal_chunk_len_s is None:
            return None
        return max(1, int(round(self.internal_chunk_len_s * sfreq)))

    def _split_signal_into_chunks(self, signal: torch.Tensor, chunk_len: int) -> torch.Tensor:
        """Split a recording [C, T] into >=1 contiguous chunks."""
        assert signal.dim() == 2, "Signal must have shape [C, T]"
        signal_len = signal.shape[-1]
        if signal_len <= chunk_len:
            return signal.unsqueeze(0).contiguous()

        n_chunks = max(1, signal_len // chunk_len)
        trimmed = n_chunks * chunk_len
        if trimmed == 0:
            return signal.unsqueeze(0).contiguous()
        signal = signal[..., :trimmed].contiguous()
        return signal.view(signal.shape[0], n_chunks, chunk_len).permute(1, 0, 2).contiguous()

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
        sfreq: int = 200,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._ch_keep is not None:
            x = x[:, self._ch_keep, :]

        if not self.use_internal_chunking:
            coords_batch = coords.unsqueeze(0).expand(x.size(0), -1, -1)
            return x, yb, coords_batch

        chunk_len = self._internal_chunk_len_samples(sfreq)
        assert chunk_len is not None

        chunked_x, chunked_y, chunked_cb = [], [], []
        for idx in range(x.size(0)):
            sample_chunks = self._split_signal_into_chunks(x[idx], chunk_len)
            chunked_x.append(sample_chunks)
            chunked_y.append(self._expand_label_for_chunks(yb[idx], sample_chunks.size(0)))
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
        sfreq: int = 200,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._ch_keep is not None:
            x = x[:, self._ch_keep, :]

        if not self.use_internal_chunking:
            coords_batch = coords.unsqueeze(0).expand(x.size(0), -1, -1)
            return x, indices, coords_batch

        chunk_len = self._internal_chunk_len_samples(sfreq)
        assert chunk_len is not None

        chunked_x, chunked_idx, chunked_cb = [], [], []
        for idx in range(x.size(0)):
            sample_chunks = self._split_signal_into_chunks(x[idx], chunk_len)
            chunked_x.append(sample_chunks)
            chunked_idx.append(indices[idx].repeat(sample_chunks.size(0)))
            chunked_cb.append(coords.unsqueeze(0).expand(sample_chunks.size(0), -1, -1))

        return (
            torch.cat(chunked_x, dim=0),
            torch.cat(chunked_idx, dim=0),
            torch.cat(chunked_cb, dim=0),
        )

    def _cache_payload(self, task_name: str, meta: List[Dict]) -> Dict:
        return {
            "task": task_name,
            "datasets": [m.get("name", f"dataset_{idx}") for idx, m in enumerate(meta)],
            "chunk_len_s": self.chunk_len_s,
            "internal_chunk_len_s": self.internal_chunk_len_s,
            "num_labels_per_chunk": self.num_labels_per_chunk,
            "backbone": REVE_BACKBONE_ID,
            "pooling": "mean_tokens",
            "preprocess": "make_dataset_2",
            "chunk_mode": "internal" if self.use_internal_chunking else "dataset",
            "model": self.name,
        }

    def _cache_file_path(self, task_name: str, meta: List[Dict]):
        return build_reve_cache_path("clinical_reve", self._cache_payload(task_name, meta))

    def _load_or_build_embedding_cache(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
        meta: List[Dict],
        task_name: str,
    ) -> Dict:
        cache_file = self._cache_file_path(task_name, meta)
        if cache_file.exists():
            logger.info("[REVE Clinical] Loading cached embeddings from %s", cache_file)
            cache_bundle = torch.load(cache_file, map_location="cpu")
            if self.model is None:
                coords, ch_keep = self._get_channel_coords(cache_bundle["metadata"]["channels"])
                self._ch_keep = ch_keep
                model_input_shape = cache_bundle["metadata"].get(
                    "model_input_shape",
                    cache_bundle["metadata"]["sample_shape"],
                )
                sample = torch.zeros(model_input_shape[0], model_input_shape[1], dtype=torch.float32)
                self._init_model(sample.numpy(), coords)
            return cache_bundle

        dataset = make_dataset_2(
            X,
            y,
            meta,
            task_name,
            self.name,
            self.chunk_len_s,
            is_train=True,
            use_cache=True,
        )
        if len(dataset) == 0:
            dataset = make_dataset_2(
                X,
                y,
                meta,
                task_name,
                self.name,
                self.chunk_len_s,
                is_train=True,
                use_cache=False,
            )
        if len(dataset) == 0:
            raise ValueError("Dataset has 0 samples after preprocessing. Cannot cache embeddings.")

        channels = self._actual_ch_names(dataset)
        coords, ch_keep = self._get_channel_coords(channels)
        self._ch_keep = ch_keep

        sample_data, _, _ = dataset[0]
        if ch_keep is not None:
            sample_data = sample_data[ch_keep, :]
        if self.use_internal_chunking:
            chunk_len = self._internal_chunk_len_samples(getattr(dataset, "sfreq", 200))
            sample_data = self._split_signal_into_chunks(sample_data, chunk_len)[0]
        self._init_model(sample_data.numpy(), coords)

        loader = DataLoader(
            dataset,
            batch_size=64 if self.chunk_len_s else 1,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_labeled_batch,
        )

        features = []
        labels = []
        ordered_record_ids = [self._parse_recording_index(name) for name in dataset.recording_names]
        record_ids = []

        self.model.eval()
        for batch_idx, (x, yb, _) in enumerate(tqdm(loader, desc="Cache REVE clinical embeddings", leave=False)):
            x = x.to(self.device)
            if not self.model.is_multilabel_task and yb.dim() > 1:
                yb = yb.argmax(dim=1)
            chunk_x, chunk_y, coords_batch = self._prepare_train_batch(
                x,
                yb,
                coords,
                getattr(dataset, "sfreq", 200),
            )
            feats = self.model.extract_features(chunk_x, coords_batch).cpu()
            features.append(feats)
            labels.append(chunk_y.cpu())

            start = batch_idx * loader.batch_size
            end = start + x.size(0)
            batch_record_ids = []
            for rec_id, sample in zip(ordered_record_ids[start:end], x):
                n_chunks = 1
                if self.use_internal_chunking:
                    chunk_len = self._internal_chunk_len_samples(getattr(dataset, "sfreq", 200))
                    n_chunks = self._split_signal_into_chunks(
                        sample[self._ch_keep, :] if self._ch_keep is not None else sample,
                        chunk_len,
                    ).shape[0]
                batch_record_ids.extend([rec_id] * n_chunks)
            record_ids.append(torch.tensor(batch_record_ids, dtype=torch.long))

        cache_bundle = {
            "features": torch.cat(features, dim=0).float(),
            "labels": torch.cat(labels, dim=0).long(),
            "recording_ids": torch.cat(record_ids, dim=0).long(),
            "metadata": {
                "channels": channels,
                "sample_shape": tuple(dataset[0][0].shape),
                "model_input_shape": tuple(sample_data.shape),
                "chunk_len_s": self.chunk_len_s,
                "internal_chunk_len_s": self.internal_chunk_len_s,
            },
        }
        torch.save(cache_bundle, cache_file)
        logger.info("[REVE Clinical] Saved cached embeddings to %s", cache_file)
        return cache_bundle

    def _compute_global_record_indices(self, X: List[np.ndarray], subset_indices: List[List[int]]) -> List[int]:
        offsets = []
        running = 0
        for dataset in X:
            offsets.append(running)
            running += len(dataset)

        selected = []
        for ds_idx, idx_list in enumerate(subset_indices):
            base = offsets[ds_idx]
            for idx in idx_list:
                selected.append(base + idx)
        return selected

    def _build_recording_mask(self, recording_ids: torch.Tensor, selected_ids: List[int]) -> torch.Tensor:
        if len(selected_ids) == 0:
            raise ValueError("subset_indices resolved to an empty selection.")
        record_np = recording_ids.cpu().numpy()
        selected_np = np.array(selected_ids, dtype=record_np.dtype)
        return torch.from_numpy(np.isin(record_np, selected_np))

    def _split_feature_dataset(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        seed: int,
        val_fraction: float = 0.15,
    ) -> tuple[TensorDataset, Optional[TensorDataset]]:
        num_samples = features.shape[0]
        num_val = int(num_samples * val_fraction)
        if num_samples <= 1 or num_val == 0 or num_samples - num_val == 0:
            return TensorDataset(features, labels), None

        generator = torch.Generator()
        generator.manual_seed(seed)
        permutation = torch.randperm(num_samples, generator=generator)
        val_idx = permutation[:num_val]
        train_idx = permutation[num_val:]
        return (
            TensorDataset(features[train_idx], labels[train_idx]),
            TensorDataset(features[val_idx], labels[val_idx]),
        )

    def _fit_classifier_on_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        class_weights: torch.Tensor,
        seed: int,
    ) -> None:
        assert self.model is not None
        train_dataset, val_dataset = self._split_feature_dataset(features, labels, seed)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
        val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
        best_state = None
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(30):
            self.model.classifier.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            total_acc_samples = 0

            for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
                batch_features = batch_features.to(self.device).float()
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model.classify_features(batch_features)
                loss = self.model.loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_features.size(0)
                total_samples += batch_features.size(0)
                if logits.dim() == 2:
                    preds = torch.argmax(logits, dim=1)
                    target = batch_labels if batch_labels.dim() == 1 else batch_labels.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    total_acc_samples += batch_features.size(0)

            metrics = {
                f"{self.name}/train_loss": total_loss / total_samples if total_samples else 0.0,
            }
            if total_acc_samples:
                metrics[f"{self.name}/train_acc"] = correct / total_acc_samples

            if val_loader is not None:
                self.model.classifier.eval()
                val_loss = 0.0
                val_samples = 0
                val_correct = 0
                val_acc_samples = 0
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device).float()
                        batch_labels = batch_labels.to(self.device)
                        logits = self.model.classify_features(batch_features)
                        loss = self.model.loss_fn(logits, batch_labels)
                        val_loss += loss.item() * batch_features.size(0)
                        val_samples += batch_features.size(0)
                        if logits.dim() == 2:
                            preds = torch.argmax(logits, dim=1)
                            target = batch_labels if batch_labels.dim() == 1 else batch_labels.argmax(dim=1)
                            val_correct += (preds == target).sum().item()
                            val_acc_samples += batch_features.size(0)

                avg_val_loss = val_loss / val_samples if val_samples else 0.0
                metrics[f"{self.name}/val_loss"] = avg_val_loss
                if val_acc_samples:
                    metrics[f"{self.name}/val_acc"] = val_correct / val_acc_samples

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if self.wandb_run:
                wandb_utils.log(metrics, step=epoch + 1)

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def fit(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
        meta: List[Dict],
        subset_fraction: float = 1.0,
        subset_seed: Optional[int] = None,
        subset_indices: Optional[List[List[int]]] = None,
    ) -> None:
        task_name = meta[0]["task_name"]
        subset_seed = 0 if subset_seed is None else subset_seed

        if self.freeze_backbone:
            cache_bundle = self._load_or_build_embedding_cache(X, y, meta, task_name)
            all_features = cache_bundle["features"]
            all_labels = cache_bundle["labels"]
            recording_ids = cache_bundle["recording_ids"]

            if subset_indices is None:
                subset_features = all_features
                subset_labels_tensor = all_labels
                weight_labels = y
            else:
                selected_global_ids = self._compute_global_record_indices(X, subset_indices)
                mask = self._build_recording_mask(recording_ids, selected_global_ids)
                subset_features = all_features[mask].contiguous()
                subset_labels_tensor = all_labels[mask].contiguous()
                weight_labels = self._gather_subset_labels(y, subset_indices)

            if subset_features.shape[0] == 0:
                raise ValueError("Selected subset yielded no REVE clinical training samples.")

            class_weights = torch.tensor(calc_class_weights(weight_labels, task_name), dtype=torch.float32, device=self.device)
            self._fit_classifier_on_features(subset_features, subset_labels_tensor, class_weights, subset_seed)
            return

        dataset_train = make_dataset_2(
            X,
            y,
            meta,
            task_name,
            self.name,
            self.chunk_len_s,
            is_train=True,
            use_cache=True,
        )
        if len(dataset_train) == 0:
            dataset_train = make_dataset_2(
                X,
                y,
                meta,
                task_name,
                self.name,
                self.chunk_len_s,
                is_train=True,
                use_cache=False,
            )
        if len(dataset_train) == 0:
            logger.warning("[REVE Clinical] Dataset is empty after preprocessing; skipping fit.")
            return

        dataset_train, dataset_val = dataset_train.split_train_val(0.15)
        coords_train, ch_keep = self._get_channel_coords(self._actual_ch_names(dataset_train))
        coords_val, _ = self._get_channel_coords(self._actual_ch_names(dataset_val))
        self._ch_keep = ch_keep

        sample_data, _, _ = dataset_train[0]
        if self._ch_keep is not None:
            sample_data = sample_data[self._ch_keep, :]
        if self.use_internal_chunking:
            chunk_len = self._internal_chunk_len_samples(getattr(dataset_train, "sfreq", 200))
            sample_data = self._split_signal_into_chunks(sample_data, chunk_len)[0]
        self._init_model(sample_data.numpy(), coords_train)

        class_weights = torch.tensor(calc_class_weights(y, task_name), dtype=torch.float32, device=self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        batch_size = 64 if self.chunk_len_s else 1

        train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_labeled_batch,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_labeled_batch,
        )

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3)
        best_state = None
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(30):
            self.model.train()
            if self.freeze_backbone:
                self.model.backbone.eval()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            total_acc_samples = 0

            for x, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
                x = x.to(self.device)
                yb = yb.to(self.device)
                if not self.model.is_multilabel_task and yb.dim() > 1:
                    yb = yb.argmax(dim=1)
                x, yb, coords_batch = self._prepare_train_batch(
                    x,
                    yb,
                    coords_train,
                    getattr(dataset_train, "sfreq", 200),
                )

                optimizer.zero_grad()
                logits = self.model(x, coords_batch)
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

            metrics = {
                f"{self.name}/train_loss": total_loss / total_samples if total_samples else 0.0,
            }
            if total_acc_samples:
                metrics[f"{self.name}/train_acc"] = correct / total_acc_samples

            self.model.eval()
            val_loss = 0.0
            val_samples = 0
            val_correct = 0
            val_acc_samples = 0
            with torch.no_grad():
                for x, yb, _ in val_loader:
                    x = x.to(self.device)
                    yb = yb.to(self.device)
                    if not self.model.is_multilabel_task and yb.dim() > 1:
                        yb = yb.argmax(dim=1)
                    x, yb, coords_batch = self._prepare_train_batch(
                        x,
                        yb,
                        coords_val,
                        getattr(dataset_val, "sfreq", 200),
                    )
                    logits = self.model(x, coords_batch)
                    loss = self.model.loss_fn(logits, yb)
                    val_loss += loss.item() * x.size(0)
                    val_samples += x.size(0)
                    if logits.dim() == 2:
                        preds = torch.argmax(logits, dim=1)
                        target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                        val_correct += (preds == target).sum().item()
                        val_acc_samples += x.size(0)

            avg_val_loss = val_loss / val_samples if val_samples else 0.0
            metrics[f"{self.name}/val_loss"] = avg_val_loss
            if val_acc_samples:
                metrics[f"{self.name}/val_acc"] = val_correct / val_acc_samples

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            if self.wandb_run:
                wandb_utils.log(metrics, step=epoch + 1)

        if best_state is not None:
            self.model.load_state_dict(best_state)

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        if self.model is None:
            logger.warning("[REVE Clinical] Predict called before fit; returning empty array.")
            return np.array([])

        task_name = meta[0]["task_name"]
        dataset_test = make_dataset_2(
            X,
            None,
            meta,
            task_name,
            self.name,
            self.chunk_len_s,
            is_train=False,
            use_cache=True,
        )
        if len(dataset_test) == 0:
            return np.array([])

        test_loader = DataLoader(
            dataset_test,
            batch_size=64 if self.chunk_len_s else 1,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_predict_batch,
        )

        coords, _ = self._get_channel_coords(self._actual_ch_names(dataset_test))
        predictions = []
        indices = []
        sample_offset = 0

        self.model.eval()
        for x, batch_idx, _, has_explicit_indices in tqdm(test_loader, desc="Predicting", leave=False):
            raw_batch_size = x.size(0)
            x = x.to(self.device)
            batch_idx = batch_idx.to(self.device)
            if not has_explicit_indices:
                batch_idx = batch_idx + sample_offset
            x, batch_idx, coords_batch = self._prepare_inference_batch(
                x,
                batch_idx,
                coords,
                getattr(dataset_test, "sfreq", 200),
            )
            logits = self.model(x, coords_batch)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred.cpu())
            indices.append(batch_idx.cpu())
            if not has_explicit_indices:
                sample_offset += raw_batch_size

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        indices = torch.cat(indices, dim=0).cpu().numpy()

        if not self.model.is_multilabel_task and predictions.shape[0] != np.unique(indices).shape[0]:
            aggregated_predictions = []
            for idx in np.unique(indices):
                idx_predictions = predictions[indices == idx]
                aggregated_predictions.append(Counter(idx_predictions).most_common(1)[0][0])
            predictions = np.array(aggregated_predictions)

        if self.model.is_multilabel_task:
            return predictions

        return np.array([map_label_reverse(int(pred), task_name) for pred in predictions])
