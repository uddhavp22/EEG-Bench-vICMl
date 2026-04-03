from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Memory
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModel

from ..abstract_model import AbstractModel
from ..reve_utils import (
    REVE_BACKBONE_ID,
    REVE_POSITIONS_ID,
    build_reve_cache_path,
    pool_reve_features,
)
from .LaBraM.make_dataset import make_dataset_reve
from .LaBraM.utils_2 import calc_class_weights, n_unique_labels, reverse_map_label
from ...config import get_config_value
from ...utils import wandb_utils

logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Simple wrapper over preprocessed BCI numpy arrays."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return {"data": self.X[idx]}
        return {"data": self.X[idx], "labels": self.y[idx]}


class REVEWrapper(nn.Module):
    """Pooled REVE backbone plus a lightweight classifier."""

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_classes: int,
        freeze_backbone: bool = True,
        coords: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            REVE_BACKBONE_ID,
            trust_remote_code=True,
            dtype="auto",
        )

        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

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
        logger.info("[REVE BCI] Using pooled feature dimension %s", self.feature_dim)

        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        pos = pos.to(x.device)
        features = pool_reve_features(self.backbone(x, pos))
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pos = pos.to(x.device)
            return pool_reve_features(self.backbone(x, pos))

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class REVEBenchmarkModel(AbstractModel):
    def __init__(self, freeze_backbone: bool = True, linear_probe: bool = False):
        super().__init__("REVEModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_probe = linear_probe
        self.freeze_backbone = freeze_backbone or linear_probe
        self.supports_full_dataset_cache = self.freeze_backbone
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

        self.pos_bank = AutoModel.from_pretrained(
            REVE_POSITIONS_ID,
            trust_remote_code=True,
            dtype="auto",
        )

        bank_names = self.pos_bank.get_all_positions()
        self._bank_vocab = set(bank_names)
        self._upper_to_bank = {name.upper(): name for name in bank_names}

        self.model: Optional[REVEWrapper] = None
        self.task_name: Optional[str] = None

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
                    logger.warning("[REVE BCI] Dropping bipolar channel '%s' (missing: %s)", name, missing)
            else:
                normed = self._normalize_ch_name(name)
                if normed in self._bank_vocab:
                    kept.append((idx, [normed]))
                    query_names.append(normed)
                else:
                    logger.warning("[REVE BCI] Dropping channel '%s' from position lookup", name)

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
        return positions, kept_indices

    def _make_collate_fn(self, channel_names: List[str]):
        positions, ch_keep = self._get_channel_coords(channel_names)

        def collate(batch):
            x_data = torch.stack([item["data"] for item in batch])
            if ch_keep is not None:
                x_data = x_data[:, ch_keep, :]
            batch_positions = positions.repeat(len(batch), 1, 1)
            payload = {"sample": x_data, "pos": batch_positions}
            if "labels" in batch[0]:
                payload["label"] = torch.stack([item["labels"] for item in batch]).long()
            return payload

        return collate

    def _init_model(self, sample: np.ndarray, channel_names: List[str], n_classes: int) -> None:
        if self.model is not None:
            return
        coords, ch_keep = self._get_channel_coords(channel_names)
        if ch_keep is not None:
            sample = sample[ch_keep, :]
        n_channels = coords.shape[0]
        if sample.shape[0] != n_channels:
            raise ValueError(
                f"Unexpected REVE feature shape: sample has {sample.shape[0]} channels, coords resolved to {n_channels}"
            )
        self.model = REVEWrapper(
            n_channels=sample.shape[0],
            n_timepoints=sample.shape[1],
            n_classes=n_classes,
            freeze_backbone=self.freeze_backbone,
            coords=coords,
        ).to(self.device)

    def _cache_payload(self, meta: List[Dict]) -> Dict:
        return {
            "task": meta[0]["task_name"],
            "datasets": [
                {
                    "name": m.get("name", f"dataset_{idx}"),
                    "sfreq": m.get("sampling_frequency"),
                    "channels": [ch.upper() for ch in m.get("channel_names", [])],
                }
                for idx, m in enumerate(meta)
            ],
            "backbone": REVE_BACKBONE_ID,
            "pooling": "mean_tokens",
            "preprocess": "make_dataset_reve_v1",
            "model": self.name,
        }

    def _encode_dataset_features(self, dataset, channel_names: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        collate_fn = self._make_collate_fn(channel_names)
        labels = dataset.labels
        if labels is None:
            raise ValueError("Training labels are required to build REVE BCI feature cache")
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)

        loader = DataLoader(
            SimpleDataset(dataset.data, labels),
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        features = []
        targets = []
        self.model.eval()
        for batch in tqdm(loader, desc="Cache REVE BCI embeddings", leave=False):
            data = batch["sample"].to(self.device)
            pos = batch["pos"].to(self.device)
            feats = self.model.extract_features(data, pos).cpu()
            features.append(feats)
            targets.append(batch["label"].cpu())

        return torch.cat(features, dim=0), torch.cat(targets, dim=0)

    def _load_or_build_feature_cache(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
        meta: List[Dict],
        n_classes: int,
    ) -> Dict:
        cache_path = build_reve_cache_path("bci_reve", self._cache_payload(meta))
        if self.model is None:
            cached_make_dataset = self.cache.cache(make_dataset_reve)
            for dataset_X, dataset_y, dataset_meta in zip(X, y, meta):
                dataset = cached_make_dataset(
                    dataset_X,
                    dataset_y,
                    meta[0]["task_name"],
                    dataset_meta["sampling_frequency"],
                    dataset_meta["channel_names"],
                    train=False,
                )
                if len(dataset) > 0:
                    self._init_model(dataset.data[0], dataset.ch_names, n_classes)
                    break

        if cache_path.exists():
            logger.info("[REVE BCI] Loading cached embeddings from %s", cache_path)
            return torch.load(cache_path, map_location="cpu")

        cached_make_dataset = self.cache.cache(make_dataset_reve)
        bundles = []
        for dataset_X, dataset_y, dataset_meta in zip(X, y, meta):
            dataset = cached_make_dataset(
                dataset_X,
                dataset_y,
                meta[0]["task_name"],
                dataset_meta["sampling_frequency"],
                dataset_meta["channel_names"],
                train=False,
            )
            if len(dataset) == 0:
                feature_dim = self.model.feature_dim if self.model is not None else 512
                bundles.append(
                    {
                        "features": torch.empty((0, feature_dim), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.long),
                        "channels": [],
                        "n_timepoints": 0,
                    }
                )
                continue

            if self.model is None:
                self._init_model(dataset.data[0], dataset.ch_names, n_classes)
            features, labels = self._encode_dataset_features(dataset, dataset.ch_names)
            bundles.append(
                {
                    "features": features.float(),
                    "labels": labels.long(),
                    "channels": list(dataset.ch_names),
                    "n_timepoints": int(dataset.data.shape[2]),
                }
            )

        cache_bundle = {"dataset_bundles": bundles, "metadata": self._cache_payload(meta)}
        torch.save(cache_bundle, cache_path)
        logger.info("[REVE BCI] Saved cached embeddings to %s", cache_path)
        return cache_bundle

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

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
        best_state = None
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(30):
            self.model.classifier.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
                batch_features = batch_features.to(self.device).float()
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model.classify_features(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_labels.size(0)
                correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                total += batch_labels.size(0)

            train_loss = total_loss / total if total else 0.0
            train_acc = correct / total if total else 0.0
            metrics = {
                f"{self.name}/train_loss": train_loss,
                f"{self.name}/train_acc": train_acc,
            }

            if val_loader is not None:
                self.model.classifier.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device).float()
                        batch_labels = batch_labels.to(self.device)
                        logits = self.model.classify_features(batch_features)
                        loss = criterion(logits, batch_labels)
                        val_loss += loss.item() * batch_labels.size(0)
                        val_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                        val_total += batch_labels.size(0)

                avg_val_loss = val_loss / val_total if val_total else 0.0
                val_acc = val_correct / val_total if val_total else 0.0
                metrics[f"{self.name}/val_loss"] = avg_val_loss
                metrics[f"{self.name}/val_acc"] = val_acc

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
        self.task_name = meta[0]["task_name"]
        n_classes = n_unique_labels(self.task_name)
        subset_seed = 0 if subset_seed is None else subset_seed

        if self.freeze_backbone:
            cache_bundle = self._load_or_build_feature_cache(X, y, meta, n_classes)
            selected_features = []
            selected_labels = []
            for ds_idx, bundle in enumerate(cache_bundle["dataset_bundles"]):
                if bundle["features"].shape[0] == 0:
                    continue
                ds_indices = subset_indices[ds_idx] if subset_indices is not None else list(range(bundle["features"].shape[0]))
                if len(ds_indices) == 0:
                    continue
                index_tensor = torch.as_tensor(ds_indices, dtype=torch.long)
                selected_features.append(bundle["features"][index_tensor])
                selected_labels.append(bundle["labels"][index_tensor])

            if not selected_features:
                raise ValueError("Selected subset yielded no REVE BCI training samples.")

            features = torch.cat(selected_features, dim=0)
            labels = torch.cat(selected_labels, dim=0)
            class_weight_labels = self._subset_original_labels(y, subset_indices)
            class_weights = torch.tensor(calc_class_weights(class_weight_labels, self.task_name), dtype=torch.float32, device=self.device)
            self._fit_classifier_on_features(features, labels, class_weights, subset_seed)
            return

        cached_make_dataset = self.cache.cache(make_dataset_reve)
        datasets = [
            cached_make_dataset(
                dataset_X,
                dataset_y,
                self.task_name,
                dataset_meta["sampling_frequency"],
                dataset_meta["channel_names"],
                train=True,
                split_size=0.15,
            )
            for dataset_X, dataset_y, dataset_meta in zip(X, y, meta)
        ]
        dataset_train_list = [dataset[0] for dataset in datasets if len(dataset[0]) > 0]
        dataset_val_list = [dataset[1] for dataset in datasets if len(dataset[1]) > 0]
        if not dataset_train_list:
            logger.warning("[REVE BCI] No training samples after preprocessing.")
            return

        self._init_model(dataset_train_list[0].data[0], dataset_train_list[0].ch_names, n_classes)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(calc_class_weights(y, self.task_name), dtype=torch.float32, device=self.device)
        )
        trainable_params = filter(lambda param: param.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=1e-3)

        train_loaders = []
        for dataset in dataset_train_list:
            labels = dataset.labels
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            train_loaders.append(
                DataLoader(
                    SimpleDataset(dataset.data, labels),
                    batch_size=64,
                    shuffle=True,
                    collate_fn=self._make_collate_fn(dataset.ch_names),
                    num_workers=0,
                )
            )

        val_loaders = []
        for dataset in dataset_val_list:
            labels = dataset.labels
            if labels.ndim > 1:
                labels = np.argmax(labels, axis=1)
            val_loaders.append(
                DataLoader(
                    SimpleDataset(dataset.data, labels),
                    batch_size=64,
                    shuffle=False,
                    collate_fn=self._make_collate_fn(dataset.ch_names),
                    num_workers=0,
                )
            )

        best_state = None
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0
        for epoch in range(10):
            self.model.train()
            if self.freeze_backbone:
                self.model.backbone.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for loader in train_loaders:
                for batch in tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False):
                    data = batch["sample"].to(self.device)
                    pos = batch["pos"].to(self.device)
                    target = batch["label"].to(self.device)
                    optimizer.zero_grad()
                    logits = self.model(data, pos)
                    loss = criterion(logits, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * target.size(0)
                    total_correct += (logits.argmax(dim=1) == target).sum().item()
                    total_samples += target.size(0)

            metrics = {
                f"{self.name}/train_loss": total_loss / total_samples if total_samples else 0.0,
                f"{self.name}/train_acc": total_correct / total_samples if total_samples else 0.0,
            }

            if val_loaders:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for loader in val_loaders:
                        for batch in loader:
                            data = batch["sample"].to(self.device)
                            pos = batch["pos"].to(self.device)
                            target = batch["label"].to(self.device)
                            logits = self.model(data, pos)
                            loss = criterion(logits, target)
                            val_loss += loss.item() * target.size(0)
                            val_correct += (logits.argmax(dim=1) == target).sum().item()
                            val_total += target.size(0)

                avg_val_loss = val_loss / val_total if val_total else 0.0
                metrics[f"{self.name}/val_loss"] = avg_val_loss
                metrics[f"{self.name}/val_acc"] = val_correct / val_total if val_total else 0.0
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

    def _subset_original_labels(self, y: List[np.ndarray], subset_indices: Optional[List[List[int]]]) -> List[np.ndarray]:
        if subset_indices is None:
            return y
        return [labels[idxs] for labels, idxs in zip(y, subset_indices)]

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        if self.model is None:
            logger.warning("[REVE BCI] Predict called before fit; returning empty array.")
            return np.array([])

        task_name = meta[0]["task_name"]
        cached_make_dataset = self.cache.cache(make_dataset_reve)
        all_preds = []
        for dataset_X, dataset_meta in zip(X, meta):
            dataset = cached_make_dataset(
                dataset_X,
                None,
                task_name,
                dataset_meta["sampling_frequency"],
                dataset_meta["channel_names"],
                train=False,
            )
            if len(dataset) == 0:
                continue

            loader = DataLoader(
                SimpleDataset(dataset.data, None),
                batch_size=64,
                shuffle=False,
                collate_fn=self._make_collate_fn(dataset.ch_names),
                num_workers=0,
            )

            for batch in tqdm(loader, desc="Predicting", leave=False):
                data = batch["sample"].to(self.device)
                pos = batch["pos"].to(self.device)
                logits = self.model(data, pos)
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())

        if not all_preds:
            return np.array([])
        predictions = np.concatenate(all_preds, axis=0)
        return np.array([reverse_map_label(int(pred), task_name) for pred in predictions])
