from __future__ import annotations

from typing import List, Dict, Optional
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..abstract_model import AbstractModel
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from ...utils import wandb_utils
from ...utils.utils import CachedArrayDataset, create_temp_cache_dir, cleanup_temp_cache_dir
from transformers import AutoModel
from collections import Counter


class REVEClinicalWrapper(nn.Module):
    """
    Wraps the HuggingFace REVE model with a classification head.
    """

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

        self.backbone = AutoModel.from_pretrained(
            "brain-bzh/reve-base",
            trust_remote_code=True,
            dtype="auto",
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Determine input_dim dynamically via a sample forward pass
        with torch.no_grad():
            backbone_device = next(self.backbone.parameters()).device
            dummy = torch.randn(1, n_channels, n_timepoints, device=backbone_device)
            if coords is not None:
                dummy_coords = coords.unsqueeze(0).to(backbone_device)
            else:
                dummy_coords = torch.zeros(1, n_channels, 3, device=dummy.device)
            dummy_out = self.backbone(dummy, dummy_coords)
            input_dim = dummy_out.reshape(1, -1).shape[1]

        out_dim = num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, out_dim),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, pos):
        pos = pos.to(x.device)
        features = self.backbone(x, pos)
        logits = self.classifier(features)
        if self.is_multilabel_task:
            logits = logits.view(x.shape[0], self.num_classes, -1)
        return logits

    def extract_features(self, x, pos):
        with torch.no_grad():
            pos = pos.to(x.device)
            features = self.backbone(x, pos)
        return features.reshape(features.shape[0], -1)

    def classify_features(self, features):
        logits = self.classifier(features)
        if self.is_multilabel_task:
            logits = logits.view(features.shape[0], self.num_classes, -1)
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
        if chunk_len_s is None and num_labels_per_chunk:
            self.chunk_len_s = 16
        else:
            self.chunk_len_s = chunk_len_s
        self.linear_probe = linear_probe
        self.freeze_backbone = freeze_backbone or linear_probe

        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", trust_remote_code=True
        ).to(self.device)

        # Build case-insensitive lookup from the bank's own vocabulary
        bank_names = self.pos_bank.get_all_positions()
        self._bank_vocab = set(bank_names)
        self._upper_to_bank = {name.upper(): name for name in bank_names}

        self.model: Optional[REVEClinicalWrapper] = None
        self._ch_keep = None
        self.supports_full_dataset_cache = self.freeze_backbone

    def _normalize_ch_name(self, name: str) -> str:
        """Map a channel/electrode name to the position bank's expected casing."""
        if name in self._bank_vocab:
            return name
        return self._upper_to_bank.get(name.upper(), name)

    def _get_channel_coords(self, ch_names: List[str]):
        """Get 3D channel coordinates from position bank.

        For bipolar channels (e.g. "FPZ-CZ") the position is approximated as
        the midpoint of the two constituent electrodes.  Non-electrode channels
        that the position bank cannot resolve are dropped.

        Returns:
            Tuple of (positions [C, 3], kept_channel_indices) where
            kept_channel_indices is None when all channels resolved, or a
            list of int indices into the original ch_names.
        """
        clean_names = [c.replace("EEG", "").strip() for c in ch_names]

        kept = []
        query_names = []

        for i, name in enumerate(clean_names):
            if "-" in name:
                parts = [p.strip() for p in name.split("-", 1)]
                normed = [self._normalize_ch_name(p) for p in parts]
                if all(n in self._bank_vocab for n in normed):
                    kept.append((i, normed))
                    query_names.extend(normed)
                else:
                    missing = [p for p, n in zip(parts, normed) if n not in self._bank_vocab]
                    print(f"[REVE] Dropping bipolar channel '{name}' — electrode(s) {missing} not in position bank")
            else:
                normed = self._normalize_ch_name(name)
                if normed in self._bank_vocab:
                    kept.append((i, [normed]))
                    query_names.append(normed)
                else:
                    print(f"[REVE] Dropping channel '{name}' — not in position bank")

        if not kept:
            raise ValueError("No channels could be resolved by the position bank")

        unique_names = list(dict.fromkeys(query_names))
        raw_positions = self.pos_bank(unique_names)
        if isinstance(raw_positions, dict):
            raw_positions = raw_positions.get(
                "positions", raw_positions.get("coords", raw_positions.get("last_hidden_state"))
            )
        if raw_positions.dim() == 3:
            raw_positions = raw_positions.squeeze(0)

        elec_to_pos = {name: raw_positions[j].float() for j, name in enumerate(unique_names)}

        positions = torch.zeros(len(kept), 3)
        for out_i, (_, electrodes) in enumerate(kept):
            if len(electrodes) == 2:
                positions[out_i] = (elec_to_pos[electrodes[0]] + elec_to_pos[electrodes[1]]) / 2.0
            else:
                positions[out_i] = elec_to_pos[electrodes[0]]

        ch_keep = [idx for idx, _ in kept] if len(kept) < len(clean_names) else None
        return positions.to(self.device), ch_keep

    @staticmethod
    def _actual_ch_names(dataset) -> List[str]:
        """Return the actual channel names stored per-recording in the dataset."""
        _, _, first_ch = dataset[0]
        if isinstance(first_ch, list) and len(first_ch) > 0:
            return first_ch
        return dataset.ch_names

    @staticmethod
    def _parse_recording_index(name: str) -> int:
        for token in name.split("_")[1:]:
            if token.isdigit():
                return int(token)
        return -1

    def _filter_dataset_to_subset(self, dataset, X, subset_indices):
        """Filter dataset recordings to only those in subset_indices."""
        global_indices = set()
        offset = 0
        for ds_idx, ds in enumerate(X):
            for idx in subset_indices[ds_idx]:
                global_indices.add(offset + idx)
            offset += len(ds)
        dataset.recording_names = [
            name for name in dataset.recording_names
            if self._parse_recording_index(name) in global_indices
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
        n_channels, n_timepoints = sample.shape[0], sample.shape[1]
        self.model = REVEClinicalWrapper(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            num_classes=self.num_classes,
            num_labels_per_chunk=self.num_labels_per_chunk,
            freeze_backbone=self.freeze_backbone,
            coords=coords,
        ).to(self.device)

    def _fit_linear_probe_cached(self, train_loader, val_loader, coords_train, coords_val) -> None:
        assert self.model is not None

        cache_dir = create_temp_cache_dir("reve_clinical_lp_")
        try:
            feature_dim = self.model.classifier[-1].in_features
            train_count = len(train_loader.dataset)
            val_count = len(val_loader.dataset)

            train_features = np.memmap(
                os.path.join(cache_dir, "train_features.dat"),
                dtype=np.float16,
                mode="w+",
                shape=(train_count, feature_dim),
            )
            train_labels = np.memmap(
                os.path.join(cache_dir, "train_labels.dat"),
                dtype=np.int64,
                mode="w+",
                shape=(train_count,) if self.num_labels_per_chunk is None else (train_count, self.num_labels_per_chunk),
            )

            idx = 0
            self.model.eval()
            for x, yb, _ in tqdm(train_loader, desc="Cache REVE clinical train", leave=False):
                x, yb = x.to(self.device), yb.to(self.device)
                if self._ch_keep is not None:
                    x = x[:, self._ch_keep, :]
                if not self.model.is_multilabel_task and yb.dim() > 1:
                    yb = yb.argmax(dim=1)
                cb = coords_train.unsqueeze(0).expand(x.size(0), -1, -1)
                feats = self.model.extract_features(x, cb).cpu().numpy().astype(np.float16)
                labels = yb.cpu().numpy()
                bsz = feats.shape[0]
                train_features[idx:idx + bsz] = feats
                train_labels[idx:idx + bsz] = labels
                idx += bsz

            train_features.flush()
            train_labels.flush()

            val_features = np.memmap(
                os.path.join(cache_dir, "val_features.dat"),
                dtype=np.float16,
                mode="w+",
                shape=(val_count, feature_dim),
            )
            val_labels = np.memmap(
                os.path.join(cache_dir, "val_labels.dat"),
                dtype=np.int64,
                mode="w+",
                shape=(val_count,) if self.num_labels_per_chunk is None else (val_count, self.num_labels_per_chunk),
            )

            idx = 0
            for x, yb, _ in tqdm(val_loader, desc="Cache REVE clinical val", leave=False):
                x, yb = x.to(self.device), yb.to(self.device)
                if self._ch_keep is not None:
                    x = x[:, self._ch_keep, :]
                if not self.model.is_multilabel_task and yb.dim() > 1:
                    yb = yb.argmax(dim=1)
                cb = coords_val.unsqueeze(0).expand(x.size(0), -1, -1)
                feats = self.model.extract_features(x, cb).cpu().numpy().astype(np.float16)
                labels = yb.cpu().numpy()
                bsz = feats.shape[0]
                val_features[idx:idx + bsz] = feats
                val_labels[idx:idx + bsz] = labels
                idx += bsz

            val_features.flush()
            val_labels.flush()

            train_feat_loader = DataLoader(
                CachedArrayDataset(train_features, train_labels),
                batch_size=256,
                shuffle=True,
                num_workers=0,
            )
            val_feat_loader = DataLoader(
                CachedArrayDataset(val_features, val_labels),
                batch_size=256,
                shuffle=False,
                num_workers=0,
            )

            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.AdamW(trainable_params, lr=1e-3)

            num_epochs = 30
            patience = 10
            patience_counter = 0
            best_val_loss = float("inf")
            best_model_state = None

            for epoch in range(num_epochs):
                self.model.classifier.train()
                total_loss = 0.0
                total_samples = 0
                correct = 0
                total_acc_samples = 0

                for feats, yb in tqdm(train_feat_loader, desc=f"Epoch {epoch}", leave=False):
                    feats, yb = feats.to(self.device).float(), yb.to(self.device)
                    optimizer.zero_grad()
                    logits = self.model.classify_features(feats)
                    loss = self.model.loss_fn(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * feats.size(0)
                    total_samples += feats.size(0)
                    if logits.dim() == 2:
                        preds = torch.argmax(logits, dim=1)
                        target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                        correct += (preds == target).sum().item()
                        total_acc_samples += feats.size(0)

                train_loss = total_loss / total_samples if total_samples else 0.0
                train_acc = correct / total_acc_samples if total_acc_samples else 0.0

                # Validation
                val_loss = 0.0
                val_samples = 0
                val_correct = 0
                val_acc_samples = 0
                self.model.classifier.eval()
                with torch.no_grad():
                    for feats, yb in tqdm(val_feat_loader, desc=f"Val {epoch}", leave=False):
                        feats, yb = feats.to(self.device).float(), yb.to(self.device)
                        logits = self.model.classify_features(feats)
                        loss = self.model.loss_fn(logits, yb)
                        val_loss += loss.item() * feats.size(0)
                        val_samples += feats.size(0)
                        if logits.dim() == 2:
                            preds = torch.argmax(logits, dim=1)
                            target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                            val_correct += (preds == target).sum().item()
                            val_acc_samples += feats.size(0)

                avg_val_loss = val_loss / val_samples if val_samples else 0.0
                val_acc = val_correct / val_acc_samples if val_acc_samples else 0.0

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f"[Epoch {epoch + 1:02d}/{num_epochs}] "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                      f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} | "
                      f"patience={patience_counter}/{patience}")

                if self.wandb_run and total_samples:
                    metrics = {f"{self.name}/train_loss": train_loss}
                    if total_acc_samples:
                        metrics[f"{self.name}/train_acc"] = train_acc
                    if val_samples:
                        metrics[f"{self.name}/val_loss"] = avg_val_loss
                        if val_acc_samples:
                            metrics[f"{self.name}/val_acc"] = val_acc
                    wandb_utils.log(metrics, step=epoch + 1)

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1} (patience={patience})")
                    break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
        finally:
            cleanup_temp_cache_dir(cache_dir)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict],
            subset_fraction: float = 1.0, subset_seed: Optional[int] = None,
            subset_indices: Optional[List[List[int]]] = None) -> None:
        task_name = meta[0]["task_name"]

        # Create training dataset (always from full X for cache hit)
        dataset_train = make_dataset_2(
            X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=True
        )
        if len(dataset_train) == 0:
            print("[Warning] Dataset empty. Retrying without cache...")
            dataset_train = make_dataset_2(
                X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=False
            )
        if len(dataset_train) == 0:
            print("[Warning] Dataset empty after retries. Skipping training.")
            return

        # Filter to subset if provided (supports_full_dataset_cache path)
        if subset_indices is not None:
            self._filter_dataset_to_subset(dataset_train, X, subset_indices)
            y_for_weights = self._gather_subset_labels(y, subset_indices)
        else:
            y_for_weights = y

        dataset_train, dataset_val = dataset_train.split_train_val(0.15)
        if len(dataset_train) == 0:
            print("[Warning] Training split is empty. Skipping training.")
            return

        # Get channel coordinates; drop channels the position bank can't resolve
        coords_train, ch_keep = self._get_channel_coords(self._actual_ch_names(dataset_train))
        coords_val, _ = self._get_channel_coords(self._actual_ch_names(dataset_val))
        self._ch_keep = ch_keep

        sample_data, _, _ = dataset_train[0]
        if self._ch_keep is not None:
            sample_data = sample_data[self._ch_keep, :]
        if self.model is None:
            self._init_model(sample_data, coords_train)

        class_weights = torch.tensor(calc_class_weights(y_for_weights, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        train_loader = DataLoader(
            dataset_train, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
        )
        val_loader = DataLoader(
            dataset_val, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
        )

        if self.linear_probe:
            self._fit_linear_probe_cached(train_loader, val_loader, coords_train, coords_val)
            return

        # Training loop with early stopping
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=1e-3)

        num_epochs = 30
        patience = 10
        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(num_epochs):
            self.model.train()
            if self.freeze_backbone:
                self.model.backbone.eval()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            total_acc_samples = 0

            for x, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
                x, yb = x.to(self.device), yb.to(self.device)
                if self._ch_keep is not None:
                    x = x[:, self._ch_keep, :]
                if not self.model.is_multilabel_task and yb.dim() > 1:
                    yb = yb.argmax(dim=1)
                cb = coords_train.unsqueeze(0).expand(x.size(0), -1, -1)

                optimizer.zero_grad()
                logits = self.model(x, cb)
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

            train_loss = total_loss / total_samples if total_samples else 0.0
            train_acc = correct / total_acc_samples if total_acc_samples else 0.0

            # Validation
            val_loss = 0.0
            val_samples = 0
            val_correct = 0
            val_acc_samples = 0
            self.model.eval()
            with torch.no_grad():
                for x, yb, _ in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
                    x, yb = x.to(self.device), yb.to(self.device)
                    if self._ch_keep is not None:
                        x = x[:, self._ch_keep, :]
                    if not self.model.is_multilabel_task and yb.dim() > 1:
                        yb = yb.argmax(dim=1)
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

            avg_val_loss = val_loss / val_samples if val_samples else 0.0
            val_acc = val_correct / val_acc_samples if val_acc_samples else 0.0

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"[Epoch {epoch + 1:02d}/{num_epochs}] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"patience={patience_counter}/{patience}")

            if self.wandb_run and total_samples:
                metrics = {f"{self.name}/train_loss": train_loss}
                if total_acc_samples:
                    metrics[f"{self.name}/train_acc"] = train_acc
                if val_samples:
                    metrics[f"{self.name}/val_loss"] = avg_val_loss
                    if val_acc_samples:
                        metrics[f"{self.name}/val_acc"] = val_acc
                wandb_utils.log(metrics, step=epoch + 1)

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} (patience={patience})")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        if self.model is None:
            print("[Warning] REVE model was not trained (fit may have been skipped). Returning empty predictions.")
            return np.array([])

        task_name = meta[0]["task_name"]
        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name, self.chunk_len_s, is_train=False, use_cache=True
        )

        if len(dataset_test) == 0:
            return np.array([])

        test_loader = DataLoader(
            dataset_test, batch_size=64 if self.chunk_len_s else 1, shuffle=False, num_workers=0
        )
        coords, _ = self._get_channel_coords(self._actual_ch_names(dataset_test))
        self.model.eval()

        predictions = []
        indices = []
        for x, idx, _ in tqdm(test_loader, desc="Predicting"):
            x = x.to(self.device)
            if self._ch_keep is not None:
                x = x[:, self._ch_keep, :]
            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
            logits = self.model(x, cb)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred.cpu())
            indices.append(idx)

        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        indices = torch.cat(indices, dim=0).cpu().numpy()

        if self.chunk_len_s is not None and not self.model.is_multilabel_task:
            unique_indices = np.unique(indices)
            aggregated_predictions = []
            for idx in unique_indices:
                idx_predictions = predictions[indices == idx]
                most_common_prediction = Counter(idx_predictions).most_common(1)[0][0]
                aggregated_predictions.append(most_common_prediction)
            predictions = np.array(aggregated_predictions)

        mapped_pred = np.array([map_label_reverse(pred, task_name) for pred in predictions])
        return mapped_pred
