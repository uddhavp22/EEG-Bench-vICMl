import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
import numpy as np
from typing import List, Dict, Union
from tqdm import tqdm
import logging
from functools import partial
from ..abstract_model import AbstractModel
from .LaBraM.utils_2 import reverse_map_label
from ...utils import wandb_utils
from ...utils.utils import CachedArrayDataset, create_temp_cache_dir, cleanup_temp_cache_dir



class SimpleDataset(Dataset):
    """
    A simple wrapper to convert List[np.ndarray] into a Torch Dataset.
    Assumes X is (N, C, T) and y is (N,)
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            # Ensure y is long for CrossEntropy
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return {"data": self.X[idx], "labels": self.y[idx]}
        return {"data": self.X[idx]}

class REVEWrapper(nn.Module):
    """
    Wraps the HuggingFace REVE model.
    Freezes the backbone and adds a custom classification head.
    """
    def __init__(self, n_channels, n_timepoints, n_classes, freeze_backbone: bool = True, coords=None):
        super().__init__()
        # Load the backbone

        self.backbone = AutoModel.from_pretrained(
            "brain-bzh/reve-base",
            trust_remote_code=True,
            dtype="auto",
        )

        # Optionally freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = not freeze_backbone

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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, n_classes),
        )

    def forward(self, x, pos):
        # REVE expects (x, pos)
        # x shape: [Batch, Channels, Time]
        # pos shape: [Batch, Channels, EmbeddingDim]
        
        # Pass through frozen backbone
        # Note: We rely on the backbone's internal forward which likely returns the hidden states
        pos = pos.to(x.device)
        features = self.backbone(x, pos)
        
        # Pass through classifier
        logits = self.classifier(features)
        return logits

    def extract_features(self, x, pos):
        with torch.no_grad():
            pos = pos.to(x.device)
            features = self.backbone(x, pos)
        return features.reshape(features.shape[0], -1)

    def classify_features(self, features):
        return self.classifier(features)


class REVEBenchmarkModel(AbstractModel):
    def __init__(self, freeze_backbone: bool = True, linear_probe: bool = False):
        super().__init__("REVEModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_probe = linear_probe
        self.freeze_backbone = freeze_backbone or linear_probe

        # Load the position bank once
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            dtype="auto",
        )

        # Build case-insensitive lookup from the bank's own vocabulary
        bank_names = self.pos_bank.get_all_positions()
        self._bank_vocab = set(bank_names)
        self._upper_to_bank = {name.upper(): name for name in bank_names}

        self.model = None
        self.common_channels = None
        self._ch_keep = None
        self.task_name = None
        self.label_encoder = None

    def _normalize_ch_name(self, name: str) -> str:
        """Map a channel/electrode name to the position bank's expected casing."""
        if name in self._bank_vocab:
            return name
        return self._upper_to_bank.get(name.upper(), name)

    def _get_channel_coords(self, ch_names: List[str]):
        """Get 3D channel coordinates from position bank.

        Handles bipolar channels via midpoint averaging and drops unresolvable channels.

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
                    print(f"[REVE BCI] Dropping bipolar channel '{name}' — electrode(s) {missing} not in position bank")
            else:
                normed = self._normalize_ch_name(name)
                if normed in self._bank_vocab:
                    kept.append((i, [normed]))
                    query_names.append(normed)
                else:
                    print(f"[REVE BCI] Dropping channel '{name}' — not in position bank")

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
        return positions, ch_keep

    def _build_common_channels(self, meta: List[Dict]) -> List[str]:
        common = []
        for dataset_meta in meta:
            for ch in dataset_meta["channel_names"]:
                normed = ch.replace("EEG", "").strip().upper()
                if normed not in common:
                    common.append(normed)
        return common

    def _align_to_channels(self, data: np.ndarray, ch_names: List[str], common_channels: List[str]) -> np.ndarray:
        channel_to_index = {ch: idx for idx, ch in enumerate(common_channels)}
        aligned = np.zeros((data.shape[0], len(common_channels), data.shape[2]), dtype=data.dtype)

        for src_idx, ch in enumerate(ch_names):
            normed = ch.replace("EEG", "").strip().upper()
            dst_idx = channel_to_index.get(normed)
            if dst_idx is not None:
                aligned[:, dst_idx, :] = data[:, src_idx, :]

        return aligned

    def _encode_labels(self, labels: List[np.ndarray]):
        all_labels = np.concatenate(labels)
        if np.issubdtype(all_labels.dtype, np.number):
            self.label_encoder = None
            return labels, len(np.unique(all_labels))

        classes = list(dict.fromkeys(all_labels.tolist()))
        self.label_encoder = {label: idx for idx, label in enumerate(classes)}
        encoded = [
            np.asarray([self.label_encoder[label] for label in label_array], dtype=np.int64)
            for label_array in labels
        ]
        return encoded, len(classes)

    def _fit_linear_probe_cached(self, train_loader, n_epochs=10):
        assert self.model is not None

        cache_dir = create_temp_cache_dir("reve_bci_lp_")
        try:
            total_samples = len(train_loader.dataset)
            feature_dim = self.model.classifier[-1].in_features
            features_path = os.path.join(cache_dir, "train_features.dat")
            labels_path = os.path.join(cache_dir, "train_labels.dat")
            train_features = np.memmap(
                features_path, dtype=np.float16, mode="w+", shape=(total_samples, feature_dim)
            )
            train_labels = np.memmap(
                labels_path, dtype=np.int64, mode="w+", shape=(total_samples,)
            )

            idx = 0
            self.model.eval()
            for batch in tqdm(train_loader, desc="Cache REVE BCI embeddings", leave=False):
                data = batch["sample"].to(self.device)
                pos = batch["pos"].to(self.device)
                labels = batch["label"].cpu().numpy()
                feats = self.model.extract_features(data, pos).cpu().numpy().astype(np.float16)
                bsz = feats.shape[0]
                train_features[idx:idx + bsz] = feats
                train_labels[idx:idx + bsz] = labels
                idx += bsz

            train_features.flush()
            train_labels.flush()

            feat_loader = DataLoader(
                CachedArrayDataset(train_features, train_labels),
                batch_size=256,
                shuffle=True,
                num_workers=0,
            )

            optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(n_epochs):
                self.model.classifier.train()
                total_loss = 0.0
                correct = 0
                total = 0
                for feats, target in tqdm(feat_loader, desc=f"Epoch {epoch+1}", leave=False):
                    feats = feats.to(self.device).float()
                    target = target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model.classify_features(feats)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * target.size(0)
                    correct += (output.argmax(dim=1) == target).sum().item()
                    total += target.size(0)

                avg_loss = total_loss / total if total else 0.0
                avg_acc = correct / total if total else 0.0
                print(f"Epoch {epoch+1} - Acc: {avg_acc:.4f} - Loss: {avg_loss:.4f}")
                if self.wandb_run:
                    wandb_utils.log(
                        {
                            f"{self.name}/train_loss": avg_loss,
                            f"{self.name}/train_acc": avg_acc,
                        },
                        step=epoch + 1,
                    )
        finally:
            cleanup_temp_cache_dir(cache_dir)

    def _get_collate_fn(self, positions):
        """Creates the collate function using pre-computed position embeddings."""
        def collate(batch, positions):
            x_data = torch.stack([x["data"] for x in batch])
            batch_positions = positions.repeat(len(batch), 1, 1)
            batch_dict = {
                "sample": x_data,
                "pos": batch_positions
            }
            if "labels" in batch[0]:
                y_label = torch.tensor([x["labels"] for x in batch])
                batch_dict["label"] = y_label.long()
            return batch_dict

        return partial(collate, positions=positions)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        print("Initializing REVE Fit...")

        sample_X = X[0]
        n_samples, _, n_timepoints = sample_X.shape
        self.task_name = meta[0]["task_name"]
        y, n_classes = self._encode_labels(y)
        self.common_channels = self._build_common_channels(meta)

        # Align all datasets into a shared channel layout before filtering
        X = [
            self._align_to_channels(dataset_X, dataset_meta["channel_names"], self.common_channels)
            for dataset_X, dataset_meta in zip(X, meta)
        ]

        # Get robust channel coordinates (handles bipolar + drops unresolvable)
        positions, ch_keep = self._get_channel_coords(self.common_channels)
        self._ch_keep = ch_keep
        n_channels = positions.shape[0]

        # Filter channels if some were dropped
        if ch_keep is not None:
            X = [x[:, ch_keep, :] for x in X]
            print(f"[REVE BCI] Kept {n_channels} of {len(self.common_channels)} channels")

        # Initialize Model
        self.model = REVEWrapper(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes,
            freeze_backbone=self.freeze_backbone,
            coords=positions,
        ).to(self.device)

        # Prepare DataLoaders
        X_all = np.concatenate(X, axis=0)
        y_all = np.concatenate(y, axis=0)

        train_dataset = SimpleDataset(X_all, y_all)
        collate_fn = self._get_collate_fn(positions)

        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        n_epochs = 10
        print(f"Starting training for {n_epochs} epochs on {self.device}...")

        if self.linear_probe:
            self._fit_linear_probe_cached(train_loader, n_epochs=n_epochs)
            return

        self.model.train()
        if self.freeze_backbone:
            self.model.backbone.eval()
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
                output = self.model(data, pos)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

                pbar.set_postfix({'loss': total_loss/total})

            avg_loss = total_loss / len(train_loader)
            avg_acc = correct / total if total else 0
            print(f"Epoch {epoch+1} - Acc: {avg_acc:.4f} - Loss: {avg_loss:.4f}")

            if self.wandb_run:
                wandb_utils.log(
                    {
                        f"{self.name}/train_loss": avg_loss,
                        f"{self.name}/train_acc": avg_acc,
                    },
                    step=epoch + 1,
                )

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        self.model.eval()
        all_preds = []

        for i, (dataset_X, dataset_meta) in enumerate(zip(X, meta)):
            dataset_X = self._align_to_channels(dataset_X, dataset_meta["channel_names"], self.common_channels)
            if self._ch_keep is not None:
                dataset_X = dataset_X[:, self._ch_keep, :]

            dataset = SimpleDataset(dataset_X, y=None)
            positions, _ = self._get_channel_coords(self.common_channels)
            collate_fn = self._get_collate_fn(positions)

            loader = DataLoader(
                dataset,
                batch_size=64,
                shuffle=False,
                collate_fn=collate_fn
            )

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Predicting batch {i}", leave=False):
                    data = batch["sample"].to(self.device)
                    pos = batch["pos"].to(self.device)

                    output = self.model(data, pos)
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                    all_preds.append(preds)

        predictions = np.concatenate(all_preds, axis=0)
        if self.label_encoder is not None:
            inverse = {idx: label for label, idx in self.label_encoder.items()}
            return np.array([inverse[int(pred)] for pred in predictions])
        return np.array([reverse_map_label(int(pred), self.task_name) for pred in predictions])
