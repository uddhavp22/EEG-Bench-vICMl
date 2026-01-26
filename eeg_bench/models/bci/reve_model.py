import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModel
import numpy as np
import os
from typing import List, Dict, Union
from tqdm import tqdm
import logging
from functools import partial
from ..abstract_model import AbstractModel
from ...utils import wandb_utils
from ...utils.utils import configure_torch_backend_for_speed, create_temp_cache_dir, cleanup_temp_cache_dir
from .LaBraM.make_dataset import make_dataset_reve
from .LaBraM.utils_2 import n_unique_labels, calc_class_weights




# Assuming AbstractModel is available in your path
# from abstract_model import AbstractModel 

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
    def __init__(self, n_channels, n_timepoints, n_classes, hidden_dim=512):
        super().__init__()
        # Load the backbone
        self.hidden_dim = hidden_dim

        self.backbone = AutoModel.from_pretrained(
            "brain-bzh/reve-base", 
            trust_remote_code=True, 
            torch_dtype="auto",
        )
        
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Define the classification head
        # REVE output is [Batch, Channels, Time, HiddenDim]
        # We flatten this to [Batch, Channels * Time * HiddenDim]
        input_dim = n_channels * n_timepoints * hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.RMSNorm(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, n_classes),
        )

    def forward(self, x, pos):
        # REVE expects (x, pos)
        # x shape: [Batch, Channels, Time]
        # pos shape: [Batch, Channels, EmbeddingDim]
        
        # Pass through frozen backbone
        # Note: We rely on the backbone's internal forward which likely returns the hidden states
        with torch.no_grad():
            features = self.backbone(x, pos)
        
        # Pass through classifier
        logits = self.classifier(features)
        return logits


class REVEBenchmarkModel(AbstractModel):
    def __init__(self):
        super().__init__("REVEModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the position bank once
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions", 
            trust_remote_code=True, 
            torch_dtype="auto",
        )
        self.model = None

    def _get_collate_fn(self, channel_names):
        """
        Creates the specific collate function required by REVE.
        Maps channel names -> REVE Position Embeddings.
        """
        # Get embeddings for the specific channels of this task
        # shape: [1, n_channels, embed_dim]
        raw_positions = self.pos_bank(channel_names)
        if isinstance(raw_positions, dict):
            raw_positions = raw_positions.get(
                "positions", raw_positions.get("coords", raw_positions.get("last_hidden_state"))
            )
        if raw_positions.dim() == 3:
            raw_positions = raw_positions.squeeze(0)
        
        def collate(batch, positions):
            # Stack data: [Batch, Channels, Time]
            x_data = torch.stack([x["data"] for x in batch])
            
            # Repeat positions for the batch: [Batch, Channels, EmbedDim]
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
        print("Initializing REVE Fit...")

        # 1. Get metadata
        meta_data = meta[0]
        task_name = meta_data["task_name"]
        channel_names = meta_data["channel_names"]
        n_classes = n_unique_labels(task_name)

        # 2. Preprocess data using REVE-specific pipeline (200 Hz, 0.5-99.5 Hz bandpass, z-score + clip)
        print("[REVE] Applying REVE-specific preprocessing...")
        datasets = [
            make_dataset_reve(
                X_, y_, task_name,
                m_["sampling_frequency"],
                m_["channel_names"],
                train=True,
                split_size=0.15
            )
            for X_, y_, m_ in zip(X, y, meta)
        ]

        # Get train datasets
        dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]

        # Get processed channel names from first dataset
        if dataset_train_list:
            channel_names = dataset_train_list[0].ch_names

        # Get shape from processed data
        sample_data = dataset_train_list[0].data
        n_channels = sample_data.shape[1]
        n_timepoints = sample_data.shape[2]

        # 3. Initialize Model with processed dimensions
        self.model = REVEWrapper(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        ).to(self.device)
        print(f"[REVE] Initialized model with {n_channels} channels and {n_timepoints} timepoints (200 Hz)")

        # 4. Prepare DataLoaders from preprocessed datasets
        X_all = np.concatenate([d.data for d in dataset_train_list], axis=0)
        y_all = np.concatenate([d.labels for d in dataset_train_list], axis=0)
        # Convert one-hot back to class indices
        if y_all.ndim > 1:
            y_all = np.argmax(y_all, axis=1)

        configure_torch_backend_for_speed()

        train_dataset = SimpleDataset(X_all, y_all)
        collate_fn = self._get_collate_fn(channel_names)

        num_workers = 2
        loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=64 if self.chunk_len_s is None else 64, 
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs
        )
        
        # 4. Optimizer
        # Only optimize the classifier head (model.classifier)
        optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # 5. Training Loop
        self.model.train()
        n_epochs = 10 

        cache_dir = create_temp_cache_dir("reve_lp_")
        try:
            feature_dim = n_channels * n_timepoints * self.model.hidden_dim
            train_count = len(train_dataset)
            features_path = os.path.join(cache_dir, "train_features.dat")
            labels_path = os.path.join(cache_dir, "train_labels.dat")
            features = np.memmap(
                features_path, dtype=np.float16, mode="w+", shape=(train_count, feature_dim)
            )
            labels = np.memmap(
                labels_path, dtype=np.int64, mode="w+", shape=(train_count,)
            )

            idx = 0
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Cache train embeddings", leave=False):
                    data = batch["sample"].to(self.device)
                    pos = batch["pos"].to(self.device)
                    target = batch["label"].cpu().numpy()

                    feats = self.model.backbone(data, pos)
                    feats = feats.reshape(feats.size(0), -1).cpu().numpy().astype(np.float16, copy=False)
                    bsz = feats.shape[0]
                    features[idx:idx + bsz] = feats
                    labels[idx:idx + bsz] = target
                    idx += bsz

            features.flush()
            labels.flush()

            train_dataset_cached = TensorDataset(
                torch.from_numpy(features),
                torch.from_numpy(labels)
            )
            train_feat_loader = DataLoader(
                train_dataset_cached,
                batch_size=256,
                shuffle=True,
                num_workers=0
            )

            print(f"Starting training for {n_epochs} epochs on {self.device}...")

            for epoch in range(n_epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                pbar = tqdm(train_feat_loader, desc=f"Epoch {epoch+1}", leave=False)
                for feats, target in pbar:
                    feats = feats.to(self.device, dtype=torch.float32)
                    target = target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model.classifier(feats)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    preds = torch.argmax(output, dim=1)
                    correct += (preds == target).sum().item()
                    total += target.size(0)
                    
                    pbar.set_postfix({'loss': total_loss/total})

                avg_loss = total_loss / len(train_feat_loader)
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
        finally:
            cleanup_temp_cache_dir(cache_dir)

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        self.model.eval()
        all_preds = []
        
        # We iterate over the list because meta might differ (though unlikely for one task)
        # or just to handle memory chunks
        for i, (dataset_X, dataset_meta) in enumerate(zip(X, meta)):
            
            dataset = SimpleDataset(dataset_X, y=None)
            collate_fn = self._get_collate_fn(dataset_meta["channel_names"])
            num_workers = 2
            loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
            if num_workers > 0:
                loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

            loader = DataLoader(
                dataset, 
                batch_size=64, 
                shuffle=False, 
                collate_fn=collate_fn,
                **loader_kwargs
            )
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Predicting batch {i}", leave=False):
                    data = batch["sample"].to(self.device)
                    pos = batch["pos"].to(self.device)
                    
                    output = self.model(data, pos)
                    preds = torch.argmax(output, dim=1).cpu().numpy()
                    all_preds.append(preds)
                    
        return np.concatenate(all_preds, axis=0)
