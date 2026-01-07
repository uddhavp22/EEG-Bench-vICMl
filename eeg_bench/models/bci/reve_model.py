import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
import numpy as np
import os
from typing import List, Dict
from tqdm import tqdm
from ..abstract_model import AbstractModel
from ...utils import wandb_utils
from .LaBraM.make_dataset import make_dataset
from .LaBraM.utils_2 import reverse_map_label, n_unique_labels
from joblib import Memory
from ...config import get_config_value

class REVEWrapper(nn.Module):
    """
    Wraps the HuggingFace REVE model.
    Freezes the backbone and adds a custom classification head.
    """
    def __init__(self, n_channels, n_timepoints, n_classes, hidden_dim=None):
        super().__init__()
        # Load the backbone

        # Get HuggingFace token from environment variable for security
        hf_token = os.environ.get('HF_TOKEN', None)
        self.backbone = AutoModel.from_pretrained(
            "brain-bzh/reve-base",
            trust_remote_code=True,
            torch_dtype="auto",
            token=hf_token
        )

        if hidden_dim is None:
            config = getattr(self.backbone, "config", None)
            hidden_dim = (
                getattr(config, "hidden_size", None)
                or getattr(config, "hidden_dim", None)
                or getattr(config, "d_model", None)
                or 512
            )

        self.hidden_dim = hidden_dim

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Define the classification head
        # REVE output is [Batch, Channels, Time, HiddenDim]
        # We average over time, then flatten channels
        # This is more efficient than flattening everything
        input_dim = n_channels * hidden_dim

        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, n_classes),
        )

    def forward(self, x, pos):
        # REVE expects (x, pos)
        # x shape: [Batch, Channels, Time]
        # pos shape: [Batch, Channels, EmbeddingDim]

        # Pass through frozen backbone
        # Note: We rely on the backbone's internal forward which likely returns the hidden states
        features = self.backbone(x, pos)
        if isinstance(features, dict):
            features = features.get("last_hidden_state", features.get("features", features))

        # Average over time dimension for more efficient representation
        # features shape: [Batch, Channels, Time, HiddenDim] -> [Batch, Channels, HiddenDim]
        features = features.mean(dim=2)

        # Pass through classifier
        logits = self.classifier(features)
        return logits


class REVEBenchmarkModel(AbstractModel):
    def __init__(self):
        super().__init__("REVEModel")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

        # Load the position bank once
        # Get HuggingFace token from environment variable for security
        hf_token = os.environ.get('HF_TOKEN', None)
        self.pos_bank = AutoModel.from_pretrained(
            "brain-bzh/reve-positions",
            trust_remote_code=True,
            torch_dtype="auto",
            token=hf_token
        ).to(self.device)
        self.model = None

    def _get_coords(self, ch_names):
        """Get channel position embeddings from REVE position bank."""
        clean_names = [c.replace("EEG", "").strip().upper() for c in ch_names]
        c = self.pos_bank(clean_names)
        if isinstance(c, dict):
            c = c.get("positions", c.get("coords", c.get("last_hidden_state")))
        if c.dim() == 3:
            c = c.squeeze(0)
        return c.float().to(self.device)

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        print("Initializing REVE BCI Fit with proper preprocessing...")

        task_name = meta[0]["task_name"]
        num_classes = n_unique_labels(task_name)

        # Apply REVE-specific preprocessing (200 Hz, uV units, no scaling)
        datasets = [self.cache.cache(make_dataset)(X_, y_, task_name, m_["sampling_frequency"], m_["channel_names"], train=True, split_size=0.15, model_name="REVEModel")
                    for X_, y_, m_ in zip(X, y, meta)]

        dataset_train_list = [dataset[0] for dataset in datasets]
        dataset_val_list = [dataset[1] for dataset in datasets]
        dataset_train_list = [dataset for dataset in dataset_train_list if len(dataset) > 0]
        dataset_val_list = [dataset for dataset in dataset_val_list if len(dataset) > 0]
        ch_names_list = [dataset.ch_names for dataset in dataset_train_list]

        # Initialize model based on preprocessed data shape
        if len(dataset_train_list) > 0:
            sample_data = dataset_train_list[0][0]  # Get first sample
            n_channels = sample_data.shape[0]
            n_timepoints = sample_data.shape[1]
            self.model = REVEWrapper(n_channels, n_timepoints, num_classes).to(self.device)
        else:
            print("No training data available")
            return

        # Prepare DataLoaders
        batch_size = 64
        num_workers = 8
        train_loader_list = [
            DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
            for train_dataset in dataset_train_list
        ]
        valid_loader_list = [
            DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
            for valid_dataset in dataset_val_list
        ]

        # Training setup
        optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        max_epochs = 30
        print(f"Starting REVE BCI training for {max_epochs} epochs...")

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            epoch_train_loss = 0.0
            epoch_train_correct = 0
            epoch_train_total = 0

            for train_loader, ch_names in zip(train_loader_list, ch_names_list):
                coords = self._get_coords(ch_names)

                for x, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
                    x = x.to(self.device)
                    y_batch = y_batch.to(self.device).argmax(dim=1)  # Convert one-hot to class indices
                    cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)

                    optimizer.zero_grad()
                    logits = self.model(x, cb)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item() * x.size(0)
                    preds = torch.argmax(logits, dim=1)
                    epoch_train_correct += (preds == y_batch).sum().item()
                    epoch_train_total += x.size(0)

            avg_train_loss = epoch_train_loss / epoch_train_total if epoch_train_total > 0 else 0
            avg_train_acc = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0

            # Validation
            if valid_loader_list:
                self.model.eval()
                epoch_val_loss = 0.0
                epoch_val_correct = 0
                epoch_val_total = 0

                with torch.no_grad():
                    for valid_loader, ch_names in zip(valid_loader_list, ch_names_list):
                        coords = self._get_coords(ch_names)

                        for x, y_batch in tqdm(valid_loader, desc=f"Val {epoch}", leave=False):
                            x = x.to(self.device)
                            y_batch = y_batch.to(self.device).argmax(dim=1)
                            cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)

                            logits = self.model(x, cb)
                            loss = criterion(logits, y_batch)

                            epoch_val_loss += loss.item() * x.size(0)
                            preds = torch.argmax(logits, dim=1)
                            epoch_val_correct += (preds == y_batch).sum().item()
                            epoch_val_total += x.size(0)

                avg_val_loss = epoch_val_loss / epoch_val_total if epoch_val_total > 0 else 0
                avg_val_acc = epoch_val_correct / epoch_val_total if epoch_val_total > 0 else 0

                print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

                if self.wandb_run:
                    wandb_utils.log(
                        {
                            f"{self.name}/train_loss": avg_train_loss,
                            f"{self.name}/train_acc": avg_train_acc,
                            f"{self.name}/val_loss": avg_val_loss,
                            f"{self.name}/val_acc": avg_val_acc,
                        },
                        step=epoch,
                    )
            else:
                print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")

                if self.wandb_run:
                    wandb_utils.log(
                        {
                            f"{self.name}/train_loss": avg_train_loss,
                            f"{self.name}/train_acc": avg_train_acc,
                        },
                        step=epoch,
                    )

    @torch.no_grad()
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        task_name = meta[0]["task_name"]
        self.model.eval()

        # Apply REVE-specific preprocessing (200 Hz, uV units, no scaling)
        dataset_test_list = [self.cache.cache(make_dataset)(X_, None, task_name, meta_["sampling_frequency"], meta_["channel_names"], train=False, split_size=0, model_name="REVEModel")
                             for X_, meta_ in zip(X, meta)]
        dataset_test_list = [dataset for dataset in dataset_test_list if len(dataset) > 0]
        ch_names_list = [dataset.ch_names for dataset in dataset_test_list]

        batch_size = 64
        test_loader_list = [DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False) for test_dataset in dataset_test_list]

        predictions = []
        for test_loader, ch_names in zip(test_loader_list, ch_names_list):
            coords = self._get_coords(ch_names)
            preds_all = []
            for x in tqdm(test_loader, desc="REVE BCI Predicting", leave=False):
                x = x.to(self.device)
                cb = coords.unsqueeze(0).expand(x.size(0), -1, -1)
                logits = self.model(x, cb)
                preds_all.append(torch.argmax(logits, dim=1).cpu())
            predictions.append(torch.cat(preds_all, dim=0))

        predictions = torch.cat(predictions, dim=0).numpy()
        mapped_pred = np.array([reverse_map_label(idx, task_name) for idx in predictions])
        return mapped_pred
