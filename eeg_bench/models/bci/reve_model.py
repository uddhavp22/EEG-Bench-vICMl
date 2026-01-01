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
from ...utils import wandb_utils




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
        
        # 1. Determine Input Shapes and Classes
        # We assume all datasets in the list have the same basic shape/channels for the task
        sample_X = X[0]
        sample_y = y[0]
        meta_data = meta[0]
        
        n_samples, n_channels, n_timepoints = sample_X.shape
        # Assuming y contains class indices 0..N-1
        # You might need np.unique(np.concatenate(y)) if indices are sparse
        n_classes = len(np.unique(np.concatenate(y)))
        
        channel_names = meta_data["channel_names"]

        # 2. Initialize Model
        self.model = REVEWrapper(
            n_channels=n_channels, 
            n_timepoints=n_timepoints, 
            n_classes=n_classes
        ).to(self.device)
        
        # 3. Prepare DataLoaders
        # Concatenate all datasets for training (or keep separate if you prefer epoch-loops)
        # Here we concatenate for simplicity as done in standard ML, 
        # but you can loop over list like LaBraM if needed.
        X_all = np.concatenate(X, axis=0)
        y_all = np.concatenate(y, axis=0)
        
        train_dataset = SimpleDataset(X_all, y_all)
        collate_fn = self._get_collate_fn(channel_names)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=64, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0 # Set >0 if on Linux/Mac
        )
        
        # 4. Optimizer
        # Only optimize the classifier head (model.classifier)
        optimizer = torch.optim.AdamW(self.model.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # 5. Training Loop
        self.model.train()
        n_epochs = 10 
        
        print(f"Starting training for {n_epochs} epochs on {self.device}...")
        
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
                
                # REVE forward pass requires (data, pos)
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
        
        # We iterate over the list because meta might differ (though unlikely for one task)
        # or just to handle memory chunks
        for i, (dataset_X, dataset_meta) in enumerate(zip(X, meta)):
            
            dataset = SimpleDataset(dataset_X, y=None)
            collate_fn = self._get_collate_fn(dataset_meta["channel_names"])
            
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
                    
        return np.concatenate(all_preds, axis=0)
