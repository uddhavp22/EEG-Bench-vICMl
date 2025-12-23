from ..abstract_model import AbstractModel
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

# Import the Abstract Backbone we defined previously
# (Assuming you saved the BCI abstract class in a file, or you can copy the config logic here)
from models.bci.EEGLeJEPA_model import AbstractLeJEPAForBCI 

class SimpleClinicalDataset(Dataset):
    """
    Dataset wrapper that handles the channel locations required by LeJEPA.
    """
    def __init__(self, X_flat, y_flat, channel_coords):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_flat]
        self.y = torch.tensor(y_flat, dtype=torch.long) if y_flat is not None else None
        self.coords = torch.tensor(channel_coords, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # LeJEPA expects: x, channel_locations
        item = {
            "x": self.X[idx],
            "channel_locations": self.coords # Broadcasted later or passed directly
        }
        if self.y is not None:
            item["y"] = self.y[idx]
        return item

class LEJEPAClinicalModel(AbstractModel):
    def __init__(
        self, 
        num_classes: int = 2, 
        num_labels_per_chunk: Optional[int] = None,
        freeze_encoder: bool = True
    ):
        super().__init__("LEJEPA-Clinical")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # Initialize the backbone using the Abstract Class from before
        # This loads the hardcoded config (DIM=384, etc.)
        self.backbone_wrapper = ConcreteLeJEPA(
            num_classes=num_classes, 
            freeze_encoder=freeze_encoder
        )
        self.model = self.backbone_wrapper.to(self.device)

    def _flatten_data(self, X, y=None):
        """Flattens List[List[np.ndarray]] -> List[np.ndarray]"""
        X_flat = []
        y_flat = []
        
        # Check if X is nested (List of Lists)
        if len(X) > 0 and isinstance(X[0], list):
            for i, sublist in enumerate(X):
                X_flat.extend(sublist)
                if y is not None:
                    # Handle y being nested or flat
                    if isinstance(y[i], (list, np.ndarray)) and len(y[i]) == len(sublist):
                         y_flat.extend(y[i])
                    else:
                        y_flat.extend([y[i]] * len(sublist))
        else:
            X_flat = X
            y_flat = y if y is not None else []

        if y is not None:
            return X_flat, np.array(y_flat)
        return X_flat, None

    def _get_channel_coords(self, channel_names: List[str]) -> np.ndarray:
        """
        Maps channel names to 3D coordinates.
        You need a standard 10-20 dictionary here.
        """
        # Simplified Mock 10-20 Dictionary (You should replace this with the real one from the benchmark utils)
        # The benchmark likely has a utility for this: e.g. from eeg_bench.utils import get_coords
        standard_coords = {
            "Fp1": [ -0.03, 0.08, 0.03], "Fp2": [ 0.03, 0.08, 0.03],
            "F7": [-0.07, 0.04, 0.01], "F3": [-0.04, 0.05, 0.04],
            "Fz": [0.0, 0.06, 0.06], "F4": [0.04, 0.05, 0.04], "F8": [0.07, 0.04, 0.01],
            # ... add full list ...
            "C3": [-0.05, 0.0, 0.08], "Cz": [0.0, 0.0, 0.1], "C4": [0.05, 0.0, 0.08],
            # Fallback for unknown
        }
        
        coords = []
        for name in channel_names:
            # SANITIZE: "EEG Fp1-REF" -> "Fp1"
            clean_name = name.replace('EEG', '').replace('-Ref', '').replace(' ', '').strip()
            if clean_name.upper().startswith('FP'): clean_name = 'Fp' + clean_name[2:]
            
            if clean_name in standard_coords:
                coords.append(standard_coords[clean_name])
            else:
                # Default/Mean coord if unknown (risky but prevents crash)
                coords.append([0, 0, 0]) 
                
        return np.array(coords)

    def fit(self, X: List, y: List, meta: List[Dict]) -> None:
        # 1. Flatten Data
        X_flat, y_flat = self._flatten_data(X, y)
        
        # 2. Prepare Metadata / Coords
        # Use first dataset's meta for channel names (assuming consistency)
        channel_coords = self._get_channel_coords(meta[0]["channel_names"])
        
        # 3. Create Dataset/Loader
        dataset = SimpleClinicalDataset(X_flat, y_flat, channel_coords)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 4. Train Loop
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        print(f"Training LEJEPA-Clinical on {len(X_flat)} samples...")
        
        for epoch in range(5): # Short epochs for benchmark
            for batch in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
                x = batch["x"].to(self.device)
                coords = batch["channel_locations"].to(self.device)
                targets = batch["y"].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(x, coords)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

    def predict(self, X: List, meta: List[Dict]) -> np.ndarray:
        X_flat, _ = self._flatten_data(X)
        channel_coords = self._get_channel_coords(meta[0]["channel_names"])
        
        dataset = SimpleClinicalDataset(X_flat, None, channel_coords)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                x = batch["x"].to(self.device)
                coords = batch["channel_locations"].to(self.device)
                
                logits = self.model(x, coords)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        
        return np.concatenate(preds)

# --- Concrete Implementation of the Abstract Backbone ---
class ConcreteLeJEPA(AbstractLeJEPAForBCI):
    def __init__(self, num_classes, freeze_encoder=True):
        super().__init__(freeze_encoder=freeze_encoder)
        # Simple Linear Head on top of the 384-dim CLS token
        self.head = nn.Linear(384, num_classes)
        
    def forward(self, x, channel_locations):
        # x: (B, C, T)
        # 1. Extract Features (CLS token)
        # Note: We duplicate the coords for the batch if needed, or pass as is 
        # depending on how your backbone expects it. 
        # Usually backbone expects (B, C, 3).
        if channel_locations.dim() == 2: # (C, 3)
            B = x.shape[0]
            channel_locations = channel_locations.unsqueeze(0).expand(B, -1, -1)
            
        feats = self.extract_features(x, channel_locations)
        
        # 2. Classify
        return self.head(feats)