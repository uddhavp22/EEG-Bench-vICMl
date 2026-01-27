
from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Literal, Optional
import numpy as np
from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse, LaBraMDataset2, make_multilabels
from .LaBraM import utils
import torch
from timm.models import create_model
import numpy as np
import hashlib
import json
from mne.io import BaseRaw
from .LaBraM import modeling_finetune # important to load the models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
from ...config import get_config_value
from ...utils import wandb_utils
import gc
from collections import Counter
import logging
import os
import requests
from pathlib import Path

def check_and_download_pretrained_model():
    chkpt_dir = Path(get_config_value("chkpt"))
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir, exist_ok=True)
    encoder_path = chkpt_dir / "labram-base.pth"
    if not os.path.exists(encoder_path):
        print("Labram-Base file not found. Downloading labram-base.pth ...")
        url = "https://github.com/935963004/LaBraM/raw/refs/heads/main/checkpoints/labram-base.pth"
        response = requests.get(url, stream=True)
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        with open(encoder_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return encoder_path

class LaBraMBCIModel(nn.Module):
    def __init__(self, num_classes, num_labels_per_chunk, device, chunks, freeze_encoder: bool = True):
        super().__init__()
        self.device = device
        self.chunks = chunks
        checkpoint = torch.load(check_and_download_pretrained_model(), weights_only=False)
        new_checkpoint = {}
        for k,v in checkpoint['model'].items():
            if k.startswith('student.'):
                new_checkpoint[k[len('student.'):]] = v
        model = create_model("labram_base_patch200_200",
                                # checkpoint_path= ,
                                qkv_bias=False,
                                rel_pos_bias=True,
                                num_classes=num_classes,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                attn_drop_rate=0.0,
                                drop_block_rate=None,
                                use_mean_pooling=True,
                                init_scale=0.001,
                                use_rel_pos_bias=True,
                                use_abs_pos_emb=True,
                                init_values=0.1,)
        missing, unexpected = model.load_state_dict(new_checkpoint, strict=False)
        print("Missing keys", missing)
        print("Unexpected", unexpected)
        if freeze_encoder:
            # 1. Turn off gradients for EVERYTHING
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        self.feature = model
        self.is_multilabel_task = num_labels_per_chunk is not None
        self.head = nn.Linear(200, num_classes * (num_labels_per_chunk if self.is_multilabel_task else 1))
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder

    def extract_features(self, x, input_chans):
        B, C, T = x.shape

        if self.chunks is not None and (self.chunks <= 10 or self.is_multilabel_task):
            x = x.to(self.device, non_blocking=True)
            if T % 200 != 0: 
                x = x[:,:,0:T-T%200]
                T = T - T % 200
            x = x.reshape((B, C, T // 200, 200))
            x = x / 100
            
            tokens = self.feature.forward_features(x, input_chans=input_chans, return_all_tokens=False)
            return tokens.flatten(1)

        if len(input_chans) <= 24:
            chunk_length = 2000
        elif len(input_chans) <= 32:
            chunk_length = 1600
        elif len(input_chans) <= 50:
            chunk_length = 1000
        elif len(input_chans) <= 64:
            chunk_length = 800
        else:
            raise ValueError("Unsupported input channel configuration: {}".format(input_chans))

        n_chunks = T // chunk_length
        if n_chunks < 1:
            raise ValueError(
                "Recording too short: expected at least one chunk of length {}, got T={}".format(chunk_length, T)
            )
        # Crop extra samples to have only full chunks
        T_new = n_chunks * chunk_length
        x = x[:, :, :T_new]  # shape: (B, C, T_new)

        # Reshape to split recording into chunks:
        x = x.reshape(B, C, n_chunks, chunk_length)
        x = x.permute(0, 2, 1, 3)  # shape: (B, n_chunks, C, chunk_length)

        # Merge batch and chunks dimensions to process all chunks together:
        x = x.reshape(B * n_chunks, C, chunk_length)
        
        # Tokenize each chunk: each token is 200 samples.
        tokens = x.reshape(B * n_chunks, C, chunk_length // 200, 200)
        tokens = tokens / 100.0

        tokens = tokens.to(self.device, non_blocking=True)

        # Extract features for each chunk using the pre-trained feature extractor.
        # Expected output shape: (B * n_chunks, feature_dim)
        chunk_features = self.feature.forward_features(tokens, input_chans=input_chans, return_all_tokens=False)
        feature_dim = chunk_features.shape[-1]
        
        # Reshape back to separate recordings and chunks: (B, n_chunks, feature_dim)
        chunk_features = chunk_features.view(B, n_chunks, feature_dim)
        
        # Aggregate features across chunks by averaging (mean pooling)
        aggregated_features = chunk_features.mean(dim=1)  # shape: (B, feature_dim)

        return aggregated_features

    def classify_from_features(self, features):
        logits = self.head(features)
        if self.is_multilabel_task:
            logits = logits.reshape((features.shape[0], self.num_classes, -1))
        return logits

    def forward(self, x, input_chans):
        features = self.extract_features(x, input_chans)
        logits = self.classify_from_features(features)
        return features, logits


def build_embedding_cache(model, dataloader, input_chans, split_name: str):
    """
    Runs the (frozen) encoder over a dataloader once and stores the resulting embeddings + labels.
    """
    cached_features = []
    cached_labels = []
    model.eval()
    model.feature.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Embedding cache ({split_name})", leave=True):
            x, y, channels = batch
            batch_input_chans = input_chans
            if channels != -1 and channels[0] != -1:
                channels = [ch_arr[0] for ch_arr in channels]
                batch_input_chans = utils.get_input_chans(channels)

            features = model.extract_features(x, batch_input_chans)
            cached_features.append(features.cpu())

            if isinstance(y, torch.Tensor):
                y_tensor = y
            else:
                y_tensor = torch.as_tensor(y)
            cached_labels.append(y_tensor.cpu())

            del x
            torch.cuda.empty_cache()

    features_tensor = torch.cat(cached_features, dim=0)
    labels_tensor = torch.cat(cached_labels, dim=0)
    del cached_features, cached_labels
    print(f"[Cache] Stored {features_tensor.shape[0]} {split_name} embeddings of dim {features_tensor.shape[1]}")
    return TensorDataset(features_tensor, labels_tensor)

def train_epoch(model, dataloader, optimizer, scheduler, device, input_chans, cached_features: bool = False):
    model.train()
    if hasattr(model, "feature"):
        model.feature.eval()  # keep encoder deterministic during LP fine-tuning
    running_loss, running_corrects, total_samples = 0.0, 0, 0

    print([n for n, p in model.named_parameters() if p.requires_grad])

    for batch in tqdm(dataloader, desc="Training", leave=True):
        if cached_features:
            features, y = batch
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.classify_from_features(features.to(device))
            batch_size = features.size(0)
        else:
            x, y, channels = batch
            y = y.to(device)

            batch_input_chans = input_chans
            if channels != -1 and channels[0] != -1:
                channels = [ch_arr[0] for ch_arr in channels]
                batch_input_chans = utils.get_input_chans(channels)

            optimizer.zero_grad(set_to_none=True)
            _, logits = model(x, batch_input_chans)
            batch_size = x.size(0)
        
        loss = model.loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        running_corrects += torch.sum(preds == y).item()
        total_samples += batch_size

        if cached_features:
            del features, y, logits, loss
        else:
            del x, y, logits, loss  # Delete tensors no longer needed
        gc.collect()  # Invoke garbage collection
        torch.cuda.empty_cache()  # Clear cached memory on GPU
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, device, input_chans, cached_features: bool = False):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=True):
            if cached_features:
                features, y = batch
                y = y.to(device)
                logits = model.classify_from_features(features.to(device))
                batch_size = features.size(0)
            else:
                x, y, channels = batch
                y = y.to(device)
                
                batch_input_chans = input_chans
                if channels != -1 and channels[0] != -1:
                    channels = [ch_arr[0] for ch_arr in channels]
                    batch_input_chans = utils.get_input_chans(channels)
                
                _, logits = model(x, batch_input_chans)
                batch_size = x.size(0)

            loss = model.loss_fn(logits, y)
            
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            running_corrects += torch.sum(preds == y).item()
            total_samples += batch_size
            
            all_labels.append(y.cpu())
            all_logits.append(logits.cpu())

            if cached_features:
                del features, y, logits
            else:
                del x, y, logits  # Delete tensors no longer needed
            torch.cuda.empty_cache()  # Clear cached memory on GPU
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    # Concatenate predictions and labels
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    if model.is_multilabel_task:
        # additionally flatten, i.e. reduce multilabel to single-classification task
        all_labels = all_labels.flatten()
        all_logits = all_logits.transpose(-1, -2).flatten(0, 1)
    
    # Compute additional metrics using get_metrics
    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
    results = utils.get_metrics(all_logits.numpy(), all_labels.numpy(), metrics, False)
    
    return epoch_loss, epoch_acc, results

def inference(model, dataloader, device, input_chans):
    model.eval()
    predictions = []
    indices = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=True):
            x, idx, channels  = batch
            if channels != -1 and channels[0] != -1:
                channels = [ch_arr[0] for ch_arr in channels]
                input_chans = utils.get_input_chans(channels)

            # x = x.to(device) will be done in the model
            _, logits = model(x, input_chans)
            preds = torch.argmax(logits, dim=1)
            predictions.append(preds.cpu())
            indices.append(idx)

            del x, idx, logits  # Delete tensors no longer needed
            torch.cuda.empty_cache()  # Clear cached memory on GPU
    predictions = torch.cat(predictions, dim=0).cpu()
    indices = torch.cat(indices, dim=0).cpu()
    return predictions, indices

class LaBraMModel(AbstractModel):
    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        freeze_encoder: bool = True,
        cache_encoder_outputs: bool = True,
    ):
        super().__init__("LaBraMModel")
        print("inside init LaBraMModel")
        assert torch.cuda.is_available(), "CUDA is not available"

        self.chunk_len_s = None if num_labels_per_chunk is None else 16
        self.use_cache = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels_per_chunk = num_labels_per_chunk
        self.model = LaBraMBCIModel(num_classes=num_classes, num_labels_per_chunk=num_labels_per_chunk, device=self.device, chunks=self.chunk_len_s, freeze_encoder=freeze_encoder).to(self.device)
        self.save = False
        self.cache_encoder_outputs = cache_encoder_outputs and freeze_encoder
        if cache_encoder_outputs and not freeze_encoder:
            print("[Warn] cache_encoder_outputs requested but encoder is trainable; disabling cache.")
        self.cached_batch_multiplier = 4
        self.supports_full_dataset_cache = self.cache_encoder_outputs
        self.last_data_stats = None

    def fit(
        self,
        X: List[np.ndarray | List[BaseRaw]],
        y: List[np.ndarray | List[str]],
        meta: List[Dict],
        subset_fraction: float = 1.0,
        subset_seed: Optional[int] = None,
        subset_indices: Optional[List[List[int]]] = None,
    ) -> None:
        if not self.cache_encoder_outputs or not self.supports_full_dataset_cache:
            return self._fit_standard(X, y, meta)

        if subset_indices is None:
            subset_indices = [list(range(len(dataset))) for dataset in X]

        return self._fit_with_cache(
            X,
            y,
            meta,
            subset_fraction=subset_fraction,
            subset_seed=subset_seed,
            subset_indices=subset_indices,
        )

    def _fit_standard(self, X: List[np.ndarray|List[BaseRaw]], y: List[np.ndarray|List[str]], meta: List[Dict]) -> None:  
        print("inside fit")
        task_name = meta[0]["task_name"]
        
        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        print("class_weights", class_weights)
        self.model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        dataset_train  = make_dataset_2(X, y, meta, task_name, self.name, self.chunk_len_s, is_train=True, use_cache=self.use_cache)

        # Check if dataset has any samples
        if len(dataset_train) == 0:
            raise ValueError(f"Dataset has 0 samples after processing. Cannot train model.")

        val_split = 0.2
        if val_split is not None:
            dataset_train, dataset_val = dataset_train.split_train_val(val_split)
        else:
            dataset_val = None

        # Verify we still have training samples after split
        if len(dataset_train) == 0:
            raise ValueError(f"Training set has 0 samples after train/val split. Try using more training data.")

        del X, y, meta
        gc.collect()
        torch.cuda.empty_cache()

        ch_names_train = dataset_train.ch_names
        if dataset_val is not None:
            ch_names_val = dataset_val.ch_names


        if self.chunk_len_s is None:
            batch_size = 1
        else: 
            batch_size = 64
            
        num_workers = 8  # Increase this based on your CPU core count

        def make_loader(dataset, shuffle):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=True,
            )

        train_loader = make_loader(dataset_train, shuffle=True)
        valid_loader = make_loader(dataset_val, shuffle=False) if dataset_val is not None else None

        train_input_chans = utils.get_input_chans(ch_names_train)
        val_input_chans = utils.get_input_chans(ch_names_val) if dataset_val is not None else None

        using_cached_train = False
        using_cached_val = False
        if self.cache_encoder_outputs:
            cache_loader = make_loader(dataset_train, shuffle=False)
            cached_train_ds = build_embedding_cache(self.model, cache_loader, train_input_chans, split_name="train")
            cache_batch_size = min(1024, max(batch_size * self.cached_batch_multiplier, batch_size))
            train_loader = DataLoader(
                cached_train_ds,
                batch_size=cache_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            using_cached_train = True
            del cache_loader
            torch.cuda.empty_cache()

            if dataset_val is not None:
                cache_val_loader = make_loader(dataset_val, shuffle=False)
                cached_val_ds = build_embedding_cache(self.model, cache_val_loader, val_input_chans, split_name="val")
                valid_loader = DataLoader(
                    cached_val_ds,
                    batch_size=cache_batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
                using_cached_val = True
                del cache_val_loader
                torch.cuda.empty_cache()

        max_epochs = 30
        steps_per_epoch = len(train_loader)
        max_lr = 4e-4
        
        # Set up optimizer and OneCycleLR scheduler
        # Filter parameters to ONLY include those where requires_grad=True (i.e., self.head)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params, # Optimized
            lr=1e-6, 
            weight_decay=0.01)
            
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        
        # --- Early Stopping Setup ---
        patience = 10 
        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        start_epoch = 1
        
        # --- Checkpoint Loading Logic ---
        if self.save and os.path.exists(os.path.join(get_config_value("chkpt"), "labram_checkpoint.pth")):
            checkpoint = torch.load(os.path.join(get_config_value("chkpt"), "labram_checkpoint.pth"))
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_loss"]
            best_model_state = checkpoint["best_model"]
            print(f"Resuming training at epoch {start_epoch}")

        # Training loop
        for epoch in range(start_epoch, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs}")
            train_loss, train_acc = train_epoch(
                self.model,
                train_loader,
                optimizer,
                scheduler,
                self.device,
                train_input_chans,
                cached_features=using_cached_train,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}")

            if valid_loader is not None:
                val_loss, val_acc, val_metrics = validate_epoch(
                    self.model,
                    valid_loader,
                    self.device,
                    val_input_chans,
                    cached_features=using_cached_val,
                )
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print("  Val Metrics:", val_metrics)
        
                # --- Early Stopping Logic & Best Model Save ---
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict() # Save best state
                    patience_counter = 0 # Reset patience
                else:
                    patience_counter += 1 # Increment patience

                if self.wandb_run:
                    wandb_utils.log(
                        {
                            f"{self.name}/train_loss": train_loss,
                            f"{self.name}/train_acc": train_acc,
                            f"{self.name}/val_loss": val_loss,
                            f"{self.name}/val_acc": val_acc,
                            f"{self.name}/lr": current_lr,
                        },
                        step=epoch,
                    )
                    
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} (Patience: {patience})")
                    break # Exit the training loop
            
            # --- Checkpoint Saving Logic ---
            if self.save:
                torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_model": best_model_state,
                    "best_loss": best_val_loss,
                }, os.path.join(get_config_value("chkpt"), "labram_checkpoint.pth"))
        
        # Load the best model (if saved)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def _fit_with_cache(
        self,
        X: List[np.ndarray | List[BaseRaw]],
        y: List[np.ndarray | List[str]],
        meta: List[Dict],
        subset_fraction: float,
        subset_seed: Optional[int],
        subset_indices: List[List[int]],
    ) -> None:
        print("inside cached fit")
        task_name = meta[0]["task_name"]
        subset_seed = subset_seed if subset_seed is not None else 0

        subset_labels = self._gather_subset_labels(y, subset_indices)
        class_weights = torch.tensor(calc_class_weights(subset_labels, task_name)).to(self.device)
        self.model.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        cache_bundle = self._load_or_build_embedding_cache(X, y, meta, task_name)
        all_features: torch.Tensor = cache_bundle["features"]
        all_labels: torch.Tensor = cache_bundle["labels"]
        recording_ids: torch.Tensor = cache_bundle["recording_ids"]
        cache_channels = cache_bundle["metadata"]["channels"]
        train_input_chans = utils.get_input_chans(cache_channels)

        selected_global_ids = self._compute_global_record_indices(X, subset_indices)
        mask = self._build_recording_mask(recording_ids, selected_global_ids)
        if mask.sum().item() == 0:
            raise ValueError("Selected subset yielded 0 cached samples. Check subset_indices.")

        subset_features = all_features[mask].contiguous()
        subset_labels_tensor = all_labels[mask].contiguous()
        del all_features, all_labels, recording_ids
        val_split = 0.2
        num_samples = subset_features.shape[0]
        num_val = int(num_samples * val_split)
        generator = torch.Generator()
        generator.manual_seed(subset_seed)
        if num_val >= 1 and num_samples - num_val >= 1:
            perm = torch.randperm(num_samples, generator=generator)
            val_idx = perm[:num_val]
            train_idx = perm[num_val:]
            train_dataset = TensorDataset(subset_features[train_idx], subset_labels_tensor[train_idx])
            val_dataset = TensorDataset(subset_features[val_idx], subset_labels_tensor[val_idx])
        else:
            train_dataset = TensorDataset(subset_features, subset_labels_tensor)
            val_dataset = None

        cache_batch_size = min(1024, max(64 * self.cached_batch_multiplier, 64))
        train_loader = DataLoader(
            train_dataset,
            batch_size=cache_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        if val_dataset is not None:
            valid_loader = DataLoader(
                val_dataset,
                batch_size=cache_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        else:
            valid_loader = None

        max_epochs = 30
        steps_per_epoch = len(train_loader)
        max_lr = 4e-4

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2
        )

        patience = 10
        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None

        start_epoch = 1

        for epoch in range(start_epoch, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs}")
            train_loss, train_acc = train_epoch(
                self.model,
                train_loader,
                optimizer,
                scheduler,
                self.device,
                train_input_chans,
                cached_features=True,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}")

            if valid_loader is not None:
                val_loss, val_acc, val_metrics = validate_epoch(
                    self.model,
                    valid_loader,
                    self.device,
                    train_input_chans,
                    cached_features=True,
                )
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                print("  Val Metrics:", val_metrics)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.wandb_run:
                    wandb_utils.log(
                        {
                            f"{self.name}/train_loss": train_loss,
                            f"{self.name}/train_acc": train_acc,
                            f"{self.name}/val_loss": val_loss,
                            f"{self.name}/val_acc": val_acc,
                            f"{self.name}/lr": current_lr,
                        },
                        step=epoch,
                    )

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} (Patience: {patience})")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def _get_cache_root(self) -> Path:
        cache_dir = get_config_value("embedding_cache_dir")
        if cache_dir is None:
            data_root = get_config_value("data")
            if data_root is None:
                data_root = "."
            cache_dir = Path(data_root) / "embedding_cache"
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    def _cache_file_path(self, task_name: str, meta: List[Dict]) -> Path:
        dataset_names = [m.get("name", f"dataset_{idx}") for idx, m in enumerate(meta)]
        key_payload = {
            "task": task_name,
            "datasets": dataset_names,
            "chunk_len_s": self.chunk_len_s,
            "num_labels_per_chunk": self.num_labels_per_chunk,
            "model": self.name,
        }
        key_string = json.dumps(key_payload, sort_keys=True)
        cache_key = hashlib.sha1(key_string.encode("utf-8")).hexdigest()
        cache_root = self._get_cache_root()
        return cache_root / f"{task_name}_{cache_key}.pt"

    def _load_or_build_embedding_cache(
        self,
        X: List[np.ndarray | List[BaseRaw]],
        y: List[np.ndarray | List[str]],
        meta: List[Dict],
        task_name: str,
    ) -> Dict:
        cache_file = self._cache_file_path(task_name, meta)
        if cache_file.exists():
            print(f"[Cache] Loading cached embeddings from {cache_file}")
            return torch.load(cache_file, map_location="cpu")

        dataset = make_dataset_2(
            X,
            y,
            meta,
            task_name,
            self.name,
            self.chunk_len_s,
            is_train=True,
            use_cache=self.use_cache,
        )

        if len(dataset) == 0:
            raise ValueError("Dataset has 0 samples after preprocessing. Cannot cache embeddings.")

        batch_size = 64 if self.chunk_len_s is not None else 1
        # Encoding is a single long pass; tune workers + prefetch to reduce GPU idle time.
        cpu_count = os.cpu_count() or 2
        encode_workers = 2 #min(4, max(1, cpu_count // 2))
        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=encode_workers,
            shuffle=False,
            pin_memory=True,
        )
        if encode_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        dataloader = DataLoader(dataset, **loader_kwargs)
        input_chans = utils.get_input_chans(dataset.ch_names)
        cache_bundle = self._encode_dataset(dataset, dataloader, input_chans, task_name)
        del dataset
        torch.save(cache_bundle, cache_file)
        print(f"[Cache] Saved embeddings to {cache_file}")
        return cache_bundle

    def _encode_dataset(
        self,
        dataset: LaBraMDataset2,
        dataloader: DataLoader,
        input_chans,
        task_name: str,
    ) -> Dict:
        self.model.feature.eval()
        ordered_record_ids = [self._parse_recording_index(name) for name in dataset.recording_names]
        total_samples = len(ordered_record_ids)
        features_buf: torch.Tensor | None = None
        labels_buf: torch.Tensor | None = None
        record_ids_tensor = torch.empty(total_samples, dtype=torch.long)
        write_ptr = 0

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                for batch in tqdm(dataloader, desc=f"Encoding ({task_name})", leave=True):
                    x, y_tensor, channels = batch
                    batch_input_chans = input_chans
                    if channels != -1 and channels[0] != -1:
                        channels = [ch_arr[0] for ch_arr in channels]
                        batch_input_chans = utils.get_input_chans(channels)

                    feats = self.model.extract_features(x, batch_input_chans)
                    feats_cpu = feats.cpu()
                    labels_cpu = y_tensor.cpu() if isinstance(y_tensor, torch.Tensor) else torch.as_tensor(y_tensor)

                    batch_size = feats_cpu.shape[0]
                    start = write_ptr
                    end = start + batch_size

                    if features_buf is None:
                        features_buf = torch.empty((total_samples, *feats_cpu.shape[1:]), dtype=feats_cpu.dtype)
                        labels_buf = torch.empty((total_samples, *labels_cpu.shape[1:]), dtype=labels_cpu.dtype)

                    features_buf[start:end] = feats_cpu
                    labels_buf[start:end] = labels_cpu
                    record_ids_tensor[start:end] = torch.as_tensor(ordered_record_ids[start:end], dtype=torch.long)
                    write_ptr = end

        if write_ptr != total_samples or features_buf is None or labels_buf is None:
            raise RuntimeError(
                f"Encoding mismatch: expected {total_samples} samples, wrote {write_ptr}"
            )

        features_tensor = features_buf
        labels_tensor = labels_buf

        cache_bundle = {
            "features": features_tensor,
            "labels": labels_tensor,
            "recording_ids": record_ids_tensor,
            "metadata": {
                "channels": dataset.ch_names,
                "task_name": task_name,
                "chunk_len_s": self.chunk_len_s,
            },
        }
        return cache_bundle

    def _parse_recording_index(self, name: str) -> int:
        for token in name.split("_")[1:]:
            if token.isdigit():
                return int(token)
        raise ValueError(f"Unable to parse recording index from name '{name}'")

    def _compute_global_record_indices(
        self,
        X: List[np.ndarray | List[BaseRaw]],
        subset_indices: List[List[int]],
    ) -> List[int]:
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
        mask_np = np.isin(record_np, selected_np)
        return torch.from_numpy(mask_np)

    def _gather_subset_labels(
        self,
        labels: List[np.ndarray | List[str]],
        subset_indices: List[List[int]],
    ) -> List[List]:
        subset = []
        for ds_labels, idxs in zip(labels, subset_indices):
            if isinstance(ds_labels, np.ndarray):
                subset.append(ds_labels[idxs])
            else:
                subset.append([ds_labels[i] for i in idxs])
        return subset

    @torch.no_grad()
    def predict(self, X: List[np.ndarray|List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
        print("inside predict")
        task_name = meta[0]["task_name"]
        dataset_test  = make_dataset_2(X, None, meta, task_name, self.name, self.chunk_len_s, is_train=False, use_cache=self.use_cache)
        ch_names = dataset_test.ch_names
        
        if len(dataset_test) == 0:
            return np.array([])
        
        # Inference on test set

        if self.chunk_len_s is None:
            batch_size = 1
        else: 
            batch_size = 64
        test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)

        input_chans = utils.get_input_chans(ch_names)
        predictions, indices_mapping = inference(self.model, test_loader, self.device, input_chans)
        
        predictions = predictions.numpy()
        indices_mapping = indices_mapping.numpy()
        print(predictions.shape)
        print(indices_mapping.shape)

        if self.chunk_len_s is not None and not self.model.is_multilabel_task:
            # Aggregate predictions by majority voting for each unique index
            unique_indices = np.unique(indices_mapping)
            aggregated_predictions = []

            for idx in unique_indices:
                # Get all predictions corresponding to the current index
                idx_predictions = predictions[indices_mapping == idx]
                # Perform majority voting
                most_common_prediction = Counter(idx_predictions).most_common(1)[0][0]
                aggregated_predictions.append(most_common_prediction)

            # Convert to numpy array
            predictions = np.array(aggregated_predictions)

        mapped_pred = np.array([map_label_reverse(pred, task_name) for pred in predictions])
        
        print(mapped_pred)
        return mapped_pred
