from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import math
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

from .LaBraM.make_dataset_2 import make_dataset as make_dataset_2
from .LaBraM.utils_2 import calc_class_weights, map_label_reverse
from ..abstract_model import AbstractModel
from ...config import get_config_value
from ...utils import wandb_utils

from braindecode.models import SignalJEPA_PreLocal
import mne


SJEPA_WEIGHTS_URL = (
    "https://huggingface.co/braindecode/SignalJEPA/resolve/main/"
    "signal-jepa_16s-60_adeuwv4s.pth"
)


def download_sjepa_weights() -> Path:
    cache_dir = Path(get_config_value("chkpt"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "signal-jepa.pth"
    if not dest.exists():
        import urllib.request
        print(f"[S-JEPA] Downloading pretrained weights to {dest} ...")
        urllib.request.urlretrieve(SJEPA_WEIGHTS_URL, dest)
        print("[S-JEPA] Download complete.")
    return dest


def make_chs_info(ch_names: List[str], sfreq: float) -> list:
    info = mne.create_info(ch_names, sfreq, ch_types='eeg')
    info.set_montage('standard_1020', match_case=False, on_missing='ignore')
    return info['chs']


class SJEPAClinicalTorchModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_labels_per_chunk: Optional[int],
        chs_info: list,
        chunk_len_s: float = 4.0,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.is_multilabel = num_labels_per_chunk is not None
        self.num_classes = num_classes
        n_outputs = num_classes * (num_labels_per_chunk if self.is_multilabel else 1)

        self.inner = SignalJEPA_PreLocal(
            sfreq=250,
            input_window_seconds=chunk_len_s,
            chs_info=chs_info,
            n_outputs=n_outputs,
        )
        # Load pretrained weights, filtering out transformer.* keys
        weights_path = download_sjepa_weights()
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('transformer.')}
        self.inner.load_state_dict(state_dict, strict=False)
        print("[S-JEPA Clinical] Loaded pretrained weights.")

        if freeze_encoder:
            for name, param in self.inner.named_parameters():
                if 'feature_encoder' in name:
                    param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        logits = self.inner(x)
        if self.is_multilabel:
            logits = logits.reshape(x.shape[0], self.num_classes, -1)
        return x, logits


class SJEPAClinicalModel(AbstractModel):
    def __init__(
        self,
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
        freeze_encoder: bool = True,
    ):
        super().__init__("SJEPAClinicalModel")
        assert torch.cuda.is_available(), "S-JEPA Clinical requires CUDA"
        self.device = torch.device("cuda")
        self.num_classes = num_classes
        self.num_labels_per_chunk = num_labels_per_chunk
        self.freeze_encoder = freeze_encoder
        self.chunk_len_s = 4.0 if num_labels_per_chunk is None else 16.0
        self.use_cache = True

    def fit(self, X, y, meta) -> None:
        task_name = meta[0]["task_name"]

        dataset_train = make_dataset_2(
            X, y, meta, task_name, self.name,
            self.chunk_len_s, is_train=True, use_cache=self.use_cache,
        )
        if len(dataset_train) == 0:
            print("[Warning] Dataset empty. Retrying without cache...")
            dataset_train = make_dataset_2(
                X, y, meta, task_name, self.name,
                self.chunk_len_s, is_train=True, use_cache=False,
            )

        dataset_train, dataset_val = dataset_train.split_train_val(0.15)

        chs_info = make_chs_info(dataset_train.ch_names, 250)
        self.model = SJEPAClinicalTorchModel(
            self.num_classes, self.num_labels_per_chunk,
            chs_info, self.chunk_len_s, self.freeze_encoder
        ).to(self.device)

        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        del X, y, meta
        gc.collect()
        torch.cuda.empty_cache()

        bs = 64 if self.chunk_len_s else 1
        train_loader = DataLoader(dataset_train, batch_size=bs, num_workers=8,
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=bs, num_workers=8,
                                shuffle=False, pin_memory=True)

        max_epochs = 30
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=4e-4, steps_per_epoch=len(train_loader),
            epochs=max_epochs, pct_start=0.1
        )

        patience = 10
        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            acc_samples = 0
            for x, yb, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False):
                x, yb = x.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                _, logits = self.model(x)
                loss = self.model.loss_fn(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)
                if logits.dim() == 2:
                    preds = logits.argmax(dim=1)
                    target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                    correct += (preds == target).sum().item()
                    acc_samples += x.size(0)
                del x, yb, logits
                torch.cuda.empty_cache()
            train_loss = total_loss / total_samples
            train_acc = correct / acc_samples if acc_samples else 0.0

            self.model.eval()
            val_loss = 0.0
            val_samples = 0
            val_correct = 0
            val_acc_samples = 0
            with torch.no_grad():
                for x, yb, _ in val_loader:
                    x, yb = x.to(self.device), yb.to(self.device)
                    _, logits = self.model(x)
                    loss = self.model.loss_fn(logits, yb)
                    val_loss += loss.item() * x.size(0)
                    val_samples += x.size(0)
                    if logits.dim() == 2:
                        preds = logits.argmax(dim=1)
                        target = yb if yb.dim() == 1 else yb.argmax(dim=1)
                        val_correct += (preds == target).sum().item()
                        val_acc_samples += x.size(0)
                    del x, yb, logits
                    torch.cuda.empty_cache()
            avg_val_loss = val_loss / val_samples if val_samples else 0.0
            val_acc = val_correct / val_acc_samples if val_acc_samples else 0.0

            current_lr = scheduler.get_last_lr()[0]
            metrics = {
                f"{self.name}/train_loss": train_loss,
                f"{self.name}/train_acc": train_acc,
                f"{self.name}/val_loss": avg_val_loss,
                f"{self.name}/val_acc": val_acc,
                f"{self.name}/lr": current_lr,
            }
            if self.wandb_run:
                wandb_utils.log(metrics, step=epoch)

            print(
                f"[Epoch {epoch:02d}/{max_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} | "
                f"lr={current_lr:.2e} patience={patience_counter}/{patience}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    @torch.no_grad()
    def predict(self, X, meta) -> np.ndarray:
        task_name = meta[0]["task_name"]
        dataset_test = make_dataset_2(
            X, None, meta, task_name, self.name,
            self.chunk_len_s, is_train=False, use_cache=self.use_cache,
        )
        if len(dataset_test) == 0:
            return np.array([])

        loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
        self.model.eval()
        preds_all = []
        idx_map_all = []
        for x, idx, _ in tqdm(loader, desc="Predicting"):
            x = x.to(self.device)
            _, logits = self.model(x)
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            idx_map_all.append(idx.cpu().numpy())

        preds = np.concatenate(preds_all)
        idx_map = np.concatenate(idx_map_all)
        unique_indices = np.unique(idx_map)
        final_predictions = []
        for i in unique_indices:
            patient_votes = preds[idx_map == i]
            final_predictions.append(Counter(patient_votes).most_common(1)[0][0])
        return np.array([map_label_reverse(p, task_name) for p in final_predictions])
