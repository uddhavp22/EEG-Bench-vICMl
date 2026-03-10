from __future__ import annotations

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import math
import random
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from joblib import Memory

from .LaBraM.make_dataset import make_dataset_sjepa
from .LaBraM.labram_datasets import LaBraMBCIDataset
from .LaBraM.utils_2 import calc_class_weights, reverse_map_label, n_unique_labels
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


class SJEPABCITorchModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        chs_info: list,
        input_window_seconds: float,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.inner = SignalJEPA_PreLocal(
            sfreq=250,
            input_window_seconds=input_window_seconds,
            chs_info=chs_info,
            n_outputs=num_classes,
        )
        # Load pretrained weights, filtering out transformer.* keys
        weights_path = download_sjepa_weights()
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('transformer.')}
        self.inner.load_state_dict(state_dict, strict=False)
        print("[S-JEPA BCI] Loaded pretrained weights.")

        if freeze_encoder:
            for name, param in self.inner.named_parameters():
                if 'feature_encoder' in name:
                    param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


class SJEPAModel(AbstractModel):
    def __init__(self, freeze_encoder: bool = True):
        super().__init__("SJEPAModel")
        assert torch.cuda.is_available(), "S-JEPA BCI requires CUDA"
        self.device = torch.device("cuda")
        self.freeze_encoder = freeze_encoder
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

    def _train_epoch(self, dataloader, optimizer, scheduler):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for x, y_batch in tqdm(dataloader, desc="Train", leave=False):
            x = x.to(self.device)
            y_int = y_batch.to(self.device).argmax(dim=1)
            optimizer.zero_grad()
            logits = self.model(x)
            loss = self.model.loss_fn(logits, y_int)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y_int).sum().item()
            total_samples += x.size(0)
        return total_loss / total_samples, total_correct / total_samples

    def _validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for x, y_batch in dataloader:
                x = x.to(self.device)
                y_int = y_batch.to(self.device).argmax(dim=1)
                logits = self.model(x)
                loss = self.model.loss_fn(logits, y_int)
                total_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(dim=1) == y_int).sum().item()
                total_samples += x.size(0)
        return total_loss / total_samples, total_correct / total_samples

    def fit(self, X, y, meta) -> None:
        task_name = meta[0]["task_name"]
        num_classes = n_unique_labels(task_name)

        datasets = [
            self.cache.cache(make_dataset_sjepa)(
                X_, y_, task_name, m_["sampling_frequency"], m_["channel_names"],
                train=True, split_size=0.15
            )
            for X_, y_, m_ in zip(X, y, meta)
        ]
        dataset_train_list = [ds[0] for ds in datasets if len(ds[0]) > 0]
        dataset_val_list = [ds[1] for ds in datasets if len(ds[1]) > 0]

        # Build chs_info once from the first training dataset's channel names
        chs_info = make_chs_info(dataset_train_list[0].ch_names, 250)
        sample_x = dataset_train_list[0][0][0]  # shape: (C, T)
        input_window_seconds = sample_x.shape[-1] / 250.0

        self.model = SJEPABCITorchModel(
            num_classes, chs_info, input_window_seconds, self.freeze_encoder
        ).to(self.device)

        class_weights = torch.tensor(calc_class_weights(y, task_name)).to(self.device)
        self.model.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        del X, y, meta
        gc.collect()
        torch.cuda.empty_cache()

        batch_size = 64
        num_workers = 8
        train_loader_list = [
            DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                       shuffle=True, pin_memory=True)
            for ds in dataset_train_list
        ]
        valid_loader_list = [
            DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                       shuffle=False, pin_memory=True)
            for ds in dataset_val_list
        ]

        max_epochs = 30
        steps_per_epoch = math.ceil(sum(len(l) for l in train_loader_list))
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-6, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=4e-4, steps_per_epoch=steps_per_epoch,
            epochs=max_epochs, pct_start=0.2
        )

        patience = 10
        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(1, max_epochs + 1):
            random.shuffle(train_loader_list)
            epoch_train_loss, epoch_train_acc, n_train = 0.0, 0.0, 0
            for loader in train_loader_list:
                tl, ta = self._train_epoch(loader, optimizer, scheduler)
                epoch_train_loss += tl
                epoch_train_acc += ta
                n_train += 1
            avg_train_loss = epoch_train_loss / n_train
            avg_train_acc = epoch_train_acc / n_train

            epoch_val_loss, epoch_val_acc, n_val = 0.0, 0.0, 0
            for loader in valid_loader_list:
                vl, va = self._validate_epoch(loader)
                epoch_val_loss += vl
                epoch_val_acc += va
                n_val += 1
            avg_val_loss = epoch_val_loss / max(n_val, 1)
            avg_val_acc = epoch_val_acc / max(n_val, 1)

            current_lr = scheduler.get_last_lr()[0]
            metrics = {
                f"{self.name}/train_loss": avg_train_loss,
                f"{self.name}/train_acc": avg_train_acc,
                f"{self.name}/val_loss": avg_val_loss,
                f"{self.name}/val_acc": avg_val_acc,
                f"{self.name}/lr": current_lr,
            }
            if self.wandb_run:
                wandb_utils.log(metrics, step=epoch)

            print(
                f"[Epoch {epoch:02d}/{max_epochs}] "
                f"train_loss={avg_train_loss:.4f} train_acc={avg_train_acc:.4f} | "
                f"val_loss={avg_val_loss:.4f} val_acc={avg_val_acc:.4f} | "
                f"lr={current_lr:.2e} patience={patience_counter}/{patience}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    @torch.no_grad()
    def predict(self, X, meta) -> np.ndarray:
        task_name = meta[0]["task_name"]
        self.model.eval()

        dataset_test_list = [
            self.cache.cache(make_dataset_sjepa)(
                X_, None, task_name, m_["sampling_frequency"], m_["channel_names"], train=False
            )
            for X_, m_ in zip(X, meta)
        ]
        dataset_test_list = [ds for ds in dataset_test_list if len(ds) > 0]

        predictions = []
        for ds in dataset_test_list:
            loader = DataLoader(ds, batch_size=64, num_workers=0, shuffle=False)
            preds_all = []
            for x in tqdm(loader, desc="BCI Predicting", leave=False):
                logits = self.model(x.to(self.device))
                preds_all.append(torch.argmax(logits, dim=1).cpu())
            predictions.append(torch.cat(preds_all, dim=0))

        predictions = torch.cat(predictions, dim=0).numpy()
        return np.array([reverse_map_label(idx, task_name) for idx in predictions])
