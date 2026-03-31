import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
import logging
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Iterable
from collections import Counter
from sklearn.model_selection import train_test_split
from ..config import get_config_value
from .eeg_noise import format_noise_tag

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_multilabel_tasks():
    return set(["seizure_clinical", "sleep_stages_clinical", "binary_artifact_clinical", "multiclass_artifact_clinical"])


def configure_torch_backend_for_speed():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def create_temp_cache_dir(prefix: str = "lp_cache_") -> str:
    cache_root = get_config_value("cache") or "/tmp"
    return tempfile.mkdtemp(prefix=prefix, dir=cache_root)


def cleanup_temp_cache_dir(path: str) -> None:
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


class CachedArrayDataset(Dataset):
    """Dataset wrapper that safely materializes tensors from memmaps/ndarrays."""

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(np.asarray(self.features[idx]))
        label = torch.tensor(np.asarray(self.labels[idx]))
        return feature, label


def _is_multilabel_data(y_i) -> bool:
    """
    Detect if y_i contains multi-label annotations (list of event tuples per recording).

    Multi-label format: y_i = [[event_type, start, stop], [event_type, start, stop], ...]
    Recording-level multi-label format: y_i = [[[event_type, start, stop], ...], ...]
    Single-label format: y_i = label (scalar) or y_i = [label1, label2, ...] (1D array)
    """
    if isinstance(y_i, np.ndarray):
        return False
    if not isinstance(y_i, list) or len(y_i) == 0:
        return False
    def _looks_like_event(value) -> bool:
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 3
            and isinstance(value[1], (int, float, np.integer, np.floating))
        )
    # Check if first element is a list/tuple with 3 elements (event annotation)
    for entry in y_i:
        if _looks_like_event(entry):
            return True
        if isinstance(entry, list) and entry:
            if _looks_like_event(entry[0]):
                return True
    return False


def subsample_data_stratified(
    X: List[np.ndarray],
    y: List[np.ndarray],
    percentage: float,
    random_state: int = 42,
    return_indices: bool = False,
) -> Union[
    Tuple[List[np.ndarray], List[np.ndarray], Dict],
    Tuple[List[np.ndarray], List[np.ndarray], Dict, List[List[int]]],
]:
    """
    Subsample training data while maintaining class proportions.

    Args:
        X: List of numpy arrays, one per dataset
        y: List of label arrays, one per dataset
        percentage: Fraction of data to keep (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        X_sub: Subsampled X
        y_sub: Subsampled y
        stats: Dict with samples_per_class and total_samples
        indices_per_dataset: Optional list of indices (per dataset) referencing the
            original X entries that were kept. Returned when ``return_indices`` is True.
    """
    MIN_SAMPLES_PER_CLASS = 2  # Hardcoded minimum to ensure class representation

    def _collect_label_values_multilabel(y_list):
        """Collect all label values from multi-label annotations."""
        def _looks_like_event(value) -> bool:
            return (
                isinstance(value, (list, tuple))
                and len(value) >= 3
                and isinstance(value[1], (int, float, np.integer, np.floating))
            )
        values = []
        for y_i in y_list:
            for entry in y_i:
                if _looks_like_event(entry):
                    values.append(entry[0])  # event = [type, start, stop]
                elif isinstance(entry, list):
                    for event in entry:
                        if _looks_like_event(event):
                            values.append(event[0])
        return values

    def _collect_label_values_singlelabel(labels):
        """Collect label values from single-label data."""
        if isinstance(labels, np.ndarray):
            return labels.tolist()
        return list(labels)

    def _take_indices(items, indices):
        return items[indices] if isinstance(items, np.ndarray) else [items[i] for i in indices]

    def _get_label_array(y_i):
        """Convert labels to numpy array for class counting (single-label only)."""
        if isinstance(y_i, np.ndarray):
            return y_i
        return np.array(y_i)

    # Detect if this is multi-label data
    is_multilabel = any(_is_multilabel_data(y_i) for y_i in y)

    selected_indices_all: List[List[int]] = [[] for _ in X]

    def _finalize(X_ret, y_ret, stats_ret):
        if return_indices:
            return X_ret, y_ret, stats_ret, selected_indices_all
        return X_ret, y_ret, stats_ret

    if percentage >= 1.0:
        if is_multilabel:
            all_labels = _collect_label_values_multilabel(y)
        else:
            all_labels = []
            for y_i in y:
                all_labels.extend(_collect_label_values_singlelabel(y_i))
        # Convert labels to hashable form for Counter
        all_labels_hashable = [str(l) if isinstance(l, list) else l for l in all_labels]
        stats = {
            "samples_per_class": dict(Counter(all_labels_hashable)),
            "total_samples": len(all_labels)
        }
        for ds_idx, dataset in enumerate(X):
            selected_indices_all[ds_idx] = list(range(len(dataset)))
        return _finalize(X, y, stats)

    X_sub, y_sub = [], []
    rng = np.random.RandomState(random_state)

    if is_multilabel:
        # For multi-label tasks, subsample at the recording level (no stratification)
        logger.info("Multi-label task detected: subsampling at recording level")
        for ds_idx, (X_i, y_i) in enumerate(zip(X, y)):
            n_recordings = len(X_i)
            n_keep = max(1, int(n_recordings * percentage))

            if n_keep >= n_recordings:
                X_sub.append(X_i)
                y_sub.append(y_i)
                selected_indices_all[ds_idx] = list(range(n_recordings))
            else:
                indices = rng.choice(n_recordings, size=n_keep, replace=False)
                indices = sorted(indices)  # Keep order for reproducibility
                X_sub.append(_take_indices(X_i, indices))
                y_sub.append(_take_indices(y_i, indices))
                selected_indices_all[ds_idx] = list(indices)

        all_labels = _collect_label_values_multilabel(y_sub)
        all_labels_hashable = [str(l) if isinstance(l, list) else l for l in all_labels]
        stats = {
            "samples_per_class": dict(Counter(all_labels_hashable)),
            "total_samples": sum(len(y_i) for y_i in y_sub),
            "total_recordings": sum(len(X_i) for X_i in X_sub)
        }
        return _finalize(X_sub, y_sub, stats)

    # Single-label task: use stratified sampling
    all_labels_sub = []

    for ds_idx, (X_i, y_i) in enumerate(zip(X, y)):
        n_samples = len(y_i)
        label_arr = _get_label_array(y_i)
        unique_classes, class_counts = np.unique(label_arr, return_counts=True)
        n_classes = len(unique_classes)

        # Calculate minimum samples needed to maintain class representation
        min_samples_needed = n_classes * MIN_SAMPLES_PER_CLASS
        n_keep = max(min_samples_needed, int(n_samples * percentage))

        if n_keep >= n_samples:
            # Use full dataset
            X_sub.append(X_i)
            y_sub.append(y_i)
            all_labels_sub.extend(_collect_label_values_singlelabel(y_i))
            selected_indices_all[ds_idx] = list(range(n_samples))
            continue

        # Check if stratification is feasible
        min_class_count = class_counts.min()
        samples_per_class_target = int(min_class_count * percentage)

        if samples_per_class_target < MIN_SAMPLES_PER_CLASS:
            # Percentage too low for this dataset - use minimum samples per class
            logger.warning(
                f"Percentage {percentage:.1%} too low for dataset with {n_samples} samples "
                f"and {n_classes} classes. Using {MIN_SAMPLES_PER_CLASS} samples per class minimum."
            )
            # Sample exactly MIN_SAMPLES_PER_CLASS from each class
            indices = []
            for cls in unique_classes:
                cls_indices = np.where(label_arr == cls)[0]
                n_to_sample = min(MIN_SAMPLES_PER_CLASS, len(cls_indices))
                indices.extend(rng.choice(cls_indices, size=n_to_sample, replace=False))
            indices = np.array(indices)
            X_sub.append(_take_indices(X_i, indices))
            y_sub.append(_take_indices(y_i, indices))
            all_labels_sub.extend(_collect_label_values_singlelabel(_take_indices(y_i, indices)))
            selected_indices_all[ds_idx] = list(indices.tolist())
        else:
            # Try stratified split
            try:
                train_indices, _ = train_test_split(
                    np.arange(n_samples),
                    label_arr,
                    train_size=percentage,
                    stratify=label_arr,
                    random_state=random_state
                )
                train_indices = np.array(train_indices)
                X_sub.append(_take_indices(X_i, train_indices))
                y_sub.append(_take_indices(y_i, train_indices))
                all_labels_sub.extend(_collect_label_values_singlelabel(_take_indices(y_i, train_indices)))
                selected_indices_all[ds_idx] = list(train_indices.tolist())
            except ValueError as e:
                # Fallback: sample proportionally from each class
                logger.warning(f"Stratified split failed: {e}. Using per-class sampling.")
                indices = []
                for cls, count in zip(unique_classes, class_counts):
                    cls_indices = np.where(label_arr == cls)[0]
                    n_to_sample = max(MIN_SAMPLES_PER_CLASS, int(count * percentage))
                    n_to_sample = min(n_to_sample, len(cls_indices))
                    indices.extend(rng.choice(cls_indices, size=n_to_sample, replace=False))
                indices = np.array(indices)
                X_sub.append(_take_indices(X_i, indices))
                y_sub.append(_take_indices(y_i, indices))
                all_labels_sub.extend(_collect_label_values_singlelabel(_take_indices(y_i, indices)))
                selected_indices_all[ds_idx] = list(indices.tolist())

    stats = {
        "samples_per_class": dict(Counter(all_labels_sub)),
        "total_samples": len(all_labels_sub)
    }

    # Warn if any class has very few samples
    for cls, count in stats["samples_per_class"].items():
        if count < MIN_SAMPLES_PER_CLASS:
            logger.warning(f"Class {cls} has only {count} samples after subsampling!")

    return _finalize(X_sub, y_sub, stats)

def save_results(
    y_trains,
    y_trues,
    models_names,
    results,
    dataset_names,
    task_name,
    seeds: Optional[List[int]] = None,
    data_percentage: float = 1.0,
    data_stats: Optional[Dict] = None,
    linear_probe: bool = False,
    result_prefix: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
    eval_noise_types: Optional[Iterable[str]] = None,
    eval_noise_snr_db: Optional[float] = None,
    eval_noise_channel_dropout_prob: Optional[float] = None,
):

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_names_unique = list(set(models_names))
    models_str = "_".join(models_names_unique) if models_names_unique else "models"

    # Build the filename with optional prefix, task name, models, checkpoint ID, percentage, LP indicator, and timestamp
    prefix_str = f"{result_prefix}_" if result_prefix else ""
    ckpt_str = f"_ckpt_{checkpoint_id}" if checkpoint_id else ""
    pct_str = f"_pct{int(data_percentage * 100)}" if data_percentage < 1.0 else ""
    seed_str = f"_seed{int(seeds[0])}" if seeds is not None and len(seeds) == 1 else ""
    lp_str = "_LP" if linear_probe else ""
    noise_tag = format_noise_tag(eval_noise_types, eval_noise_snr_db)
    noise_str = f"_{noise_tag}" if noise_tag != "clean" else ""
    filename = os.path.join(
        get_config_value("results"),
        "raw",
        f"{prefix_str}{task_name}_{models_str}{ckpt_str}{pct_str}{seed_str}{lp_str}{noise_str}_{timestamp}.json",
    )

    if task_name in get_multilabel_tasks():
        y_trains = [[[y_2.tolist() for y_2 in y] for y in y_train] for y_train in y_trains]
        y_trues = [[[y_2.tolist() for y_2 in y] for y in y_true] for y_true in y_trues]
    else:
        if isinstance(y_trains[0][0], np.ndarray):
            y_trains = [[y.tolist() for y in y_train] for y_train in y_trains]
            y_trues = [[y.tolist() for y in y_true] for y_true in y_trues]    

    # y_trains = [[y.tolist() if isinstance(y, np.ndarray) else [y_2.tolist() for y_2 in y] for y in y_train] for y_train in y_trains]
    # y_trues = [[y.tolist() if isinstance(y, np.ndarray) else [y_2.tolist() for y_2 in y] for y in y_test] for y_test in y_trues]
    results = [result.tolist() if isinstance(result, np.ndarray) else [y_2.tolist() for y_2 in result] for result in results]

    # Prepare the data to be saved
    data_to_save = {
        "y_train": y_trains,
        "y_test": y_trues,
        "models_names": models_names,
        "results": results,
        "dataset_names": dataset_names,
        "task_name": task_name,
        "timestamp": timestamp,
        "seeds": seeds,
        "seed": int(seeds[0]) if seeds is not None and len(seeds) == 1 else None,
        "data_percentage": data_percentage,
        "data_stats": data_stats,
        "linear_probe": linear_probe,
        "result_prefix": result_prefix,
        "checkpoint_id": checkpoint_id,
        "eval_noise_types": list(eval_noise_types) if eval_noise_types is not None else None,
        "eval_noise_snr_db": eval_noise_snr_db,
        "eval_noise_channel_dropout_prob": eval_noise_channel_dropout_prob,
    }

    # Save the results to the file
    json_folder = os.path.join(get_config_value("results"), "raw")
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)
    with open(filename, "w") as f:
        json.dump(data_to_save, f)
    print(f"Results saved to {filename}")
