import random
import numpy as np
import torch
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.model_selection import train_test_split
from ..config import get_config_value

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


def subsample_data_stratified(
    X: List[np.ndarray],
    y: List[np.ndarray],
    percentage: float,
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
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
    """
    if percentage >= 1.0:
        all_labels = np.concatenate(y)
        stats = {
            "samples_per_class": dict(Counter(all_labels.tolist())),
            "total_samples": len(all_labels)
        }
        return X, y, stats

    X_sub, y_sub = [], []
    all_labels_sub = []

    for X_i, y_i in zip(X, y):
        n_samples = len(y_i)
        n_keep = max(2, int(n_samples * percentage))  # Need at least 2 for stratified split

        if n_keep >= n_samples:
            X_sub.append(X_i)
            y_sub.append(y_i)
            all_labels_sub.extend(y_i.tolist())
        else:
            try:
                X_keep, _, y_keep, _ = train_test_split(
                    X_i, y_i,
                    train_size=percentage,
                    stratify=y_i,
                    random_state=random_state
                )
                X_sub.append(X_keep)
                y_sub.append(y_keep)
                all_labels_sub.extend(y_keep.tolist())
            except ValueError as e:
                # Stratification failed (e.g., too few samples per class)
                logger.warning(f"Stratified split failed, using random sample: {e}")
                indices = np.random.RandomState(random_state).choice(
                    n_samples, size=n_keep, replace=False
                )
                X_sub.append(X_i[indices])
                y_sub.append(y_i[indices])
                all_labels_sub.extend(y_i[indices].tolist())

    stats = {
        "samples_per_class": dict(Counter(all_labels_sub)),
        "total_samples": len(all_labels_sub)
    }
    return X_sub, y_sub, stats

def save_results(
    y_trains,
    y_trues,
    models_names,
    results,
    dataset_names,
    task_name,
    data_percentage: float = 1.0,
    data_stats: Optional[Dict] = None,
):

    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_names_unique = list(set(models_names))
    models_str = "_".join(models_names_unique) if models_names_unique else "models"

    # Build the filename with task name, models, percentage, and timestamp
    pct_str = f"_pct{int(data_percentage * 100)}" if data_percentage < 1.0 else ""
    filename = os.path.join(get_config_value("results"), "raw", f"{task_name}_{models_str}{pct_str}_{timestamp}.json")

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
        "data_percentage": data_percentage,
        "data_stats": data_stats,
    }

    # Save the results to the file
    json_folder = os.path.join(get_config_value("results"), "raw")
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)
    with open(filename, "w") as f:
        json.dump(data_to_save, f)
    print(f"Results saved to {filename}")
