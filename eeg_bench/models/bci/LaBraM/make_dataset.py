from .labram_datasets import LaBraMBCIDataset
from .utils_2 import map_label, n_unique_labels
import numpy as np
from typing import List, Tuple, Optional, cast
from resampy import resample
from mne.filter import filter_data, notch_filter
from mne.io import BaseRaw
from tqdm import tqdm
import gc
import os
import pickle
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import logging
from ...lejepa_preprocessing import apply_defossez_scaling, filter_resample_array


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9H', 'TTP7H', 'TPP9H', 'FTT10H', 'TPP8H', 'TPP10H', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def _ordered_target_channels(ch_names: List[str]) -> List[str]:
    """Return channels in deterministic standard_1020 order."""
    available = set(ch_names)
    return [ch.upper() for ch in standard_1020 if ch.upper() in available]

def make_dataset(data: np.ndarray, labels: np.ndarray|None, task_name: str, sampling_rate: int, 
                 ch_names: List[str], target_rate: int = 200, target_channels: Optional[List[str]] = None,
                 l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, split_size=0.1) -> LaBraMBCIDataset:
    """
    data: np.ndarray, shape=(n_trials, n_channels, n_samples)
    labels: np.ndarray, shape=(n_trials,)
    ch_names: List[str], list of channel names
    target_channels: List[str], list of target channel names
    sampling_rate: int, sampling rate of the data
    target_rate: int, target sampling rate
    l_freq: int, low cut-off frequency
    h_freq: int, high cut-off frequency
    """
    print("\ndata shape: ", data.shape)
    logging.info(f"data shape: {data.shape}")
    if len(data) == 0:
        if train:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names), LaBraMBCIDataset(data, labels, sampling_rate, ch_names)
        else:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names)
    # filter out the channels that are not in the target_channels
    if target_channels is not None:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = [ch.upper() for ch in target_channels]
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]
    else:
        # target_channels = ch_names
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = _ordered_target_channels(ch_names)
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]

    # bandpass filter
    data = filter_data(data, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    # notch filter
    data = notch_filter(data, Fs=sampling_rate, freqs=50, verbose=False)
    # resample data
    data = resample(data, sampling_rate, target_rate, axis=2, filter='kaiser_best')
    
    logging.info(f"data shape after resampling: {data.shape}")
    # Extend data to have a whole number of seconds by padding with zeros or trimming
    n_samples = data.shape[2]
    n_seconds = np.floor(n_samples / target_rate).astype(int)
    new_n_samples = n_seconds * target_rate
    if new_n_samples > n_samples:
        padding = new_n_samples - n_samples
        data = np.pad(data, ((0, 0), (0, 0), (0, padding)), mode='constant', constant_values=0)
    elif new_n_samples < n_samples:
        data = data[:, :, :new_n_samples]

    # One hot encode labels if they are not None
    if labels is not None:
        labels = np.array([map_label(label, task_name) for label in labels])
        labels = np.eye(n_unique_labels(task_name))[labels]
        print("labels shape: ", labels.shape)  
    if train:
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=split_size, random_state=42)
        return LaBraMBCIDataset(data_train, labels_train, target_rate, target_channels), LaBraMBCIDataset(data_val, labels_val, target_rate, target_channels)
    else:
        return LaBraMBCIDataset(data, labels, target_rate, target_channels)


def make_dataset_lejepa(data: np.ndarray, labels: np.ndarray|None, task_name: str, sampling_rate: int,
                        ch_names: List[str], target_rate: int = 250, target_channels: Optional[List[str]] = None,
                        l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, split_size=0.1) -> LaBraMBCIDataset:
    """
    LeJEPA-specific dataset creation with Defossez scaling and 250 Hz sample rate.

    Args:
        data: np.ndarray, shape=(n_trials, n_channels, n_samples)
        labels: np.ndarray, shape=(n_trials,)
        task_name: str, name of the task
        sampling_rate: int, sampling rate of the data
        ch_names: List[str], list of channel names
        target_rate: int, target sampling rate (default 250 Hz for LeJEPA)
        target_channels: List[str], optional list of target channel names
        l_freq: float, low cut-off frequency
        h_freq: float, high cut-off frequency
        train: bool, whether to split into train/val
        split_size: float, validation split size
    """
    print("\ndata shape: ", data.shape)
    logging.info(f"data shape: {data.shape}")
    if len(data) == 0:
        if train:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names), LaBraMBCIDataset(data, labels, sampling_rate, ch_names)
        else:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names)

    # Filter channels
    if target_channels is not None:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = [ch.upper() for ch in target_channels]
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]
    else:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = _ordered_target_channels(ch_names)
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]

    data = filter_resample_array(data, sampling_rate, target_rate)

    logging.info(f"data shape after resampling: {data.shape}")

    # Pad/trim to whole seconds
    n_samples = data.shape[2]
    n_seconds = np.floor(n_samples / target_rate).astype(int)
    new_n_samples = n_seconds * target_rate
    if new_n_samples > n_samples:
        padding = new_n_samples - n_samples
        data = np.pad(data, ((0, 0), (0, 0), (0, padding)), mode='constant', constant_values=0)
    elif new_n_samples < n_samples:
        data = data[:, :, :new_n_samples]

    # Apply Defossez scaling (auto-detects if data is in V or µV)
    data = apply_defossez_scaling(data, median_axis=1, scale_axis=None)

    # One hot encode labels
    if labels is not None:
        labels = np.array([map_label(label, task_name) for label in labels])
        labels = np.eye(n_unique_labels(task_name))[labels]
        print("labels shape: ", labels.shape)

    if train:
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=split_size, random_state=42)
        return LaBraMBCIDataset(data_train, labels_train, target_rate, target_channels), LaBraMBCIDataset(data_val, labels_val, target_rate, target_channels)
    else:
        return LaBraMBCIDataset(data, labels, target_rate, target_channels)


def make_dataset_luna(data: np.ndarray, labels: np.ndarray|None, task_name: str, sampling_rate: int,
                      ch_names: List[str], target_rate: int = 256, target_channels: Optional[List[str]] = None,
                      l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, split_size=0.1,
                      patch_size: int = 40):
    """
    LUNA preprocessing:
    - Bandpass: 0.1-75 Hz
    - Notch: 50 Hz
    - Resample: 256 Hz
    - Patch-size alignment for tokenization
    """
    print(f"\n[LUNA] Processing data with shape: {data.shape}")
    logging.info(f"[LUNA] data shape: {data.shape}, sampling_rate: {sampling_rate} Hz")

    if len(data) == 0:
        if train:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names), LaBraMBCIDataset(data, labels, sampling_rate, ch_names)
        return LaBraMBCIDataset(data, labels, sampling_rate, ch_names)

    if target_channels is not None:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = [ch.upper() for ch in target_channels]
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]
    else:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = _ordered_target_channels(ch_names)
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]

    data = filter_data(data, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    data = notch_filter(data, Fs=sampling_rate, freqs=50, verbose=False)
    data = resample(data, sampling_rate, target_rate, axis=2, filter='kaiser_best')
    logging.info(f"[LUNA] data shape after resampling: {data.shape}")

    n_samples = data.shape[2]
    n_seconds = np.floor(n_samples / target_rate).astype(int)
    new_n_samples = n_seconds * target_rate
    if new_n_samples > n_samples:
        padding = new_n_samples - n_samples
        data = np.pad(data, ((0, 0), (0, 0), (0, padding)), mode='constant', constant_values=0)
    elif new_n_samples < n_samples:
        data = data[:, :, :new_n_samples]

    remainder = data.shape[2] % patch_size
    if remainder != 0:
        pad = patch_size - remainder
        data = np.pad(data, ((0, 0), (0, 0), (0, pad)), mode='constant', constant_values=0)

    if labels is not None:
        labels = np.array([map_label(label, task_name) for label in labels])
        labels = np.eye(n_unique_labels(task_name))[labels]

    if train:
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=split_size, random_state=42)
        return LaBraMBCIDataset(data_train, labels_train, target_rate, target_channels), LaBraMBCIDataset(data_val, labels_val, target_rate, target_channels)
    return LaBraMBCIDataset(data, labels, target_rate, target_channels)


def make_dataset_sjepa(data: np.ndarray, labels: np.ndarray|None, task_name: str, sampling_rate: int,
                       ch_names: List[str], target_rate: int = 250, target_channels: Optional[List[str]] = None,
                       l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, split_size=0.1) -> LaBraMBCIDataset:
    """
    Signal-JEPA preprocessing:
    - Bandpass: 0.1-75 Hz
    - Notch: 50 Hz
    - Resample: 250 Hz
    - Defossez scaling
    """
    print(f"\n[S-JEPA] Processing data with shape: {data.shape}")
    logging.info(f"[S-JEPA] data shape: {data.shape}, sampling_rate: {sampling_rate} Hz")

    if len(data) == 0:
        if train:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names), LaBraMBCIDataset(data, labels, sampling_rate, ch_names)
        return LaBraMBCIDataset(data, labels, sampling_rate, ch_names)

    if target_channels is not None:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = [ch.upper() for ch in target_channels]
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]
    else:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = _ordered_target_channels(ch_names)
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]

    data = filter_data(data, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    data = notch_filter(data, Fs=sampling_rate, freqs=50, verbose=False)
    data = resample(data, sampling_rate, target_rate, axis=2, filter='kaiser_best')
    logging.info(f"[S-JEPA] data shape after resampling: {data.shape}")

    n_samples = data.shape[2]
    n_seconds = np.floor(n_samples / target_rate).astype(int)
    new_n_samples = n_seconds * target_rate
    if new_n_samples > n_samples:
        padding = new_n_samples - n_samples
        data = np.pad(data, ((0, 0), (0, 0), (0, padding)), mode='constant', constant_values=0)
    elif new_n_samples < n_samples:
        data = data[:, :, :new_n_samples]

    data = apply_defossez_scaling(data, median_axis=1, scale_axis=None)

    if labels is not None:
        labels = np.array([map_label(label, task_name) for label in labels])
        labels = np.eye(n_unique_labels(task_name))[labels]
        print("labels shape: ", labels.shape)

    if train:
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=split_size, random_state=42)
        return LaBraMBCIDataset(data_train, labels_train, target_rate, target_channels), LaBraMBCIDataset(data_val, labels_val, target_rate, target_channels)
    return LaBraMBCIDataset(data, labels, target_rate, target_channels)


def make_dataset_cbramod(data: np.ndarray, labels: np.ndarray|None, task_name: str, sampling_rate: int,
                         ch_names: List[str], target_rate: int = 200, target_channels: Optional[List[str]] = None,
                         l_freq: float = 0.5, h_freq: float = 40.0, train: bool = True, split_size=0.1):
    """
    CBraMod preprocessing:
    - Bandpass: 0.5-40 Hz
    - Notch: 50 Hz
    - Resample: 200 Hz
    - Per-recording z-score normalization
    """
    print(f"\n[CBraMod] Processing data with shape: {data.shape}")
    logging.info(f"[CBraMod] data shape: {data.shape}, sampling_rate: {sampling_rate} Hz")

    if len(data) == 0:
        if train:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names), LaBraMBCIDataset(data, labels, sampling_rate, ch_names)
        return LaBraMBCIDataset(data, labels, sampling_rate, ch_names)

    if target_channels is not None:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = [ch.upper() for ch in target_channels]
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]
    else:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = _ordered_target_channels(ch_names)
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]

    data = filter_data(data, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    data = notch_filter(data, Fs=sampling_rate, freqs=50, verbose=False)
    data = resample(data, sampling_rate, target_rate, axis=2, filter='kaiser_best')
    logging.info(f"[CBraMod] data shape after resampling: {data.shape}")

    n_samples = data.shape[2]
    n_seconds = np.floor(n_samples / target_rate).astype(int)
    new_n_samples = n_seconds * target_rate
    if new_n_samples > n_samples:
        padding = new_n_samples - n_samples
        data = np.pad(data, ((0, 0), (0, 0), (0, padding)), mode='constant', constant_values=0)
    elif new_n_samples < n_samples:
        data = data[:, :, :new_n_samples]

    for i in range(data.shape[0]):
        mean = np.mean(data[i], axis=1, keepdims=True)
        std = np.std(data[i], axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        data[i] = (data[i] - mean) / std

    if labels is not None:
        labels = np.array([map_label(label, task_name) for label in labels])
        labels = np.eye(n_unique_labels(task_name))[labels]

    if train:
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=split_size, random_state=42)
        return LaBraMBCIDataset(data_train, labels_train, target_rate, target_channels), LaBraMBCIDataset(data_val, labels_val, target_rate, target_channels)
    return LaBraMBCIDataset(data, labels, target_rate, target_channels)
