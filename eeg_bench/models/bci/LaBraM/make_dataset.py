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


def apply_defossez_scaling(
    signals: np.ndarray,
    is_uv: bool = None,
    clip_range: tuple = (-20.0, 20.0),
    min_scale: float = 1e-6
) -> np.ndarray:
    """
    Apply Defossez-style robust scaling for LeJEPA preprocessing.

    Args:
        signals: Input EEG signals, shape (n_trials, n_channels, n_timepoints) or (n_channels, n_timepoints)
        is_uv: If True, data is already in microvolts; if False, converts from V to uV.
               If None (default), auto-detects based on data magnitude.
        clip_range: Tuple of (min, max) values for clipping after scaling
        min_scale: Minimum scale value to prevent division by near-zero

    Returns:
        Scaled signals as float32 array
    """
    # Auto-detect if data is in volts or microvolts
    if is_uv is None:
        # EEG in volts: typical magnitude ~1e-6 to 1e-4
        # EEG in microvolts: typical magnitude ~1 to 1000
        median_magnitude = np.median(np.abs(signals))
        is_uv = median_magnitude > 1e-3  # If median > 1mV, assume already in µV

    if not is_uv:
        signals = signals * 1e6
    signals = signals - np.median(signals, axis=-1, keepdims=True)
    scale = np.percentile(signals, 75, axis=None) - np.percentile(signals, 25, axis=None)
    if scale < min_scale:
        scale = 1.0
    signals = np.clip(signals / scale, clip_range[0], clip_range[1])
    return signals.astype(np.float32)


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
        target_channels = list(set([ch.upper() for ch in standard_1020]).intersection(set(ch_names)))
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
        target_channels = list(set([ch.upper() for ch in standard_1020]).intersection(set(ch_names)))
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]

    # Bandpass filter
    data = filter_data(data, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    # Notch filter
    data = notch_filter(data, Fs=sampling_rate, freqs=50, verbose=False)
    # Resample to 250 Hz (LeJEPA training expectation)
    data = resample(data, sampling_rate, target_rate, axis=2, filter='kaiser_best')

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
    data = apply_defossez_scaling(data)

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

