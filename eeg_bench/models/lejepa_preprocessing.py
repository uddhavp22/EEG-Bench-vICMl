from __future__ import annotations

import numpy as np
from mne.filter import filter_data, notch_filter
from resampy import resample


def apply_defossez_scaling(
    signals: np.ndarray,
    is_uv: bool | None = None,
    clip_range: tuple = (-20.0, 20.0),
    min_scale: float = 1e-6,
    median_axis: int | None = -1,
    scale_axis: int | None = None,
) -> np.ndarray:
    """Apply Defossez-style robust scaling for LeJEPA preprocessing."""
    if is_uv is None:
        median_magnitude = np.median(np.abs(signals))
        is_uv = median_magnitude > 1e-3

    if not is_uv:
        signals = signals * 1e6
    signals = signals - np.median(signals, axis=median_axis, keepdims=True)
    scale = np.percentile(signals, 75, axis=scale_axis) - np.percentile(
        signals, 25, axis=scale_axis
    )
    if scale < min_scale:
        scale = 1.0
    signals = np.clip(signals / scale, clip_range[0], clip_range[1])
    return signals.astype(np.float32)


def filter_resample_array(
    signals: np.ndarray,
    sfreq: float,
    out_sfreq: float,
    l_freq: float = 0.5,
    h_freq: float = 100.0,
) -> np.ndarray:
    signals = filter_data(
        signals.astype(np.float64),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq if sfreq//2>100 else 75,
        method="fir",
        verbose=False,
    )
    freqs = [50, 60] if 0.5 * sfreq > 60.0 else ([50] if 0.5 * sfreq > 50.0 else [])
    if freqs:
        signals = notch_filter(signals, Fs=sfreq, freqs=freqs, verbose=False)
    return resample(signals.astype(np.float32), sfreq, out_sfreq, axis=-1, filter="kaiser_best")


def process_lejepa_raw(raw, out_sfreq: float = 250) -> np.ndarray:
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=100.0 if 0.5 * raw.info["sfreq"] > 100.0 else None)
    if 0.5 * raw.info["sfreq"] > 60.0:
        raw.notch_filter([50.0, 60.0])
    elif 0.5 * raw.info["sfreq"] > 50.0:
        raw.notch_filter([50.0])
    raw.resample(out_sfreq)
    signals = raw.get_data(units="uV")
    return apply_defossez_scaling(signals, median_axis=0, scale_axis=None)
