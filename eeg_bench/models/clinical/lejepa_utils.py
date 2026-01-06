import numpy as np
from mne.filter import filter_data, notch_filter
from resampy import resample
from .preprocess_utils import process_filter


def apply_lejepa_scaling(signals: np.ndarray) -> np.ndarray:
    signals = signals * 1e6
    signals -= np.median(signals, axis=0, keepdims=True)
    scale = np.percentile(signals, 75, axis=None) - np.percentile(signals, 25, axis=None)
    if scale < 1e-6:
        scale = 1.0
    return np.clip(signals / scale, -20.0, 20.0).astype(np.float32)


def process_lejepa(raw, chs, out_sfreq=250):
    raw = raw.reorder_channels(chs)
    max_duration_s = 30 * 60
    if raw.times[-1] > max_duration_s:
        raw.crop(tmax=max_duration_s)
    raw = process_filter(raw, out_sfreq)
    signals = raw.get_data(units="uV")
    return apply_lejepa_scaling(signals)


def preprocess_lejepa_clinical(
    signals: np.ndarray,
    o_channels,
    required_channels,
    sfreq: int,
    l_freq: float = 0.1,
    h_freq: float = 75.0,
    out_freq: int = 250,
    max_duration_s: int = 30 * 60,
):
    ch_names = [ch.upper() for ch in o_channels]
    required_channels = [c.upper() for c in required_channels]
    target_channels = [ch for ch in required_channels if ch in ch_names]

    if len(target_channels) == 0:
        raise ValueError("No required LeJEPA clinical channels found in recording")

    signals = signals[[ch_names.index(ch) for ch in target_channels], :]

    if signals.shape[1] > int(max_duration_s * sfreq):
        signals = signals[:, : int(max_duration_s * sfreq)]

    signals = filter_data(
        signals.astype(np.float64),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        verbose=False,
    )
    signals = notch_filter(signals, Fs=sfreq, freqs=50, verbose=False)
    signals = resample(signals.astype(np.float32), sfreq, out_freq, axis=1, filter="kaiser_best")
    signals = apply_lejepa_scaling(signals)
    return signals, out_freq, target_channels
