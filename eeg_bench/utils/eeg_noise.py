"""EEG-aligned evaluation noise utilities.

These helpers are designed for *evaluation-time* robustness sweeps where the
linear probe stays fixed and noise is injected into the raw EEG prior to
encoding.
"""

from __future__ import annotations

from typing import Iterable, List, Optional
import math

import torch

VALID_NOISE_TYPES = {
    "gaussian",
    "one_over_f",
    "emg",
    "channel_dropout",
}


def _rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-(batch,channel) RMS over time with numerical safety."""
    return torch.sqrt(torch.clamp(x.pow(2).mean(dim=-1, keepdim=True), min=eps))


def _parse_noise_types(noise_types: Optional[Iterable[str]]) -> List[str]:
    if not noise_types:
        return []
    parsed = [str(t).strip().lower() for t in noise_types if str(t).strip()]
    unknown = sorted(set(parsed) - VALID_NOISE_TYPES)
    if unknown:
        raise ValueError(f"Unknown noise types: {unknown}. Valid: {sorted(VALID_NOISE_TYPES)}")
    # Preserve order while removing duplicates
    return list(dict.fromkeys(parsed))


def _make_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def _target_noise_rms(signal_ref: torch.Tensor, snr_db: float, power_split: float) -> torch.Tensor:
    """Compute target noise RMS for a given SNR and power split factor."""
    sig_rms = _rms(signal_ref)
    snr_linear = 10.0 ** (float(snr_db) / 20.0)
    return (sig_rms / snr_linear) * power_split


def _scale_noise_to_snr(
    signal_ref: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    power_split: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale a noise tensor to match a target SNR relative to signal_ref."""
    target = _target_noise_rms(signal_ref, snr_db, power_split)
    noise_rms = _rms(noise, eps=eps)
    scale = target / torch.clamp(noise_rms, min=eps)
    return noise * scale


def _band_limited_noise(
    shape: torch.Size,
    sfreq: float,
    fmin: float,
    fmax: float,
    alpha: float = 0.0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate band-limited noise using an FFT-domain construction."""
    if len(shape) != 3:
        raise ValueError(f"Expected shape (B, C, T); got {tuple(shape)}")
    b, c, t = shape
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    nyq = 0.5 * float(sfreq)
    fmin = max(0.0, float(fmin))
    fmax = min(float(fmax), nyq * 0.999)
    if fmax <= fmin or t < 4:
        return torch.zeros(shape, device=device, dtype=dtype)

    freqs = torch.fft.rfftfreq(t, d=1.0 / float(sfreq)).to(device=device)
    mask = (freqs >= fmin) & (freqs <= fmax)

    # 1/f^alpha shaping within the band; avoid singularity at 0 Hz.
    if alpha > 0.0:
        denom = torch.clamp(freqs, min=max(fmin, 1e-3))
        weights = (1.0 / denom.pow(alpha)) * mask
    else:
        weights = mask.to(freqs.dtype)

    # Ensure DC component is not inflated by shaping.
    weights = weights.clone()
    weights[0] = 0.0

    n_freq = freqs.numel()
    real = torch.randn((b, c, n_freq), generator=generator, device=device, dtype=dtype)
    imag = torch.randn((b, c, n_freq), generator=generator, device=device, dtype=dtype)
    spectrum = torch.complex(real, imag) * weights

    noise = torch.fft.irfft(spectrum, n=t, dim=-1)

    # Normalize to unit RMS per (B,C) to make SNR scaling predictable.
    noise_rms = _rms(noise)
    noise = noise / torch.clamp(noise_rms, min=1e-6)
    return noise.to(dtype=dtype)


def apply_eeg_noise(
    x: torch.Tensor,
    *,
    sfreq: float,
    snr_db: Optional[float],
    noise_types: Optional[Iterable[str]],
    channel_dropout_prob: float = 0.0,
    one_over_f_band: tuple[float, float] = (0.5, 40.0),
    emg_band: tuple[float, float] = (30.0, 100.0),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Apply EEG-aligned evaluation noise to a (B, C, T) tensor.

    Notes:
    - `snr_db=None` or `math.isinf(snr_db)` returns the input unchanged.
    - If multiple additive noise types are used, their power is split evenly
      so the *combined* noise approximately matches the requested SNR.
    """
    if snr_db is None or math.isinf(float(snr_db)):
        return x

    noise_types_list = _parse_noise_types(noise_types)
    if not noise_types_list:
        return x

    x_noisy = x
    device = x.device
    dtype = x.dtype
    gen = _make_generator(device, seed)

    # Channel dropout is multiplicative; apply first.
    if "channel_dropout" in noise_types_list and channel_dropout_prob > 0.0:
        keep_prob = float(max(0.0, min(1.0, 1.0 - channel_dropout_prob)))
        mask = torch.bernoulli(
            torch.full((x.shape[0], x.shape[1], 1), keep_prob, device=device, dtype=dtype),
            generator=gen,
        )
        x_noisy = x_noisy * mask

    additive_types = [t for t in noise_types_list if t != "channel_dropout"]
    if not additive_types:
        return x_noisy

    # Split power across additive types so total noise stays near target SNR.
    power_split = 1.0 / math.sqrt(len(additive_types))
    signal_ref = x  # Keep SNR referenced to clean signal.

    for noise_type in additive_types:
        if noise_type == "gaussian":
            noise = torch.randn(x_noisy.shape, device=device, dtype=dtype, generator=gen)
        elif noise_type == "one_over_f":
            fmin, fmax = one_over_f_band
            noise = _band_limited_noise(
                x_noisy.shape,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                alpha=1.0,
                generator=gen,
                device=device,
                dtype=dtype,
            )
        elif noise_type == "emg":
            nyq = 0.5 * float(sfreq)
            fmin, fmax = emg_band
            # Keep EMG band within Nyquist.
            fmax = min(float(fmax), nyq * 0.95)
            noise = _band_limited_noise(
                x_noisy.shape,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                alpha=0.0,
                generator=gen,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        noise = _scale_noise_to_snr(signal_ref, noise, float(snr_db), power_split=power_split)
        x_noisy = x_noisy + noise

    return x_noisy


def format_noise_tag(noise_types: Optional[Iterable[str]], snr_db: Optional[float]) -> str:
    """Create a compact tag for filenames/labels."""
    types_list = _parse_noise_types(noise_types)
    if not types_list or snr_db is None or math.isinf(float(snr_db)):
        return "clean"
    types_tag = "+".join(types_list)
    snr_tag = str(int(float(snr_db))) if float(snr_db).is_integer() else str(snr_db)
    return f"noise_{types_tag}_snr{snr_tag}dB"
