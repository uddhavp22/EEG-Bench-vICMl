from mne.filter import filter_data, notch_filter


def process_filter(raw, sfreq):
    l_freq: float = 0.1
    h_freq: float = 75.0
    raw.load_data()
    raw.set_eeg_reference("average")
    raw.filter(l_freq=l_freq, h_freq=h_freq if h_freq < 0.5 * raw.info["sfreq"] else None)
    if 0.5 * raw.info["sfreq"] > 50.0:
        raw.notch_filter(50.0)
    raw.resample(sfreq)
    return raw
