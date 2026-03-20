import mne
p = "/radraid2/spanchavati/EEGBench/data/tuab/train/abnormal/01_tcp_ar/aaaaamkt_s001_t000.edf"

raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
print("opened")

raw.load_data()
print("load_data ok")

r = raw.copy()
r.set_eeg_reference("average")
print("ref ok")

r.filter(0.1, None)
print("filter ok")

r.notch_filter([50.0, 60.0])
print("notch ok")

r.resample(256)
print("resample ok")

x = r.get_data()   # no units arg
print("get_data ok", x.shape)