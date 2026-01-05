import importlib

from .base_clinical_dataset import BaseClinicalDataset

_DATASET_IMPORTS = {
    "Cavanagh2018aDataset": "eeg_bench.datasets.clinical.d001_cavanagh2018a",
    "Cavanagh2018bDataset": "eeg_bench.datasets.clinical.d002_cavanagh2018b",
    "Albrecht2019Dataset": "eeg_bench.datasets.clinical.d004_albrecht2019",
    "Singh2018Dataset": "eeg_bench.datasets.clinical.d005_singh2018",
    "Brown2020Dataset": "eeg_bench.datasets.clinical.d007_brown2020",
    "Gruendler2009Dataset": "eeg_bench.datasets.clinical.d008_gruendler2009",
    "Cavanagh2019Dataset": "eeg_bench.datasets.clinical.d009_cavanagh2019",
    "Singh2020Dataset": "eeg_bench.datasets.clinical.d011_singh2020",
    "Singh2021Dataset": "eeg_bench.datasets.clinical.d014_singh2021",
    "TUEGEpilepsyDataset": "eeg_bench.datasets.clinical.tueg_epilepsy",
    "TUEGAbnormalDataset": "eeg_bench.datasets.clinical.tueg_abnormal",
    "CHBMITDataset": "eeg_bench.datasets.clinical.chb_mit",
    "SleepTelemetryDataset": "eeg_bench.datasets.clinical.sleep_telemetry",
    "TUARDataset": "eeg_bench.datasets.clinical.tueg_artifact",
}

__all__ = ["BaseClinicalDataset", *_DATASET_IMPORTS.keys()]


def __getattr__(name: str):
    if name in _DATASET_IMPORTS:
        module = importlib.import_module(_DATASET_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
