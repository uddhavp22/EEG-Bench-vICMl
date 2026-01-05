import importlib

_MODEL_IMPORTS = {
    "BrainfeaturesLDAModel": "eeg_bench.models.clinical.brainfeatures_lda_model",
    "BrainfeaturesSVMModel": "eeg_bench.models.clinical.brainfeatures_svm_model",
    "LaBraMModel": "eeg_bench.models.clinical.labram_model",
    "BENDRModel": "eeg_bench.models.clinical.bendr_model",
    "NeuroGPTModel": "eeg_bench.models.clinical.neurogpt_model",
    "FartfmClinicalModel": "eeg_bench.models.clinical.fartfm_model",
    "REVEClinicalModel": "eeg_bench.models.clinical.reve_model",
}

__all__ = list(_MODEL_IMPORTS.keys())


def __getattr__(name: str):
    if name in _MODEL_IMPORTS:
        module = importlib.import_module(_MODEL_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
