import importlib

_MODEL_IMPORTS = {
    "BENDRModel": "eeg_bench.models.bci.bendr_model",
    "CSPLDAModel": "eeg_bench.models.bci.csp_lda_model",
    "CSPSVMModel": "eeg_bench.models.bci.csp_svm_model",
    "LaBraMModel": "eeg_bench.models.bci.labram_model",
    "NeuroGPTModel": "eeg_bench.models.bci.neurogpt_model",
    "REVEBenchmarkModel": "eeg_bench.models.bci.reve_model",
    "FartfmBCIModel": "eeg_bench.models.bci.fartfm_model",
}

__all__ = list(_MODEL_IMPORTS.keys())


def __getattr__(name: str):
    if name in _MODEL_IMPORTS:
        module = importlib.import_module(_MODEL_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
