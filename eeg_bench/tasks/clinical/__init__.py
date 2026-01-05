import importlib

from .abstract_clinical_task import AbstractClinicalTask

_TASK_IMPORTS = {
    "AbnormalClinicalTask": "eeg_bench.tasks.clinical.abnormal_clinical_task",
    "SchizophreniaClinicalTask": "eeg_bench.tasks.clinical.schizophrenia_clinical_task",
    "MTBIClinicalTask": "eeg_bench.tasks.clinical.mtbi_clinical_task",
    "OCDClinicalTask": "eeg_bench.tasks.clinical.ocd_clinical_task",
    "EpilepsyClinicalTask": "eeg_bench.tasks.clinical.epilepsy_clinical_task",
    "ParkinsonsClinicalTask": "eeg_bench.tasks.clinical.parkinsons_clinical_task",
    "SeizureClinicalTask": "eeg_bench.tasks.clinical.seizure_clinical_task",
    "SleepStagesClinicalTask": "eeg_bench.tasks.clinical.sleep_stages_clinical_task",
    "ArtifactMulticlassClinicalTask": "eeg_bench.tasks.clinical.multiclass_artifact_clinical_task",
    "ArtifactBinaryClinicalTask": "eeg_bench.tasks.clinical.binary_artifact_clinical_task",
}

__all__ = ["AbstractClinicalTask", *_TASK_IMPORTS.keys()]


def __getattr__(name: str):
    if name in _TASK_IMPORTS:
        module = importlib.import_module(_TASK_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
