import argparse
import csv
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bench.enums.split import Split
from eeg_bench.tasks.clinical import (
    AbnormalClinicalTask,
    SchizophreniaClinicalTask,
    MTBIClinicalTask,
    OCDClinicalTask,
    EpilepsyClinicalTask,
    ParkinsonsClinicalTask,
    SeizureClinicalTask,
    ArtifactBinaryClinicalTask,
    ArtifactMulticlassClinicalTask,
    SleepStagesClinicalTask,
)
from eeg_bench.tasks.bci import (
    LeftHandvRightHandMITask,
    RightHandvFeetMITask,
    LeftHandvRightHandvFeetvTongueMITask,
    FiveFingersMITask,
)
from eeg_bench.models.clinical import (
    BrainfeaturesLDAModel as BrainfeaturesLDA,
    BrainfeaturesSVMModel as BrainfeaturesSVM,
    LaBraMModel as LaBraMClinical,
    BENDRModel as BENDRClinical,
    NeuroGPTModel as NeuroGPTClinical,
    EEGLeJEPAClinicalModel as LeJEPAClinical,
    REVEClinicalModel as REVEClinical,
)
from eeg_bench.models.bci import (
    CSPLDAModel as CSPLDA,
    CSPSVMModel as CSPSVM,
    LaBraMModel as LaBraMBci,
    BENDRModel as BENDRBci,
    NeuroGPTModel as NeuroGPTBci,
    REVEBenchmarkModel as REVEBci,
    EEGLeJEPABCIModel as LeJEPABci
)
from eeg_bench.utils.evaluate_and_plot import print_classification_results, generate_classification_plots
from eeg_bench.utils.utils import set_seed, save_results, get_multilabel_tasks
from eeg_bench.config import get_config_value
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from eeg_bench.models.clinical.LaBraM.utils_2 import make_multilabels
from eeg_bench.utils import wandb_utils
# NOTE: Removed 'from asyncio.tasks import ALL_COMPLETED' as it was unused and caused an error in some environments.

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


ALL_TASKS_CLASSES = [
    ParkinsonsClinicalTask,
    SchizophreniaClinicalTask,
    MTBIClinicalTask,
    OCDClinicalTask,
    EpilepsyClinicalTask,
    AbnormalClinicalTask,
    SleepStagesClinicalTask,
    SeizureClinicalTask,
    ArtifactBinaryClinicalTask,
    ArtifactMulticlassClinicalTask,
    LeftHandvRightHandMITask,
    RightHandvFeetMITask,
    LeftHandvRightHandvFeetvTongueMITask,
    FiveFingersMITask,

]

def _subset_dataset(data, labels, frac, rng):
    if frac >= 1 or len(data) == 0:
        return data, labels
    n_keep = max(1, int(round(len(data) * frac)))
    idx = rng.choice(len(data), size=n_keep, replace=False)
    if isinstance(data, np.ndarray):
        data_sub = data[idx]
    else:
        data_sub = [data[i] for i in idx]
    if labels is None:
        labels_sub = None
    elif isinstance(labels, np.ndarray):
        labels_sub = labels[idx]
    else:
        labels_sub = [labels[i] for i in idx]
    return data_sub, labels_sub

def _subset_data(X, y, frac, rng):
    X_sub = []
    y_sub = []
    for data, labels in zip(X, y):
        data_sub, labels_sub = _subset_dataset(data, labels, frac, rng)
        X_sub.append(data_sub)
        y_sub.append(labels_sub)
    return X_sub, y_sub

def _calculate_metrics(y_true, y_pred, multilabel=False):
    if multilabel:
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
        }
    encoder = LabelEncoder()
    y_true_enc = encoder.fit_transform(y_true)
    y_pred_enc = encoder.transform(y_pred)
    return {
        "accuracy": accuracy_score(y_true_enc, y_pred_enc),
        "balanced_accuracy": balanced_accuracy_score(y_true_enc, y_pred_enc),
        "weighted_f1": f1_score(y_true_enc, y_pred_enc, average="weighted"),
        "macro_f1": f1_score(y_true_enc, y_pred_enc, average="macro"),
        "precision": precision_score(y_true_enc, y_pred_enc, average="weighted"),
        "recall": recall_score(y_true_enc, y_pred_enc, average="weighted"),
    }

def _append_csv_rows(rows, filename):
    file_path = Path(get_config_value("results")) / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not file_path.exists()
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def _collect_metrics_rows(task_name, dataset_names, y_true, y_pred, model_name, seed, rep, data_frac, ckpt_path):
    rows = []
    is_multilabel_task = task_name in get_multilabel_tasks()
    if is_multilabel_task:
        combined_y_true = np.concatenate([np.concatenate(y) for y in y_true])
        combined_y_pred = np.concatenate([np.concatenate(y) for y in y_pred])
        combined_metrics = _calculate_metrics(combined_y_true, combined_y_pred, multilabel=True)
    else:
        combined_y_true = np.concatenate(y_true)
        combined_y_pred = np.concatenate(y_pred)
        combined_metrics = _calculate_metrics(combined_y_true, combined_y_pred, multilabel=False)

    rows.append({
        "task": task_name,
        "dataset": "overall",
        "model": model_name,
        "seed": seed,
        "rep": rep,
        "data_frac": data_frac,
        "checkpoint_path": ckpt_path or "",
        **combined_metrics,
    })

    if task_name == "parkinsons_clinical":
        singh_index = dataset_names.index("Singh2020") if "Singh2020" in dataset_names else None
        if singh_index is not None:
            held_metrics = _calculate_metrics(y_true[singh_index], y_pred[singh_index], multilabel=False)
            rows.append({
                "task": task_name,
                "dataset": "held_out",
                "model": model_name,
                "seed": seed,
                "rep": rep,
                "data_frac": data_frac,
                "checkpoint_path": ckpt_path or "",
                **held_metrics,
            })
    elif task_name == "Left Hand vs Right Hand MI":
        zhou_index = dataset_names.index("Zhou2016") if "Zhou2016" in dataset_names else None
        if zhou_index is not None:
            held_metrics = _calculate_metrics(y_true[zhou_index], y_pred[zhou_index], multilabel=False)
            rows.append({
                "task": task_name,
                "dataset": "held_out",
                "model": model_name,
                "seed": seed,
                "rep": rep,
                "data_frac": data_frac,
                "checkpoint_path": ckpt_path or "",
                **held_metrics,
            })
    elif task_name == "Right Hand vs Feet MI":
        zhou_index = dataset_names.index("Zhou2016") if "Zhou2016" in dataset_names else None
        if zhou_index is not None:
            held_metrics = _calculate_metrics(y_true[zhou_index], y_pred[zhou_index], multilabel=False)
            rows.append({
                "task": task_name,
                "dataset": "held_out",
                "model": model_name,
                "seed": seed,
                "rep": rep,
                "data_frac": data_frac,
                "checkpoint_path": ckpt_path or "",
                **held_metrics,
            })

    for i, (this_y_true, this_y_pred) in enumerate(zip(y_true, y_pred)):
        if is_multilabel_task:
            dataset_metrics = _calculate_metrics(np.concatenate(this_y_true), np.concatenate(this_y_pred), multilabel=True)
        else:
            dataset_metrics = _calculate_metrics(this_y_true, this_y_pred, multilabel=False)
        rows.append({
            "task": task_name,
            "dataset": dataset_names[i],
            "model": model_name,
            "seed": seed,
            "rep": rep,
            "data_frac": data_frac,
            "checkpoint_path": ckpt_path or "",
            **dataset_metrics,
        })
    return rows

def benchmark(tasks, models, seed, reps=1, wandb_run=None, data_frac=1.0, ckpt_path=None, csv_filename="sweep_results.csv"): # Default reps=1
    print("running bench")
    if tasks=="full":
        tasks=[cls() for cls in ALL_TASKS_CLASSES] # Instantiate task classes here
    print(tasks)

    # for task in tasks:
    #     # --- ADD THIS DATA CHECK ---
    #     try:
    #         X_train, y_train, meta_train = task.get_data(Split.TRAIN)
    #         X_test, y_test, meta_test = task.get_data(Split.TEST)
            
    #         if len(X_train) == 0:
    #             logger.warning(f"Skipping task {task.name}: No training data found (check if dataset is downloaded).")
    #             continue
    #     except Exception as e:
    #         logger.error(f"Failed to load data for task {task.name}: {e}")
    #         continue
    
    for task in tasks:
        # Logging for Task Clarity

        
        logger.info(f"============================================================")
        logger.info(f"STARTING BENCHMARK for TASK: {task.name}") 
        logger.info(f"============================================================")
        
        X_train, y_train, meta_train = task.get_data(Split.TRAIN)
        X_test, y_test, meta_test = task.get_data(Split.TEST)

        metrics = task.get_metrics()
        dataset_names = [m["name"] for m in meta_train]
        models_names = []
        results = []
        y_trues = []
        y_trains = []
        is_multilabel_task = task.name in get_multilabel_tasks()
        
        for model_name, model_factory in tqdm(models, desc=f"Models for Task: {task.name}"):
            logger.info(f"--- Starting Model: {model_name}")
            
            for i in range(reps):
                # Logging for Repetition Clarity
                logger.info(f"--- REPETITION {i+1}/{reps} (Seed: {seed + i}) ---")
                
                set_seed(seed + i)  # set seed for reproducibility
                rng = np.random.default_rng(seed + i)
                X_train_use, y_train_use = _subset_data(X_train, y_train, data_frac, rng)
                
                if is_multilabel_task:
                    num_classes = len(task.clinical_classes) + 1
                    model = model_factory(num_classes=num_classes, num_labels_per_chunk=task.num_labels_per_chunk)
                    this_y_train = make_multilabels(X_train_use, y_train_use, task.event_map, task.chunk_len_s, task.num_labels_per_chunk, model.name)
                    this_y_test = make_multilabels(X_test, y_test, task.event_map, task.chunk_len_s, task.num_labels_per_chunk, model.name)
                else:
                    model = model_factory()
                    this_y_train = y_train_use
                    this_y_test = y_test
                
                print(model)

                if hasattr(model, "set_wandb_run"):
                    model.set_wandb_run(wandb_run)
                # model.fit(X_train, this_y_train, meta_train)
                y_pred = []
                for x, m in zip(X_test, meta_test):
                    y_pred.append(model.predict([x], [m]))

                models_names.append(str(model))
                results.append(y_pred)
                y_trues.append(this_y_test)
                y_trains.append(this_y_train)
                csv_rows = _collect_metrics_rows(
                    task.name,
                    dataset_names,
                    this_y_test,
                    y_pred,
                    str(model),
                    seed + i,
                    i + 1,
                    data_frac,
                    ckpt_path,
                )
                _append_csv_rows(csv_rows, csv_filename)

        save_results(y_trains, y_trues, models_names, results, dataset_names, task.name)
        print_classification_results(
            y_trains, y_trues, models_names, results, dataset_names, task.name, metrics
        )
        generate_classification_plots(y_trains, y_trues, models_names, results, dataset_names, task.name, metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Run EEG-Bench for a specific task and model."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to run. Options: full, parkinsons, schizophrenia, mtbi, ocd, epilepsy, abnormal, sleep_stages, seizure, binary_artifact, multiclass_artifact, left_right, right_feet, left_right_feet_tongue, 5_fingers"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use. Options: lda, svm, labram, bendr, neurogpt, reve, lejepa"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for reproducibility (default: 100)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of repetitions with different seeds for variability assessment"
    )
    parser.add_argument(
        "--data-frac",
        type=float,
        nargs="+",
        default=[1.0],
        help="Fractions of training data to use (e.g., --data-frac 1 0.5 0.25)"
    )
    parser.add_argument(
        "--lejepa-ckpt",
        type=str,
        default=None,
        help="Path to a LeJEPA checkpoint file"
    )
    parser.add_argument(
        "--lejepa-ckpt-dir",
        type=str,
        default=None,
        help="Directory of LeJEPA checkpoints to evaluate"
    )
    parser.add_argument(
        "--lejepa-base-path",
        type=str,
        default=None,
        help="Base path for LeJEPA lightning logs (overrides model default)"
    )
    parser.add_argument(
        "--lejepa-version",
        type=int,
        default=None,
        help="Version number under the LeJEPA base path (overrides model default)"
    )
    parser.add_argument(
        "--no-freeze-encoder",
        action="store_true",
        help="Unfreeze the LeJEPA encoder during downstream training"
    )
    
    # --- CHANGE 1: Added defaults to Project and Entity ---
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="eeg-bench-default", # Default project name (you can rename this)
        help="Weights & Biases project name for logging loss curves"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="saarangp-ucla", # Defaulted to your username
        help="Weights & Biases entity (team/user) for logging loss curves"
    )
    # ------------------------------------------------------

    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode (online, offline, disabled)"
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Weights & Biases group name for runs"
    )

    # --- CHANGE 2: Added a flag to explicitly DISABLE wandb ---
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Set this flag to disable WandB logging (overrides defaults)"
    )
    # ----------------------------------------------------------

    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Run all combinations of tasks and models"
    )
    args = parser.parse_args()

    # Mapping command-line strings to task classes
    tasks_map = {
        "parkinsons": ParkinsonsClinicalTask,
        "schizophrenia": SchizophreniaClinicalTask,
        "mtbi": MTBIClinicalTask,
        "ocd": OCDClinicalTask,
        "epilepsy": EpilepsyClinicalTask,
        "abnormal": AbnormalClinicalTask,
        "left_right": LeftHandvRightHandMITask,
        "right_feet": RightHandvFeetMITask,
        "left_right_feet_tongue": LeftHandvRightHandvFeetvTongueMITask,
        "5_fingers": FiveFingersMITask,
        "sleep_stages": SleepStagesClinicalTask,
        "seizure": SeizureClinicalTask,
        "binary_artifact": ArtifactBinaryClinicalTask,
        "multiclass_artifact": ArtifactMulticlassClinicalTask,
    }

    def _make_factory(model_cls, **fixed_kwargs):
        def _factory(**extra_kwargs):
            return model_cls(**fixed_kwargs, **extra_kwargs)
        return _factory

    if args.lejepa_ckpt and args.lejepa_ckpt_dir:
        parser.error("Use only one of --lejepa-ckpt or --lejepa-ckpt-dir.")

    if args.lejepa_ckpt_dir:
        ckpt_dir = Path(args.lejepa_ckpt_dir)
        ckpt_paths = sorted([str(p) for p in ckpt_dir.iterdir() if p.is_file() and p.suffix in {".ckpt", ".pth"}])
        if not ckpt_paths:
            parser.error(f"No checkpoints found in {ckpt_dir}.")
    else:
        ckpt_paths = [args.lejepa_ckpt] if args.lejepa_ckpt else [None]

    if (args.lejepa_ckpt or args.lejepa_ckpt_dir) and (args.model and args.model.lower() != "lejepa") and not args.all:
        parser.error("Checkpoint options are only supported with --model lejepa or --all.")

    freeze_encoder = not args.no_freeze_encoder

    def _build_model_maps(ckpt_path):
        lejepa_kwargs = {"freeze_encoder": freeze_encoder, "pretrained_path": ckpt_path}
        if args.lejepa_base_path is not None:
            lejepa_kwargs["base_path"] = args.lejepa_base_path
        if args.lejepa_version is not None:
            lejepa_kwargs["version"] = args.lejepa_version
        clinical_models_map = {
            "lda": ("lda", _make_factory(BrainfeaturesLDA)),
            "svm": ("svm", _make_factory(BrainfeaturesSVM)),
            "labram": ("labram", _make_factory(LaBraMClinical)),
            "bendr": ("bendr", _make_factory(BENDRClinical)),
            "neurogpt": ("neurogpt", _make_factory(NeuroGPTClinical)),
            "lejepa": ("lejepa", _make_factory(LeJEPAClinical, **lejepa_kwargs)),
            "reve": ("reve", _make_factory(REVEClinical)),
        }
        bci_models_map = {
            "lda": ("lda", _make_factory(CSPLDA)),
            "svm": ("svm", _make_factory(CSPSVM)),
            "labram": ("labram", _make_factory(LaBraMBci)),
            "bendr": ("bendr", _make_factory(BENDRBci)),
            "neurogpt": ("neurogpt", _make_factory(NeuroGPTBci)),
            "reve": ("reve", _make_factory(REVEBci)),
            "lejepa": ("lejepa", _make_factory(
                LeJEPABci,
                pretrained_path=ckpt_path,
                freeze_encoder=freeze_encoder,
            )),
        }
        return clinical_models_map, bci_models_map

    wandb_run = None
    
    # --- CHANGE 3: Logic now checks if "no_wandb" is FALSE ---
    # It will run by default because args.no_wandb is False unless the flag is used.
    if not args.no_wandb and args.wandb_mode != "disabled":
        wandb_run = wandb_utils.init_run(
            project=args.wandb_project, # Uses the default "eeg-bench-default"
            entity=args.wandb_entity,   # Uses the default "upanchavati"
            group=args.wandb_group,
            name="EEG-Bench",
            mode=args.wandb_mode,
            config={
                "seed": args.seed,
                "reps": args.reps,
                "task": args.task,
                "model": args.model,
                "all": args.all,
            },
        )
        wandb_utils.set_run(wandb_run)
    # ---------------------------------------------------------

    try:
        if args.all:
            logger.info("Running all task/model combinations...")
            for ckpt_path in ckpt_paths:
                for data_frac in args.data_frac:
                    clinical_models_map, bci_models_map = _build_model_maps(ckpt_path)
                    for task_key, task_cls in tasks_map.items():
                        if task_key in ["parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy", "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"]:
                            models_map = clinical_models_map
                        else:
                            models_map = bci_models_map

                        task_instance = task_cls()
                        model_specs = list(models_map.values())
                        benchmark([task_instance], model_specs, args.seed, args.reps, wandb_run=wandb_run, data_frac=data_frac, ckpt_path=ckpt_path)

        else:
            if not args.task or not args.model:
                parser.error("Both --task and --model must be specified unless --all is used.")
            
            task_key = args.task.lower()
            model_key = args.model.lower()
            
            if task_key == "full":
                tasks_to_run = "full" 
            elif task_key not in tasks_map:
                parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())} or 'full'")
            else:
                tasks_to_run = [tasks_map[task_key]()] 
            
            for ckpt_path in ckpt_paths:
                for data_frac in args.data_frac:
                    clinical_models_map, bci_models_map = _build_model_maps(ckpt_path)
                    if task_key in ["parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy", "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"]:
                        models_map = clinical_models_map
                    elif task_key in ["left_right", "right_feet", "left_right_feet_tongue", "5_fingers"]:
                        models_map = bci_models_map
                    elif task_key == "full": 
                        models_map = clinical_models_map 
                    else:
                        models_map = {}
                        parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())} or 'full'")
                    
                    if model_key not in models_map:
                        parser.error(f"Invalid model specified. Choose from: {', '.join(models_map.keys())}")
                    
                    model_spec = models_map[model_key]

                    benchmark(tasks_to_run, [model_spec], args.seed, args.reps, wandb_run=wandb_run, data_frac=data_frac, ckpt_path=ckpt_path)
    finally:
        wandb_utils.finish()

if __name__ == "__main__":
    main()
