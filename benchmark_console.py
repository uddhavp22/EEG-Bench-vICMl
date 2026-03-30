import argparse
import logging
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple
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
    LUNAClinicalModel as LUNAClinical,
    SJEPAClinicalModel as SJEPAClinical,
    CBraModClinicalModel as CBraModClinical,
)
from eeg_bench.models.bci import (
    CSPLDAModel as CSPLDA,
    CSPSVMModel as CSPSVM,
    LaBraMModel as LaBraMBci,
    BENDRModel as BENDRBci,
    NeuroGPTModel as NeuroGPTBci,
    REVEBenchmarkModel as REVEBci,
    EEGLeJEPABCIModel as LeJEPABci,
    LUNABCIModel as LUNABci,
    SJEPABCIModel as SJEPABci,
    CBraModBCIModel as CBraModBci,
)
from eeg_bench.utils.evaluate_and_plot import print_classification_results, generate_classification_plots
from eeg_bench.utils.utils import set_seed, save_results, get_multilabel_tasks, subsample_data_stratified
from eeg_bench.models.clinical.LaBraM.utils_2 import make_multilabels
from eeg_bench.utils import wandb_utils
from eeg_bench.config import load_config, load_lejepa_config, merge_lejepa_config_with_cli
from eeg_bench.utils.eeg_noise import VALID_NOISE_TYPES
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
CLINICAL_TASK_KEYS = {
    "parkinsons",
    "schizophrenia",
    "mtbi",
    "ocd",
    "epilepsy",
    "abnormal",
    "sleep_stages",
    "seizure",
    "binary_artifact",
    "multiclass_artifact",
}


def _is_clinical_task_key(task_key: str) -> bool:
    return task_key in CLINICAL_TASK_KEYS

def _build_eval_conditions(
    eval_noise_types: Optional[Iterable[str]],
    eval_noise_levels_db: Optional[Iterable[float]],
    mix_all: bool = False,
) -> List[Tuple[Optional[List[str]], Optional[float]]]:
    if not eval_noise_types or not eval_noise_levels_db:
        return [(None, None)]
    noise_types_list = [str(t).strip().lower() for t in eval_noise_types if str(t).strip()]
    noise_levels_list = [float(x) for x in eval_noise_levels_db]
    if mix_all:
        return [(noise_types_list, snr_db) for snr_db in noise_levels_list]
    return [([noise_type], snr_db) for noise_type, snr_db in product(noise_types_list, noise_levels_list)]


def benchmark(
    tasks,
    models,
    seed,
    reps=1,
    wandb_run=None,
    data_percentages=None,
    linear_probe=False,
    result_prefix=None,
    checkpoint_id=None,
    eval_noise_types: Optional[Iterable[str]] = None,
    eval_noise_levels_db: Optional[Iterable[float]] = None,
    eval_noise_channel_dropout_prob: float = 0.0,
    eval_noise_mix_all: bool = False,
):
    print("running bench")
    if tasks=="full":
        tasks=[cls() for cls in ALL_TASKS_CLASSES] # Instantiate task classes here
    print(tasks)

    if data_percentages is None:
        data_percentages = [1.0]

    if linear_probe:
        logger.info("Running in LINEAR PROBE mode (encoders frozen)")

    for task in tasks:
        # Logging for Task Clarity

        logger.info(f"============================================================")
        logger.info(f"STARTING BENCHMARK for TASK: {task.name}")
        logger.info(f"============================================================")

        X_train_full, y_train_full, meta_train = task.get_data(Split.TRAIN)
        X_test, y_test, meta_test = task.get_data(Split.TEST)

        metrics = task.get_metrics()
        dataset_names = [m["name"] for m in meta_train]
        is_multilabel_task = task.name in get_multilabel_tasks()
        eval_conditions = _build_eval_conditions(
            eval_noise_types,
            eval_noise_levels_db,
            mix_all=eval_noise_mix_all,
        )

        for pct_idx, percentage in enumerate(data_percentages):
            logger.info(f"============================================================")
            logger.info(f"DATA PERCENTAGE: {int(percentage * 100)}%")
            logger.info(f"============================================================")

            subset_seed = seed + pct_idx
            X_train, y_train, data_stats, selected_indices = subsample_data_stratified(
                X_train_full,
                y_train_full,
                percentage,
                random_state=subset_seed,
                return_indices=True,
            )
            logger.info(f"Training samples: {data_stats['total_samples']}, per class: {data_stats['samples_per_class']}")

            # Reset collectors per noise condition for this percentage
            condition_collectors: Dict[
                Tuple[Tuple[str, ...], Optional[float]],
                Dict[str, object],
            ] = {}
            for noise_types, snr_db in eval_conditions:
                key = (tuple(noise_types) if noise_types else tuple(), snr_db)
                condition_collectors[key] = {
                    "noise_types": noise_types,
                    "snr_db": snr_db,
                    "models_names": [],
                    "results": [],
                    "y_trues": [],
                    "y_trains": [],
                }

            for model_entry in tqdm(models, desc=f"Models for Task: {task.name} ({int(percentage*100)}%)"):
                model_name = model_entry.__name__ if hasattr(model_entry, "__name__") else str(model_entry)
                logger.info(f"--- Starting Model: {model_name}")

                for rep_idx in range(reps):
                    rep_seed = seed + rep_idx
                    logger.info(f"--- REPETITION {rep_idx + 1}/{reps} (Seed: {rep_seed}) ---")
                    set_seed(rep_seed)

                    if is_multilabel_task:
                        num_classes = len(task.clinical_classes) + 1
                        model = model_entry(num_classes=num_classes, num_labels_per_chunk=task.num_labels_per_chunk)
                        full_y_train = make_multilabels(
                            X_train_full,
                            y_train_full,
                            task.event_map,
                            task.chunk_len_s,
                            task.num_labels_per_chunk,
                            model.name,
                        )
                        this_y_train = [
                            [full_y_train[ds_idx][i] for i in ds_indices]
                            for ds_idx, ds_indices in enumerate(selected_indices)
                        ]
                        this_y_test = make_multilabels(
                            X_test,
                            y_test,
                            task.event_map,
                            task.chunk_len_s,
                            task.num_labels_per_chunk,
                            model.name,
                        )
                    else:
                        model = model_entry()
                        full_y_train = y_train_full
                        this_y_train = y_train
                        this_y_test = y_test

                    print(model)

                    if hasattr(model, "set_wandb_run"):
                        model.set_wandb_run(wandb_run)

                    if getattr(model, "supports_full_dataset_cache", False):
                        model.fit(
                            X_train_full,
                            full_y_train,
                            meta_train,
                            subset_fraction=percentage,
                            subset_seed=subset_seed,
                            subset_indices=selected_indices,
                        )
                        train_labels_used = this_y_train
                    else:
                        model.fit(X_train, this_y_train, meta_train)
                        train_labels_used = this_y_train

                    # Evaluate across noise conditions without retraining
                    for noise_types, snr_db in eval_conditions:
                        if hasattr(model, "set_eval_noise"):
                            model.set_eval_noise(
                                noise_types=noise_types,
                                snr_db=snr_db,
                                channel_dropout_prob=eval_noise_channel_dropout_prob,
                                seed=rep_seed,
                            )

                        y_pred = [model.predict([x], [m]) for x, m in zip(X_test, meta_test)]

                        key = (tuple(noise_types) if noise_types else tuple(), snr_db)
                        collector = condition_collectors[key]
                        collector["models_names"].append(str(model))
                        collector["results"].append(y_pred)
                        collector["y_trues"].append(this_y_test)
                        collector["y_trains"].append(train_labels_used)

                    # Reset to clean after sweep to avoid accidental carryover
                    if hasattr(model, "set_eval_noise"):
                        model.set_eval_noise(noise_types=None, snr_db=None)

            # Persist and report results per noise condition
            for collector in condition_collectors.values():
                noise_types = collector["noise_types"]
                snr_db = collector["snr_db"]
                models_names = collector["models_names"]
                results = collector["results"]
                y_trues = collector["y_trues"]
                y_trains = collector["y_trains"]

                save_results(
                    y_trains,
                    y_trues,
                    models_names,
                    results,
                    dataset_names,
                    task.name,
                    data_percentage=percentage,
                    data_stats=data_stats,
                    linear_probe=linear_probe,
                    result_prefix=result_prefix,
                    checkpoint_id=checkpoint_id,
                    eval_noise_types=noise_types if noise_types else None,
                    eval_noise_snr_db=snr_db,
                    eval_noise_channel_dropout_prob=eval_noise_channel_dropout_prob if noise_types else None,
                )
                print_classification_results(
                    y_trains, y_trues, models_names, results, dataset_names, task.name, metrics
                )
                generate_classification_plots(
                    y_trains, y_trues, models_names, results, dataset_names, task.name, metrics
                )


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
        help="Model to use. Options: lda, svm, labram, bendr, neurogpt, reve, lejepa, luna, sjepa, cbramod"
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

    parser.add_argument(
        "--data-percentages",
        type=float,
        nargs="+",
        default=None,
        help="Training data percentages to test (e.g., 0.01 0.1 0.25 0.5 0.75 1.0). Runs benchmark at each percentage for data efficiency analysis."
    )

    parser.add_argument(
        "--linear-probe",
        action="store_true",
        default=False,
        help="Freeze encoder and train only the classification head (linear probe evaluation). Applies to all foundation models."
    )
    parser.add_argument(
        "--eval-noise-types",
        type=str,
        nargs="+",
        default=None,
        help=f"Evaluation-time noise types to sweep (e.g., gaussian one_over_f emg channel_dropout). Valid: {sorted(VALID_NOISE_TYPES)}",
    )
    parser.add_argument(
        "--eval-noise-levels-db",
        type=float,
        nargs="+",
        default=None,
        help="Evaluation-time SNR levels in dB to sweep (e.g., 30 20 10 0). Requires --eval-noise-types.",
    )
    parser.add_argument(
        "--eval-noise-channel-dropout-prob",
        type=float,
        default=0.0,
        help="Channel dropout probability used when channel_dropout is included in --eval-noise-types.",
    )
    parser.add_argument(
        "--eval-noise-mix-all",
        action="store_true",
        help="Apply all --eval-noise-types simultaneously at each SNR level (instead of one type at a time).",
    )
    parser.add_argument(
        "--luna-pretrained-path",
        type=str,
        default=None,
        help="Path to LUNA pretrained checkpoint (.safetensors)"
    )
    parser.add_argument(
        "--luna-biofoundation-path",
        type=str,
        default=None,
        help="Path to BioFoundation repo used by LUNA"
    )
    parser.add_argument(
        "--cbramod-pretrained-path",
        type=str,
        default=None,
        help="Path to CBraMod pretrained checkpoint (.pth)"
    )
    parser.add_argument(
        "--cbramod-path",
        type=str,
        default=None,
        help="Path to CBraMod repo used for imports"
    )

    # LeJEPA configuration
    parser.add_argument(
        "--lejepa-config",
        type=str,
        default=None,
        help="Path to LeJEPA JSON config file (overrides default config.json lejepa section)"
    )
    parser.add_argument(
        "--lejepa-checkpoint-base-path",
        type=str,
        default=None,
        help="Base path for LeJEPA checkpoint (uses version subdir)"
    )
    parser.add_argument(
        "--lejepa-checkpoint-version",
        type=int,
        default=None,
        help="Checkpoint version subdirectory for LeJEPA"
    )
    parser.add_argument(
        "--lejepa-checkpoint-full-path",
        type=str,
        default=None,
        help="Full checkpoint path for LeJEPA (overrides base_path+version)"
    )
    parser.add_argument(
        "--lejepa-pos-bank-path",
        type=str,
        default=None,
        help="Local fallback path for REVE position bank"
    )
    parser.add_argument(
        "--lejepa-freeze-encoder",
        action="store_true",
        help="Freeze the LeJEPA encoder during training"
    )
    parser.add_argument(
        "--lejepa-no-freeze-encoder",
        action="store_true",
        help="Do NOT freeze the LeJEPA encoder (allow fine-tuning)"
    )

    # Result file naming (for sweep scripts)
    parser.add_argument(
        "--result-prefix",
        type=str,
        default=None,
        help="Prefix to add to result filename (e.g., model size name)"
    )
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help="Checkpoint identifier to include in result filename (e.g., 'epoch_10', 'last')"
    )

    args = parser.parse_args()

    # Warn about conflicting flags
    if args.linear_probe and getattr(args, 'lejepa_no_freeze_encoder', False):
        logger.warning("--linear-probe and --lejepa-no-freeze-encoder conflict. "
                       "Model-specific flag takes precedence (encoder will NOT be frozen for LeJEPA).")

    # Validate eval-noise arguments early
    if args.eval_noise_types and not args.eval_noise_levels_db:
        parser.error("--eval-noise-levels-db is required when --eval-noise-types is provided.")
    if args.eval_noise_levels_db and not args.eval_noise_types:
        parser.error("--eval-noise-types is required when --eval-noise-levels-db is provided.")
    if args.eval_noise_types:
        requested = {t.strip().lower() for t in args.eval_noise_types}
        unknown = sorted(requested - set(VALID_NOISE_TYPES))
        if unknown:
            parser.error(f"Unknown --eval-noise-types: {unknown}. Valid: {sorted(VALID_NOISE_TYPES)}")

    # Load and merge LeJEPA configuration
    lejepa_config = merge_lejepa_config_with_cli(
        load_lejepa_config(args.lejepa_config), args
    )

    # Apply --linear-probe to LeJEPA config (model-specific flags already override via merge)
    if args.linear_probe and not getattr(args, 'lejepa_no_freeze_encoder', False):
        lejepa_config.freeze_encoder = True

    raw_config = load_config()
    luna_config = raw_config.get("luna", {}) if isinstance(raw_config.get("luna", {}), dict) else {}
    cbramod_config = raw_config.get("cbramod", {}) if isinstance(raw_config.get("cbramod", {}), dict) else {}
    luna_pretrained_path = args.luna_pretrained_path or luna_config.get("pretrained_path")
    luna_biofoundation_path = args.luna_biofoundation_path or luna_config.get("biofoundation_path")
    cbramod_pretrained_path = args.cbramod_pretrained_path or cbramod_config.get("pretrained_path")
    cbramod_path = args.cbramod_path or cbramod_config.get("path")

    # Factory functions for LeJEPA models (to inject config)
    def make_lejepa_clinical(num_classes=2, num_labels_per_chunk=None):
        return LeJEPAClinical(
            config=lejepa_config,
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk
        )

    def make_lejepa_bci():
        return LeJEPABci(config=lejepa_config)

    # Factory functions for other models with freeze_encoder support
    def make_labram_clinical(num_classes=2, num_labels_per_chunk=None):
        return LaBraMClinical(num_classes=num_classes,
                              num_labels_per_chunk=num_labels_per_chunk,
                              freeze_encoder=True,
                              linear_probe=args.linear_probe)

    def make_labram_bci():
        return LaBraMBci(freeze_encoder=True, linear_probe=args.linear_probe)

    def make_bendr_clinical(num_classes=2, num_labels_per_chunk=None):
        return BENDRClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            freeze_encoder=True,
        )

    def make_bendr_bci():
        return BENDRBci(freeze_encoder=True)

    def make_neurogpt_clinical(num_classes=2, num_labels_per_chunk=None):
        return NeuroGPTClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            freeze_encoder=True,
        )

    def make_neurogpt_bci():
        return NeuroGPTBci(freeze_encoder=True)

    def make_reve_clinical(num_classes=2, num_labels_per_chunk=None):
        return REVEClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            freeze_backbone=True,
            linear_probe=args.linear_probe,
        )

    def make_reve_bci():
        return REVEBci(freeze_backbone=True, linear_probe=args.linear_probe)

    def make_luna_clinical(num_classes=2, num_labels_per_chunk=None):
        return LUNAClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            pretrained_path=luna_pretrained_path,
            biofoundation_path=luna_biofoundation_path,
            freeze_backbone=True,
            linear_probe=args.linear_probe,
        )

    def make_luna_bci():
        return LUNABci(
            pretrained_path=luna_pretrained_path,
            biofoundation_path=luna_biofoundation_path,
            freeze_backbone=True,
            linear_probe=args.linear_probe,
        )

    def make_sjepa_clinical(num_classes=2, num_labels_per_chunk=None):
        return SJEPAClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            freeze_encoder=True,
        )

    def make_sjepa_bci():
        return SJEPABci(freeze_encoder=True)

    def make_cbramod_clinical(num_classes=2, num_labels_per_chunk=None):
        return CBraModClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            pretrained_path=cbramod_pretrained_path,
            cbramod_path=cbramod_path,
            freeze_backbone=True,
            linear_probe=args.linear_probe,
        )

    def make_cbramod_bci():
        return CBraModBci(
            pretrained_path=cbramod_pretrained_path,
            cbramod_path=cbramod_path,
            freeze_backbone=True,
            linear_probe=args.linear_probe,
        )

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

    # Mapping command-line strings to model classes (or factory functions)
    clinical_models_map = {
        "lda": BrainfeaturesLDA,
        "svm": BrainfeaturesSVM,
        "labram": make_labram_clinical,
        "bendr": make_bendr_clinical,
        "neurogpt": make_neurogpt_clinical,
        "lejepa": make_lejepa_clinical,
        "reve": make_reve_clinical,
        "luna": make_luna_clinical,
        "sjepa": make_sjepa_clinical,
        "cbramod": make_cbramod_clinical,
    }
    bci_models_map = {
        "lda": CSPLDA,
        "svm": CSPSVM,
        "labram": make_labram_bci,
        "bendr": make_bendr_bci,
        "neurogpt": make_neurogpt_bci,
        "reve": make_reve_bci,
        "lejepa": make_lejepa_bci,
        "luna": make_luna_bci,
        "sjepa": make_sjepa_bci,
        "cbramod": make_cbramod_bci,
    }

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
                "data_percentages": args.data_percentages,
                "linear_probe": args.linear_probe,
                "eval_noise_types": args.eval_noise_types,
                "eval_noise_levels_db": args.eval_noise_levels_db,
                "eval_noise_channel_dropout_prob": args.eval_noise_channel_dropout_prob,
                "eval_noise_mix_all": args.eval_noise_mix_all,
                "luna_pretrained_path": luna_pretrained_path,
                "luna_biofoundation_path": luna_biofoundation_path,
                "cbramod_pretrained_path": cbramod_pretrained_path,
                "cbramod_path": cbramod_path,
            },
        )
        wandb_utils.set_run(wandb_run)
    # ---------------------------------------------------------

    try:
        if args.all:
            logger.info("Running all task/model combinations...")
            for task_key, task_cls in tasks_map.items():
                if _is_clinical_task_key(task_key):
                    models_map = clinical_models_map
                else:
                    models_map = bci_models_map

                task_instance = task_cls()
                model_classes = list(models_map.values())
                benchmark([task_instance], model_classes, args.seed, args.reps, wandb_run=wandb_run,
                         data_percentages=args.data_percentages, linear_probe=args.linear_probe,
                         result_prefix=args.result_prefix, checkpoint_id=args.checkpoint_id,
                         eval_noise_types=args.eval_noise_types,
                         eval_noise_levels_db=args.eval_noise_levels_db,
                         eval_noise_channel_dropout_prob=args.eval_noise_channel_dropout_prob,
                         eval_noise_mix_all=args.eval_noise_mix_all)

        else:
            if not args.task or not args.model:
                parser.error("Both --task and --model must be specified unless --all is used.")
            
            task_key = args.task.lower()
            model_key = args.model.lower()
            
            if task_key == "full":
                if model_key not in clinical_models_map or model_key not in bci_models_map:
                    parser.error(
                        f"Model '{model_key}' must be available in both clinical and BCI maps for --task full. "
                        f"Clinical: {', '.join(sorted(clinical_models_map.keys()))} | "
                        f"BCI: {', '.join(sorted(bci_models_map.keys()))}"
                    )
                for full_task_key, full_task_cls in tasks_map.items():
                    task_models_map = clinical_models_map if _is_clinical_task_key(full_task_key) else bci_models_map
                    benchmark(
                        [full_task_cls()],
                        [task_models_map[model_key]],
                        args.seed,
                        args.reps,
                        wandb_run=wandb_run,
                        data_percentages=args.data_percentages,
                        linear_probe=args.linear_probe,
                        result_prefix=args.result_prefix,
                        checkpoint_id=args.checkpoint_id,
                        eval_noise_types=args.eval_noise_types,
                        eval_noise_levels_db=args.eval_noise_levels_db,
                        eval_noise_channel_dropout_prob=args.eval_noise_channel_dropout_prob,
                        eval_noise_mix_all=args.eval_noise_mix_all,
                    )
                return
            elif task_key not in tasks_map:
                parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())} or 'full'")
            else:
                tasks_to_run = [tasks_map[task_key]()]

            if _is_clinical_task_key(task_key):
                models_map = clinical_models_map
            elif task_key in ["left_right", "right_feet", "left_right_feet_tongue", "5_fingers"]:
                models_map = bci_models_map
            else:
                models_map = {}
                parser.error(f"Invalid task specified. Choose from: {', '.join(tasks_map.keys())} or 'full'")
            
            if model_key not in models_map:
                parser.error(f"Invalid model specified. Choose from: {', '.join(models_map.keys())}")
            
            model_instance = models_map[model_key]

            benchmark(tasks_to_run, [model_instance], args.seed, args.reps, wandb_run=wandb_run,
                     data_percentages=args.data_percentages, linear_probe=args.linear_probe,
                     result_prefix=args.result_prefix, checkpoint_id=args.checkpoint_id,
                     eval_noise_types=args.eval_noise_types,
                     eval_noise_levels_db=args.eval_noise_levels_db,
                     eval_noise_channel_dropout_prob=args.eval_noise_channel_dropout_prob,
                     eval_noise_mix_all=args.eval_noise_mix_all)
    finally:
        wandb_utils.finish()

if __name__ == "__main__":
    main()
