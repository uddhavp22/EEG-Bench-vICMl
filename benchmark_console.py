import argparse
import logging
import math
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
from eeg_bench.utils.utils import set_seed, save_results, get_multilabel_tasks, subsample_data_stratified
from eeg_bench.models.clinical.LaBraM.utils_2 import make_multilabels
from eeg_bench.utils import wandb_utils
from eeg_bench.config import load_lejepa_config, merge_lejepa_config_with_cli
from eeg_bench.utils.eeg_noise import format_noise_tag
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

def benchmark(tasks, models, seed, reps=1, wandb_run=None, data_percentages=None, linear_probe=False,
              result_prefix=None, checkpoint_id=None, eval_noise_config=None, probe_type=None, probe_layer=None):
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

        for pct_idx, percentage in enumerate(data_percentages):
            logger.info(f"============================================================")
            logger.info(f"DATA PERCENTAGE: {int(percentage * 100)}%")
            logger.info(f"============================================================")

            # Don't subsample X/y here - pass full data to fit() with percentage parameter
            logger.info(f"Training data: full dataset, will subsample to {int(percentage*100)}% in model")


            noise_types = (eval_noise_config or {}).get("noise_types") or []
            noise_levels = (eval_noise_config or {}).get("levels_db") or []
            include_clean = (eval_noise_config or {}).get("include_clean", True)
            mix_all = (eval_noise_config or {}).get("mix_all", False)

            noise_seed_base = (eval_noise_config or {}).get("seed")
            if noise_seed_base is None:
                noise_seed_base = seed

            if noise_types and noise_levels:
                logger.info(
                    "Evaluation noise sweep enabled: types=%s levels_db=%s include_clean=%s mix_all=%s",
                    noise_types,
                    noise_levels,
                    include_clean,
                    mix_all,
                )
            elif noise_types:
                logger.warning(
                    "Eval noise types provided but no levels_db; falling back to clean evaluation only."
                )

            noise_levels_to_run = noise_levels if noise_types and noise_levels else []

            # Build noise sets: either the provided mix, or each type + the mix.
            if noise_types and noise_levels_to_run:
                noise_sets = [[t] for t in noise_types] if mix_all else [list(noise_types)]
                if mix_all and len(noise_types) > 1:
                    noise_sets.append(list(noise_types))
            else:
                noise_sets = []

            per_noise_collectors = {}

            # Clean condition collector (run once, not per noise set)
            if include_clean:
                clean_tag = format_noise_tag([], math.inf)
                per_noise_collectors[clean_tag] = {
                    "models_names": [],
                    "results": [],
                    "y_trues": [],
                    "y_trains": [],
                    "noise_metadata": {
                        "tag": clean_tag,
                        "snr_db": None,
                        "noise_types": [],
                        "sfreq": (eval_noise_config or {}).get("sfreq"),
                        "channel_dropout_prob": (eval_noise_config or {}).get("channel_dropout_prob", 0.0),
                        "one_over_f_band": (eval_noise_config or {}).get("one_over_f_band"),
                        "emg_band": (eval_noise_config or {}).get("emg_band"),
                        "base_seed": noise_seed_base,
                    },
                }

            # Noise condition collectors
            for noise_set in noise_sets:
                for lvl in noise_levels_to_run:
                    tag = format_noise_tag(noise_set, lvl)
                    if tag not in per_noise_collectors:
                        per_noise_collectors[tag] = {
                            "models_names": [],
                            "results": [],
                            "y_trues": [],
                            "y_trains": [],
                            "noise_metadata": {
                                "tag": tag,
                                "snr_db": float(lvl),
                                "noise_types": list(noise_set),
                                "sfreq": (eval_noise_config or {}).get("sfreq"),
                                "channel_dropout_prob": (eval_noise_config or {}).get("channel_dropout_prob", 0.0),
                                "one_over_f_band": (eval_noise_config or {}).get("one_over_f_band"),
                                "emg_band": (eval_noise_config or {}).get("emg_band"),
                                "base_seed": noise_seed_base,
                            },
                        }

            for model_entry in tqdm(models, desc=f"Models for Task: {task.name} ({int(percentage*100)}%)"):
                # Handle both class types and factory functions
                is_factory = callable(model_entry) and not isinstance(model_entry, type)
                model_name = model_entry.__name__ if hasattr(model_entry, '__name__') else str(model_entry)
                logger.info(f"--- Starting Model: {model_name}")

                for i in range(reps):
                    # Logging for Repetition Clarity
                    logger.info(f"--- REPETITION {i+1}/{reps} (Seed: {seed + i}) ---")

                    set_seed(seed + i)  # set seed for reproducibility

                    if is_multilabel_task:
                        num_classes = len(task.clinical_classes) + 1
                        if is_factory:
                            # Factory function - call with args
                            model = model_entry(num_classes=num_classes, num_labels_per_chunk=task.num_labels_per_chunk)
                        else:
                            # Class - instantiate with args
                            model = model_entry(num_classes=num_classes, num_labels_per_chunk=task.num_labels_per_chunk)
                        this_y_train = make_multilabels(X_train_full, y_train_full, task.event_map, task.chunk_len_s, task.num_labels_per_chunk, model.name)
                        this_y_test = make_multilabels(X_test, y_test, task.event_map, task.chunk_len_s, task.num_labels_per_chunk, model.name)
                    else:
                        if is_factory:
                            # Factory function - call without args
                            model = model_entry()
                        else:
                            # Class - instantiate without args
                            model = model_entry()
                        this_y_train = y_train_full
                        this_y_test = y_test

                    print(model)

                    if hasattr(model, "set_wandb_run"):
                        model.set_wandb_run(wandb_run)
                    model.fit(X_train_full, this_y_train, meta_train, data_percentage=percentage)


                    supports_noise = hasattr(model, "set_eval_noise_config")
                    if noise_types and noise_levels and not supports_noise:
                        logger.warning(
                            "Model %s does not support eval noise injection; using clean eval.",
                            str(model),
                        )

                    def _run_and_collect(tag: str):
                        y_pred_local = []
                        for x, m in zip(X_test, meta_test):
                            y_pred_local.append(model.predict([x], [m]))
                        collector_local = per_noise_collectors[tag]
                        collector_local["models_names"].append(str(model))
                        collector_local["results"].append(y_pred_local)
                        collector_local["y_trues"].append(this_y_test)
                        collector_local["y_trains"].append(this_y_train)

                    # Clean condition once
                    if include_clean:
                        if supports_noise:
                            model.set_eval_noise_config(None)
                        clean_tag = format_noise_tag([], math.inf)
                        _run_and_collect(clean_tag)

                    # Noise conditions: each set, each SNR level
                    if supports_noise and noise_sets and noise_levels_to_run:
                        total_levels = max(1, len(noise_levels_to_run))
                        for set_idx, noise_set in enumerate(noise_sets):
                            for level_idx, level in enumerate(noise_levels_to_run):
                                noise_cfg = dict(eval_noise_config or {})
                                noise_cfg["noise_types"] = list(noise_set)
                                noise_cfg["snr_db"] = float(level)
                                combo_idx = set_idx * total_levels + level_idx
                                noise_cfg["seed"] = int(noise_seed_base) + int(combo_idx)
                                model.set_eval_noise_config(noise_cfg)
                                tag = format_noise_tag(noise_set, level)
                                _run_and_collect(tag)
                    elif not include_clean:
                        # Fallback: if no clean and no supported noise sweep, still evaluate once.
                        if supports_noise:
                            model.set_eval_noise_config(None)
                        fallback_tag = format_noise_tag([], math.inf)
                        if fallback_tag not in per_noise_collectors:
                            per_noise_collectors[fallback_tag] = {
                                "models_names": [],
                                "results": [],
                                "y_trues": [],
                                "y_trains": [],
                                "noise_metadata": {
                                    "tag": fallback_tag,
                                    "snr_db": None,
                                    "noise_types": [],
                                    "sfreq": (eval_noise_config or {}).get("sfreq"),
                                    "channel_dropout_prob": (eval_noise_config or {}).get("channel_dropout_prob", 0.0),
                                    "one_over_f_band": (eval_noise_config or {}).get("one_over_f_band"),
                                    "emg_band": (eval_noise_config or {}).get("emg_band"),
                                    "base_seed": noise_seed_base,
                                },
                            }
                        _run_and_collect(fallback_tag)

            for tag, collector in per_noise_collectors.items():
                tag_prefix = result_prefix
                if tag != "clean":
                    tag_prefix = f"{result_prefix}_{tag}" if result_prefix else tag

                logger.info("Saving/plotting results for noise condition: %s", tag)
                save_results(
                    collector["y_trains"],
                    collector["y_trues"],
                    collector["models_names"],
                    collector["results"],
                    dataset_names,
                    task.name,
                    data_percentage=percentage,
                    linear_probe=linear_probe,
                    probe_type=probe_type,
                    result_prefix=tag_prefix,
                    checkpoint_id=checkpoint_id,
                    seed=seed,
                    eval_noise_metadata=collector.get("noise_metadata"),
                    probe_layer=probe_layer,
                )
                print_classification_results(
                    collector["y_trains"],
                    collector["y_trues"],
                    collector["models_names"],
                    collector["results"],
                    dataset_names,
                    task.name,
                    metrics,
                )
                generate_classification_plots(
                    collector["y_trains"],
                    collector["y_trues"],
                    collector["models_names"],
                    collector["results"],
                    dataset_names,
                    task.name,
                    metrics,
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

    # Evaluation-time EEG noise sweep (robustness)
    parser.add_argument(
        "--eval-noise-types",
        type=str,
        nargs="+",
        default=None,
        choices=["gaussian", "one_over_f", "emg", "channel_dropout"],
        help="Noise types to inject at evaluation time (LeJEPA clinical currently supports this).",
    )

    parser.add_argument(
        "--eval-noise-mix-all",
        action="store_true",
        help="When multiple --eval-noise-types are provided, run each type separately and also the combined mix.",
    )
    parser.add_argument(
        "--eval-noise-levels-db",
        type=float,
        nargs="+",
        default=None,
        help="SNR levels in dB for evaluation sweeps (e.g., 30 20 10 0).",
    )
    parser.add_argument(
        "--eval-noise-sfreq",
        type=float,
        default=None,
        help="Sampling frequency to assume for noise synthesis (defaults to dataset/meta).",
    )
    parser.add_argument(
        "--eval-noise-channel-dropout-prob",
        type=float,
        default=0.0,
        help="Per-channel dropout probability for channel_dropout noise type.",
    )
    parser.add_argument(
        "--eval-noise-one-over-f-band",
        type=float,
        nargs=2,
        default=None,
        metavar=("FMIN", "FMAX"),
        help="Band (Hz) for 1/f noise, e.g., --eval-noise-one-over-f-band 0.5 40.",
    )
    parser.add_argument(
        "--eval-noise-emg-band",
        type=float,
        nargs=2,
        default=None,
        metavar=("FMIN", "FMAX"),
        help="Band (Hz) for EMG-like noise, e.g., --eval-noise-emg-band 30 100.",
    )
    parser.add_argument(
        "--eval-noise-seed",
        type=int,
        default=None,
        help="Base seed for evaluation noise generation (levels offset deterministically).",
    )
    parser.add_argument(
        "--eval-noise-no-clean",
        action="store_true",
        help="Exclude the clean (no-noise) condition when running eval noise sweeps.",
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
    parser.add_argument(
        "--lejepa-attentive-probe",
        action="store_true",
        help="Use an attentive pooling probe head for LeJEPA"
    )
    parser.add_argument(
        "--lejepa-probe-head",
        type=str,
        choices=["linear", "mlp", "attentive"],
        default=None,
        help="LeJEPA classification head type. Defaults to linear."
    )
    parser.add_argument(
        "--lejepa-probe-layer",
        type=float,
        default=None,
        help="Fraction of encoder depth to probe (e.g. 0.25, 0.5, 1.0). None = final layer."
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

    eval_noise_config = None
    if args.eval_noise_types and args.eval_noise_levels_db:
        eval_noise_config = {
            "noise_types": args.eval_noise_types,
            "levels_db": args.eval_noise_levels_db,
            "include_clean": not args.eval_noise_no_clean,
            "sfreq": args.eval_noise_sfreq,
            "channel_dropout_prob": args.eval_noise_channel_dropout_prob,
            "seed": args.eval_noise_seed,
            "mix_all": args.eval_noise_mix_all,
        }
        if args.eval_noise_one_over_f_band:
            eval_noise_config["one_over_f_band"] = tuple(args.eval_noise_one_over_f_band)
        if args.eval_noise_emg_band:
            eval_noise_config["emg_band"] = tuple(args.eval_noise_emg_band)
    elif args.eval_noise_types and not args.eval_noise_levels_db:
        logger.warning("--eval-noise-types provided without --eval-noise-levels-db; ignoring eval noise.")

    # Load and merge LeJEPA configuration
    lejepa_config = merge_lejepa_config_with_cli(
        load_lejepa_config(args.lejepa_config), args
    )

    # Apply --linear-probe to LeJEPA config (model-specific flags already override via merge)
    if args.linear_probe and not getattr(args, 'lejepa_no_freeze_encoder', False):
        lejepa_config.freeze_encoder = True

    probe_type = lejepa_config.probe_head
    probe_layer = lejepa_config.probe_layer

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
                              freeze_encoder=args.linear_probe)

    def make_labram_bci():
        return LaBraMBci(freeze_encoder=args.linear_probe)

    def make_bendr_clinical(num_classes=2, num_labels_per_chunk=None):
        return BENDRClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            freeze_encoder=args.linear_probe
        )

    def make_bendr_bci():
        return BENDRBci(freeze_encoder=args.linear_probe)

    def make_neurogpt_clinical(num_classes=2, num_labels_per_chunk=None):
        return NeuroGPTClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            freeze_encoder=args.linear_probe
        )

    def make_neurogpt_bci():
        return NeuroGPTBci(freeze_encoder=args.linear_probe)

    def make_reve_clinical(num_classes=2, num_labels_per_chunk=None):
        return REVEClinical(
            num_classes=num_classes,
            num_labels_per_chunk=num_labels_per_chunk,
            freeze_backbone=args.linear_probe
        )

    def make_reve_bci():
        return REVEBci(freeze_backbone=args.linear_probe)

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
    }
    bci_models_map = {
        "lda": CSPLDA,
        "svm": CSPSVM,
        "labram": make_labram_bci,
        "bendr": make_bendr_bci,
        "neurogpt": make_neurogpt_bci,
        "reve": make_reve_bci,
        "lejepa": make_lejepa_bci
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
                "lejepa_probe_head": lejepa_config.probe_head,
            },
        )
        wandb_utils.set_run(wandb_run)
    # ---------------------------------------------------------

    try:
        if args.all:
            logger.info("Running all task/model combinations...")
            for task_key, task_cls in tasks_map.items():
                if task_key in ["parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy", "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"]:
                    models_map = clinical_models_map
                else:
                    models_map = bci_models_map

                task_instance = task_cls()
                model_classes = list(models_map.values())
                benchmark([task_instance], model_classes, args.seed, args.reps, wandb_run=wandb_run,
                         data_percentages=args.data_percentages, linear_probe=args.linear_probe,
                         result_prefix=args.result_prefix, checkpoint_id=args.checkpoint_id,
                         eval_noise_config=eval_noise_config, probe_type=probe_type, probe_layer=probe_layer)

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
            
            model_instance = models_map[model_key]

            benchmark(tasks_to_run, [model_instance], args.seed, args.reps, wandb_run=wandb_run,
                     data_percentages=args.data_percentages, linear_probe=args.linear_probe,
                     result_prefix=args.result_prefix, checkpoint_id=args.checkpoint_id,
                     eval_noise_config=eval_noise_config, probe_type=probe_type, probe_layer=probe_layer)
    finally:
        wandb_utils.finish()

if __name__ == "__main__":
    main()
