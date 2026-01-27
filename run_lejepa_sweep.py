#!/usr/bin/env python3
"""
LeJEPA Checkpoint Sweep Runner.

Runs LeJEPA experiments across multiple model sizes and training checkpoints
using a YAML configuration file.

Usage:
    # Run with config file
    python run_lejepa_sweep.py --config sweep_config.yaml

    # Dry run to see what would execute
    python run_lejepa_sweep.py --config sweep_config.yaml --dry-run

    # Resume interrupted run (skip completed)
    python run_lejepa_sweep.py --config sweep_config.yaml --resume

Example YAML config:
    models:
      lejepa_small:
        base_path: /path/to/lejepa_small
        version: 0
      lejepa_base:
        base_path: /path/to/lejepa_base
        version: 0

    checkpoints:
      # Steps to evaluate (epoch is always 0)
      steps: [1000, 5000, 10000]
      include_last: true

    tasks:
      epoch_sweep: [left_right, parkinsons]
      final_only: [all]

    training:
      linear_probe: true
      data_percentages: [1.0]

    execution:
      gpus: 3
      workers_per_gpu: 2  # Back to 2 for parallelism
      log_dir: logs/lejepa_sweep
"""

import argparse
import subprocess
import os
import sys
import time
import re
import glob
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Pool
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import yaml
from tqdm import tqdm

# Task definitions
BCI_TASKS = ["left_right", "right_feet", "left_right_feet_tongue", "5_fingers"]
CLINICAL_TASKS = [
    "parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy",
    "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"
]

TASK_NAME_MAP = {
    "Left Hand vs Right Hand MI": "left_right",
    "Right Hand vs Feet MI": "right_feet",
    "Left Hand vs Right Hand vs Feet vs Tongue MI": "left_right_feet_tongue",
    "Five Fingers MI": "5_fingers",
}

def normalize_task_name(task_name):
    if task_name in TASK_NAME_MAP:
        return TASK_NAME_MAP[task_name]
    if task_name.endswith("_clinical"):
        return task_name.replace("_clinical", "")
    return task_name



EMBED_CACHE_VERSION = os.getenv("EEG_BENCH_EMBED_CACHE_VERSION", "v2")
ALL_TASKS = CLINICAL_TASKS + BCI_TASKS


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model_name: str
    base_path: str
    checkpoint_path: str
    checkpoint_id: str
    task: str
    percentage: float
    linear_probe: bool
    eval_noise_config: Optional[Dict[str, Any]] = None

    def to_cmd_args(self) -> List[str]:
        """Convert to benchmark_console.py CLI arguments."""
        args = [
            "--model", "lejepa",
            "--task", self.task,
            "--lejepa-checkpoint-full-path", self.checkpoint_path,
            "--data-percentage", str(self.percentage),  # Changed from --data-percentages
            "--no-wandb",
            "--result-prefix", self.model_name,
            "--checkpoint-id", self.checkpoint_id,
        ]
        if self.linear_probe:
            args.extend(["--linear-probe", "--lejepa-freeze-encoder"])
        else:
            args.append("--lejepa-no-freeze-encoder")

        noise_cfg = self.eval_noise_config or {}
        noise_types = noise_cfg.get("noise_types") or []
        noise_levels = noise_cfg.get("levels_db") or []
        if noise_types and noise_levels:
            args.extend(["--eval-noise-types", *map(str, noise_types)])
            args.extend(["--eval-noise-levels-db", *map(str, noise_levels)])
            if noise_cfg.get("sfreq") is not None:
                args.extend(["--eval-noise-sfreq", str(noise_cfg["sfreq"])])
            if noise_cfg.get("channel_dropout_prob") is not None:
                args.extend(["--eval-noise-channel-dropout-prob", str(noise_cfg["channel_dropout_prob"])])
            if noise_cfg.get("one_over_f_band"):
                fmin, fmax = noise_cfg["one_over_f_band"]
                args.extend(["--eval-noise-one-over-f-band", str(fmin), str(fmax)])
            if noise_cfg.get("emg_band"):
                fmin, fmax = noise_cfg["emg_band"]
                args.extend(["--eval-noise-emg-band", str(fmin), str(fmax)])
            if noise_cfg.get("seed") is not None:
                args.extend(["--eval-noise-seed", str(noise_cfg["seed"])])
            if noise_cfg.get("include_clean") is False:
                args.append("--eval-noise-no-clean")
        return args


def discover_checkpoints(base_path: str, version: int = 0) -> Dict[str, str]:
    """
    Discover all checkpoint files in a base_path/version_N/checkpoints/ directory.

    Returns:
        Dict mapping checkpoint identifier to full checkpoint path.
        Keys: "last", "epoch_0_step_1000", etc.
    """
    checkpoints = {}
    ckpt_dir = Path(base_path) / f"version_{version}" / "checkpoints"

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Find last.ckpt
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        checkpoints["last"] = str(last_ckpt)

    # Find epoch checkpoints: epoch=N-step=M.ckpt (PyTorch Lightning format)
    epoch_pattern = re.compile(r"epoch=(\d+)-step=(\d+)\.ckpt")
    for ckpt_file in ckpt_dir.glob("epoch=*-step=*.ckpt"):
        match = epoch_pattern.match(ckpt_file.name)
        if match:
            epoch_num = int(match.group(1))
            step_num = int(match.group(2))
            checkpoints[f"epoch_{epoch_num}_step_{step_num}"] = str(ckpt_file)

    return checkpoints


def filter_checkpoints(
    checkpoints: Dict[str, str],
    steps: Optional[List[int]],
    include_last: bool,
    auto_discover: bool = False
) -> Dict[str, str]:
    """Filter checkpoints based on configuration."""
    filtered = {}

    if auto_discover:
        # Use all discovered checkpoints
        filtered = checkpoints.copy()
        if not include_last and "last" in filtered:
            del filtered["last"]
    else:
        if steps:
            step_set = set(steps)
            found_steps = set()
            for ckpt_id, ckpt_path in checkpoints.items():
                if ckpt_id == "last":
                    continue
                if "_step_" not in ckpt_id or not ckpt_id.startswith("epoch_"):
                    continue
                step_str = ckpt_id.split("_step_", 1)[1]
                try:
                    step = int(step_str)
                except ValueError:
                    continue
                if step in step_set:
                    filtered[ckpt_id] = ckpt_path
                    found_steps.add(step)
            missing_steps = [s for s in steps if s not in found_steps]
            for step in missing_steps:
                print(f"  [Warning] Step {step} not found in available checkpoints")

        if include_last and "last" in checkpoints:
            filtered["last"] = checkpoints["last"]

    return filtered


def expand_task_list(tasks: List[str]) -> List[str]:
    """Expand 'all', 'bci', 'clinical' to actual task lists."""
    expanded = []
    for t in tasks:
        t_lower = t.lower()
        if t_lower == "all":
            expanded.extend(ALL_TASKS)
        elif t_lower == "bci":
            expanded.extend(BCI_TASKS)
        elif t_lower == "clinical":
            expanded.extend(CLINICAL_TASKS)
        else:
            if t in ALL_TASKS:
                expanded.append(t)
            else:
                print(f"  [Warning] Unknown task: {t}")
    # Remove duplicates while preserving order
    return list(dict.fromkeys(expanded))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set defaults
    config.setdefault("checkpoints", {})
    config["checkpoints"].setdefault("steps", [])
    config["checkpoints"].setdefault("include_last", True)
    config["checkpoints"].setdefault("auto_discover", False)

    config.setdefault("tasks", {})
    config["tasks"].setdefault("epoch_sweep", [])
    config["tasks"].setdefault("final_only", ["all"])

    config.setdefault("training", {})
    config["training"].setdefault("linear_probe", True)
    config["training"].setdefault("data_percentages", [1.0])

    config.setdefault("eval_noise", {})

    config.setdefault("execution", {})
    config["execution"].setdefault("gpus", 3)
    config["execution"].setdefault("workers_per_gpu", 2)  # Back to 2 for parallelism
    config["execution"].setdefault("log_dir", "logs/lejepa_sweep")

    return config


def generate_experiments(config: Dict[str, Any]) -> List[ExperimentConfig]:
    """Generate experiment matrix from configuration, sorted for cache efficiency."""
    experiments = []

    # Expand task lists
    epoch_sweep_tasks = expand_task_list(config["tasks"]["epoch_sweep"])
    final_only_tasks = expand_task_list(config["tasks"]["final_only"])

    # Remove epoch_sweep_tasks from final_only to avoid duplicates
    final_only_tasks = [t for t in final_only_tasks if t not in epoch_sweep_tasks]

    print(f"\nTask configuration:")
    print(f"  Epoch sweep tasks ({len(epoch_sweep_tasks)}): {epoch_sweep_tasks}")
    print(f"  Final only tasks ({len(final_only_tasks)}): {final_only_tasks}")

    for model_name, model_config in config["models"].items():
        base_path = model_config["base_path"]
        version = model_config.get("version", 0)

        print(f"\nDiscovering checkpoints for {model_name}...")
        print(f"  Base path: {base_path}")
        print(f"  Version: {version}")

        try:
            all_checkpoints = discover_checkpoints(base_path, version)
            print(f"  Found {len(all_checkpoints)} checkpoints: {list(all_checkpoints.keys())}")
        except FileNotFoundError as e:
            print(f"  [Error] {e}")
            continue

        # Get checkpoints for epoch sweep
        epoch_checkpoints = filter_checkpoints(
            all_checkpoints,
            steps=config["checkpoints"]["steps"],
            include_last=config["checkpoints"]["include_last"],
            auto_discover=config["checkpoints"]["auto_discover"]
        )
        print(f"  Using for epoch sweep: {list(epoch_checkpoints.keys())}")

        # Get last checkpoint only for final_only tasks
        last_only = {"last": all_checkpoints["last"]} if "last" in all_checkpoints else {}

        # Generate experiments for epoch_sweep tasks (all checkpoints)
        for task in epoch_sweep_tasks:
            for ckpt_id, ckpt_path in epoch_checkpoints.items():
                for pct in config["training"]["data_percentages"]:
                    experiments.append(ExperimentConfig(
                        model_name=model_name,
                        base_path=base_path,
                        checkpoint_path=ckpt_path,
                        checkpoint_id=ckpt_id,
                        task=task,
                        percentage=pct,
                        linear_probe=config["training"]["linear_probe"],
                        eval_noise_config=config.get("eval_noise"),
                    ))

        # Generate experiments for final_only tasks (last.ckpt only)
        for task in final_only_tasks:
            for ckpt_id, ckpt_path in last_only.items():
                for pct in config["training"]["data_percentages"]:
                    experiments.append(ExperimentConfig(
                        model_name=model_name,
                        base_path=base_path,
                        checkpoint_path=ckpt_path,
                        checkpoint_id=ckpt_id,
                        task=task,
                        percentage=pct,
                        linear_probe=config["training"]["linear_probe"],
                        eval_noise_config=config.get("eval_noise"),
                    ))

    # Sort experiments to maximize cache hits:
    # 1. Group by checkpoint (same embeddings)
    # 2. Then by task (same data)
    # 3. Then by percentage (descending - 100% first to populate cache)
    experiments.sort(key=lambda e: (e.checkpoint_path, e.task, -e.percentage))
    
    return experiments


def get_completed_experiments(results_dir: str = "results/raw") -> set:
    """
    Check which experiments have already completed based on result files.

    Returns set of (model_name, task, checkpoint_id, percentage) tuples.
    """
    completed = set()
    if not os.path.exists(results_dir):
        return completed

    # Pattern: {model_name}_{task}_{ModelClass}_ckpt_{checkpoint_id}[_pctXX][_LP]_{timestamp}.json
    # Examples:
    # - lejepa_base_global_proj_abnormal_clinical_LeJEPAClinical_ckpt_last_LP_20260124_122416.json
    # - lejepa_base_global_proj_Left Hand vs Right Hand vs Feet vs Tongue MI_LeJEPABCI_ckpt_last_LP_20260124_192457.json
    
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        filename = os.path.basename(f)
        
        # Match pattern: *_LeJEPA{Type}_ckpt_{checkpoint_id}[_pctXX][_LP]_{timestamp}.json
        match = re.match(
            r"(.+?)_(LeJEPA(?:Clinical|BCI))_ckpt_([^_]+(?:_step_\d+)?)(?:_pct(\d+))?(?:_LP)?_(\d{8}_\d{6})\.json",
            filename
        )
        if match:
            prefix, model_class, ckpt_id, pct_str, timestamp = match.groups()
            
            # Split prefix into model_name and task
            # Try to find a known task at the end of prefix
            model_name = None
            task = None
            
            # Try each known task (including space-separated ones)
            for known_task in ALL_TASKS:
                # Build possible task patterns to match in filename
                possible_patterns = [
                    f"{known_task}_clinical",
                    f"{known_task}_bci",
                    known_task
                ]
                
                # Also check for the reverse mapping (e.g., "Left Hand vs Right Hand vs Feet vs Tongue MI")
                for full_name, short_name in TASK_NAME_MAP.items():
                    if short_name == known_task:
                        possible_patterns.insert(0, full_name)
                
                for pattern in possible_patterns:
                    if prefix.endswith("_" + pattern):
                        model_name = prefix[:-(len(pattern) + 1)]
                        task = normalize_task_name(pattern)
                        break
                    elif prefix == pattern:
                        model_name = ""
                        task = normalize_task_name(pattern)
                        break
                
                if task:
                    break
            
            if model_name is not None and task:
                pct = int(pct_str) / 100 if pct_str else 1.0
                completed.add((model_name, task, ckpt_id, pct))

    return completed


def run_experiment(args: Tuple) -> Tuple:
    """Run a single experiment in a subprocess."""
    experiment, gpu_id, log_dir, dry_run = args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["EEG_BENCH_EMBED_CACHE_VERSION"] = EMBED_CACHE_VERSION

    cmd = [sys.executable, "benchmark_console.py"] + experiment.to_cmd_args()

    log_name = (f"{experiment.model_name}_{experiment.task}_"
                f"{experiment.checkpoint_id}_pct{int(experiment.percentage*100)}_gpu{gpu_id}.log")
    log_file = os.path.join(log_dir, log_name)

    if dry_run:
        # print(f"[DRY RUN] GPU {gpu_id}: {experiment.model_name}/{experiment.task}/{experiment.checkpoint_id}")
        # print(f"          {' '.join(cmd)}")
        return (experiment, 0, "dry_run")

    start_time = time.time()
    try:
        with open(log_file, "w") as f:
            f.write(f"Model: {experiment.model_name}\n")
            f.write(f"Checkpoint: {experiment.checkpoint_path}\n")
            f.write(f"Checkpoint ID: {experiment.checkpoint_id}\n")
            f.write(f"Task: {experiment.task}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"GPU: {gpu_id}\n")
            f.write("-" * 50 + "\n")
            f.flush()

            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )

            elapsed = time.time() - start_time
            f.write("-" * 50 + "\n")
            f.write(f"Finished: {datetime.now().isoformat()}\n")
            f.write(f"Elapsed: {elapsed:.1f}s\n")
            f.write(f"Return code: {result.returncode}\n")

        status = "success" if result.returncode == 0 else "failed"
        return (experiment, result.returncode, status)

    except Exception as e:
        return (experiment, -1, str(e))


def prewarm_embedding_cache(experiments: List[ExperimentConfig], gpus: int, dry_run: bool):
    """
    Pre-extract embeddings for all unique (checkpoint, task) combinations.
    This ensures the 100% embeddings are cached before running percentage sweeps.
    
    Note: This is informational - actual caching happens on first run of each
    (checkpoint, task) combination. The sorting in generate_experiments() ensures
    100% runs happen first.
    """
    # Find unique (checkpoint, task) combinations
    unique_combos = set((e.checkpoint_path, e.task) for e in experiments)
    
    # Count how many experiments will benefit from cache reuse
    cache_reuse_count = len(experiments) - len(unique_combos)
    
    print(f"\n=== Embedding Cache Strategy ===")
    print(f"Unique (checkpoint, task) combinations: {len(unique_combos)}")
    print(f"Experiments that will reuse cached embeddings: {cache_reuse_count}")
    print(f"Experiments sorted to run 100% first for each combination.")
    
    if dry_run:
        print(f"[DRY RUN] First run of each combination will populate cache.")
        return
    
    # The actual caching happens automatically during the first experiment
    # for each (checkpoint, task) combination. The sorting ensures 100% runs first.


def main():
    parser = argparse.ArgumentParser(
        description="Run LeJEPA checkpoint sweep with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed experiments")
    parser.add_argument("--skip-prewarm", action="store_true",
                        help="Skip cache pre-warming phase")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Setup directories
    log_dir = config["execution"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)

    gpus = config["execution"]["gpus"]
    workers_per_gpu = config["execution"]["workers_per_gpu"]
    total_workers = gpus * workers_per_gpu

    # Print configuration summary
    print("\n" + "=" * 60)
    print("LeJEPA Checkpoint Sweep")
    print("=" * 60)
    print(f"Models: {list(config['models'].keys())}")
    print(f"Checkpoints: steps={config['checkpoints']['steps']}, "
          f"include_last={config['checkpoints']['include_last']}")
    print(f"Linear probe: {config['training']['linear_probe']}")
    print(f"Data percentages: {config['training']['data_percentages']}")
    print(f"GPUs: {gpus}, Workers/GPU: {workers_per_gpu}, Total workers: {total_workers}")
    print("=" * 60)

    # Generate experiment matrix
    experiments = generate_experiments(config)
    print(f"\nTotal experiments: {len(experiments)}")

    # Filter completed experiments if resuming
    if args.resume:
        completed = get_completed_experiments()
        original_count = len(experiments)
        experiments = [
            e for e in experiments
            if (e.model_name, e.task, e.checkpoint_id, e.percentage) not in completed
        ]
        print(f"Already completed: {original_count - len(experiments)}")
        print(f"Remaining: {len(experiments)}")

    if not experiments:
        print("No experiments to run!")
        return

    # Add prewarm phase before main experiments
    if not args.skip_prewarm and config["training"]["data_percentages"] != [1.0]:
        prewarm_embedding_cache(experiments, gpus, args.dry_run)
    
    # Assign GPUs round-robin, but assign to same GPU sequentially per (checkpoint, task)
    # This ensures 100% runs before 75% on same GPU
    jobs = []
    gpu_assignment = {}  # Track (checkpoint, task) -> gpu_id for sequential execution
    
    for exp in experiments:
        key = (exp.checkpoint_path, exp.task)
        
        if key not in gpu_assignment:
            # First time seeing this (checkpoint, task): assign to next GPU
            gpu_id = len(gpu_assignment) % gpus
            gpu_assignment[key] = gpu_id
        else:
            # Reuse same GPU for this (checkpoint, task) pair
            gpu_id = gpu_assignment[key]
        
        jobs.append((exp, gpu_id, log_dir, args.dry_run))
    
    # Run experiments in parallel
    print(f"\n=== Running {len(jobs)} experiments with {total_workers} workers ===\n")
    start_time = time.time()

    with Pool(total_workers) as pool:
        results = list(tqdm(
            pool.imap(run_experiment, jobs),
            total=len(jobs),
            desc="Running experiments",
            unit="exp"
        ))

    elapsed = time.time() - start_time

    # Report results
    successes = [r for r in results if r[1] == 0]
    failures = [r for r in results if r[1] != 0]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Successful: {len(successes)}/{len(results)}")

    if failures:
        print(f"\nFailed experiments ({len(failures)}):")
        for exp, code, status in failures:
            print(f"  - {exp.model_name}/{exp.task}/{exp.checkpoint_id}: {status} (code {code})")
        print(f"\nCheck logs in {log_dir}/ for details.")
    else:
        print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
