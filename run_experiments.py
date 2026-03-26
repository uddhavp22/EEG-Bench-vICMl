#!/usr/bin/env python3
"""
Parallel experiment runner for EEG-Bench.
Runs multiple benchmark experiments across multiple GPUs.

Usage:
    # Full run (all models, all tasks, all percentages)
    python run_experiments.py

    # Dry run to see what would execute
    python run_experiments.py --dry-run

    # Resume interrupted run (skip completed)
    python run_experiments.py --resume

    # Custom configuration
    python run_experiments.py --models labram lejepa --tasks left_right parkinsons --percentages 0.1 0.5 1.0

    # Run with specific seeds
    python run_experiments.py --seeds 100 200

    # Adjust parallelism
    python run_experiments.py --gpus 3 --workers-per-gpu 3  # 9 parallel jobs
"""

import argparse
import subprocess
import os
import sys
import time
from itertools import product
from multiprocessing import Pool, Manager
from tqdm import tqdm
from datetime import datetime
import glob
import re

# Configuration
MODELS = ["labram", "bendr", "neurogpt", "reve", "lejepa"]
BCI_TASKS = ["left_right", "right_feet", "left_right_feet_tongue", "5_fingers"]
CLINICAL_TASKS = [
    "parkinsons", "schizophrenia", "mtbi", "ocd", "epilepsy",
    "abnormal", "sleep_stages", "seizure", "binary_artifact", "multiclass_artifact"
]
ALL_TASKS = BCI_TASKS + CLINICAL_TASKS
DEFAULT_PERCENTAGES = [0.25, 0.5, 0.75, 1.0] #0.01, 0.1,
DEFAULT_SEEDS = [100, 200, 300, 400, 500]

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


def get_completed_experiments(results_dir="results/raw"):
    """Check which experiments have already completed based on result files."""
    completed = set()
    if not os.path.exists(results_dir):
        return completed

    # Pattern: task_model_pctXX_LP_timestamp.json
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        filename = os.path.basename(f)
        filename = filename.replace("_clinical", "")
        # Extract model, task, percentage from filename
        match = re.match(r"^(.+)_(\w+Model)(?:_pct(\d+))?(?:_LP)?_\d{8}_\d{6}\.json$", filename)
        if match:
            task_name, model_name, pct = match.groups()
            task_name = normalize_task_name(task_name)
            pct = int(pct) / 100 if pct else 1.0
            completed.add((model_name.lower().replace("model", ""), task_name, pct))
    return completed


def run_experiment(args):
    """Run a single experiment in a subprocess."""
    model, task, pct, seed, log_dir, dry_run, gpu_queue = args
    gpu_id = gpu_queue.get()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "benchmark_console.py",
        "--model", model,
        "--task", task,
        "--data-percentages", str(pct),
        "--seed", str(seed),
        "--linear-probe",
        "--no-wandb"
    ]

    log_file = os.path.join(log_dir, f"{model}_{task}_pct{int(pct*100)}_seed{seed}_gpu{gpu_id}.log")

    start_time = time.time()
    try:
        if dry_run:
            print(f"[DRY RUN] GPU {gpu_id}: {' '.join(cmd)}")
            return (model, task, pct, seed, 0, "dry_run")

        with open(log_file, "w") as f:
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
        return (model, task, pct, seed, result.returncode, status)
    except Exception as e:
        return (model, task, pct, seed, -1, str(e))
    finally:
        gpu_queue.put(gpu_id)


def main():
    parser = argparse.ArgumentParser(
        description="Run EEG-Bench experiments in parallel across multiple GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--gpus", type=int, default=3,
                        help="Number of GPUs available (default: 3)")
    parser.add_argument("--workers-per-gpu", type=int, default=2,
                        help="Number of jobs per GPU (default: 2)")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help=f"Models to run (default: {MODELS})")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        help="Tasks to run (default: all 14 tasks)")
    parser.add_argument("--percentages", nargs="+", type=float, default=DEFAULT_PERCENTAGES,
                        help=f"Data percentages to test (default: {DEFAULT_PERCENTAGES})")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help=f"Random seeds to run (default: {DEFAULT_SEEDS})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed experiments (checks results/raw/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--log-dir", default="logs/experiments",
                        help="Directory for log files (default: logs/experiments)")
    parser.add_argument("--no-linear-probe", action="store_true",
                        help="Run without --linear-probe flag (full fine-tuning)")
    args = parser.parse_args()

    # Setup
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)

    total_workers = args.gpus * args.workers_per_gpu

    print("=" * 60)
    print("EEG-Bench Parallel Experiment Runner")
    print("=" * 60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Tasks: {len(args.tasks)} ({len([t for t in args.tasks if t in BCI_TASKS])} BCI, "
          f"{len([t for t in args.tasks if t in CLINICAL_TASKS])} Clinical)")
    print(f"Percentages: {args.percentages}")
    print(f"Seeds: {args.seeds}")
    print(f"GPUs: {args.gpus}, Workers/GPU: {args.workers_per_gpu}, Total workers: {total_workers}")
    print(f"Linear probe: {'No' if args.no_linear_probe else 'Yes'}")
    print("=" * 60)

    # Generate all experiment combinations
    all_experiments = list(product(args.models, args.tasks, args.percentages, args.seeds))
    print(f"\nTotal experiments: {len(all_experiments)}")

    # Check for completed experiments
    if args.resume:
        completed = get_completed_experiments()
        experiments = [(m, t, p, s) for m, t, p, s in all_experiments if (m, t, p) not in completed]
        print(f"Already completed: {len(all_experiments) - len(experiments)}")
        print(f"Remaining: {len(experiments)}")
    else:
        experiments = all_experiments

    if not experiments:
        print("No experiments to run!")
        return

    # Run larger percentages first to reduce tail time
    experiments = sorted(experiments, key=lambda x: x[2], reverse=True)

    # Split into Phase 1 (cache population) and Phase 2 (remaining seeds).
    # Phase 1 runs one seed per (model, task, pct) to build h5 caches.
    # Phase 2 runs remaining seeds — all caches are warm, safe to parallelize.
    seen_combos = set()
    phase1 = []
    phase2 = []
    for exp in experiments:
        model, task, pct, seed = exp
        combo = (model, task, pct)
        if combo not in seen_combos:
            seen_combos.add(combo)
            phase1.append(exp)
        else:
            phase2.append(exp)

    gpu_slots = [gpu_id for gpu_id in range(args.gpus) for _ in range(args.workers_per_gpu)]
    start_time = time.time()
    results = []

    def run_phase(phase_experiments, phase_name):
        if not phase_experiments:
            return
        jobs = [(model, task, pct, seed, args.log_dir, args.dry_run)
                for model, task, pct, seed in phase_experiments]
        print(f"\n=== {phase_name}: {len(jobs)} experiments with {total_workers} workers ===\n")
        with Manager() as manager:
            gpu_queue = manager.Queue()
            for gpu_id in gpu_slots:
                gpu_queue.put(gpu_id)
            with Pool(total_workers) as pool:
                for result in tqdm(
                    pool.imap_unordered(run_experiment, [job + (gpu_queue,) for job in jobs]),
                    total=len(jobs),
                    desc=phase_name,
                ):
                    results.append(result)

    run_phase(phase1, "Phase 1 (cache population)")
    run_phase(phase2, "Phase 2 (remaining seeds)")

    elapsed = time.time() - start_time

    # Report results
    successes = [r for r in results if r[4] == 0]
    failures = [r for r in results if r[4] != 0]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Successful: {len(successes)}/{len(results)}")

    if failures:
        print(f"\nFailed experiments ({len(failures)}):")
        for model, task, pct, seed, code, status in failures:
            print(f"  - {model}/{task}/pct{int(pct*100)}/seed{seed}: {status} (code {code})")
        print(f"\nCheck logs in {args.log_dir}/ for details.")
    else:
        print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
