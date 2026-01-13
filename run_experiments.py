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

    # Skip cache pre-warming (if already cached)
    python run_experiments.py --skip-prewarm

    # Adjust parallelism
    python run_experiments.py --gpus 3 --workers-per-gpu 3  # 9 parallel jobs
"""

import argparse
import subprocess
import os
import sys
import time
from itertools import product
from multiprocessing import Pool
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
DEFAULT_PERCENTAGES = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]


def get_completed_experiments(results_dir="results/raw"):
    """Check which experiments have already completed based on result files."""
    completed = set()
    if not os.path.exists(results_dir):
        return completed

    # Pattern: task_model_pctXX_LP_timestamp.json
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        filename = os.path.basename(f)
        # Extract model, task, percentage from filename
        # This is a simplified pattern - may need adjustment based on actual filenames
        match = re.match(r"(.+?)_(\w+Model)(?:_pct(\d+))?(?:_LP)?_\d+\.json", filename)
        if match:
            task_name, model_name, pct = match.groups()
            pct = int(pct) / 100 if pct else 1.0
            completed.add((model_name.lower().replace("model", ""), task_name, pct))

    return completed


def run_experiment(args):
    """Run a single experiment in a subprocess."""
    model, task, pct, gpu_id, log_dir, dry_run = args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable, "benchmark_console.py",
        "--model", model,
        "--task", task,
        "--data-percentages", str(pct),
        "--linear-probe",
        "--no-wandb"
    ]

    log_file = os.path.join(log_dir, f"{model}_{task}_pct{int(pct*100)}_gpu{gpu_id}.log")

    if dry_run:
        print(f"[DRY RUN] GPU {gpu_id}: {' '.join(cmd)}")
        return (model, task, pct, 0, "dry_run")

    start_time = time.time()
    try:
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
        return (model, task, pct, result.returncode, status)

    except Exception as e:
        return (model, task, pct, -1, str(e))


def prewarm_cache(tasks, models, log_dir):
    """
    Pre-warm the cache by running one experiment per task sequentially.
    This ensures all datasets are cached before parallel execution.
    """
    print("\n=== Pre-warming cache ===")
    print("Running one experiment per task to populate cache...")

    # Run smallest percentage for each task with first model
    model = models[0]
    pct = 0.01  # Smallest percentage = fastest

    for i, task in enumerate(tasks):
        print(f"  [{i+1}/{len(tasks)}] Caching {task}...")
        result = run_experiment((model, task, pct, 0, log_dir, False))
        if result[3] != 0:
            print(f"    Warning: Cache warm-up failed for {task}")

    print("Cache pre-warming complete.\n")


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
    parser.add_argument("--skip-prewarm", action="store_true",
                        help="Skip cache pre-warming phase")
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
    print(f"GPUs: {args.gpus}, Workers/GPU: {args.workers_per_gpu}, Total workers: {total_workers}")
    print(f"Linear probe: {'No' if args.no_linear_probe else 'Yes'}")
    print("=" * 60)

    # Generate all experiment combinations
    all_experiments = list(product(args.models, args.tasks, args.percentages))
    print(f"\nTotal experiments: {len(all_experiments)}")

    # Check for completed experiments
    if args.resume:
        completed = get_completed_experiments()
        experiments = [(m, t, p) for m, t, p in all_experiments if (m, t, p) not in completed]
        print(f"Already completed: {len(all_experiments) - len(experiments)}")
        print(f"Remaining: {len(experiments)}")
    else:
        experiments = all_experiments

    if not experiments:
        print("No experiments to run!")
        return

    # Pre-warm cache
    if not args.skip_prewarm and not args.dry_run:
        prewarm_cache(args.tasks, args.models, args.log_dir)

    # Assign GPUs round-robin
    jobs = []
    for i, (model, task, pct) in enumerate(experiments):
        gpu_id = i % args.gpus
        jobs.append((model, task, pct, gpu_id, args.log_dir, args.dry_run))

    # Run experiments in parallel
    print(f"\n=== Running {len(jobs)} experiments with {total_workers} workers ===\n")
    start_time = time.time()

    with Pool(total_workers) as pool:
        results = pool.map(run_experiment, jobs)

    elapsed = time.time() - start_time

    # Report results
    successes = [r for r in results if r[3] == 0]
    failures = [r for r in results if r[3] != 0]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Successful: {len(successes)}/{len(results)}")

    if failures:
        print(f"\nFailed experiments ({len(failures)}):")
        for model, task, pct, code, status in failures:
            print(f"  - {model}/{task}/pct{int(pct*100)}: {status} (code {code})")
        print(f"\nCheck logs in {args.log_dir}/ for details.")
    else:
        print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
