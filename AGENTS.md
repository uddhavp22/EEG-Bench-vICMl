# Repository Guidelines

## Project Structure & Module Organization
- `eeg_bench/` is the core package. Key subfolders:
  - `eeg_bench/datasets/` dataset loaders for clinical and BCI paradigms.
  - `eeg_bench/tasks/` task definitions and train/test splits.
  - `eeg_bench/models/` model implementations (CSP, BENDR, LaBraM, NeuroGPT, etc.).
  - `eeg_bench/utils/` shared helpers and evaluation utilities.
- `benchmark_console.py` is the CLI entry point for running experiments.
- `environment.yml` defines the conda environment for reproducibility.
- `REVE_posbank/` and `REVE_Tutorial_EEGMAT.ipynb` hold REVE assets/tutorials.

## Local Additions (EEGLeJEPA)
- EEGLeJEPA-specific additions are the primary local work in this repo.
- Model entry points live at `eeg_bench/models/clinical/EEGLejepa_model.py` and `eeg_bench/models/bci/EEGLeJEPA_model.py`.
- CLI wiring uses the `--model lejepa` key in `benchmark_console.py`.

## Build, Test, and Development Commands
- `conda env create -f environment.yml` sets up the full environment.
- `conda activate eeg_bench` activates the environment.
- `python benchmark_console.py --model labram --task lr` runs a single model/task.
- `python benchmark_console.py --model lejepa --task lr` runs EEGLeJEPA on a single task.
- `python benchmark_console.py --task mtbi --model lejepa --lejepa-ckpt /path/to/ckpt.ckpt --data-frac 1 0.5` runs a checkpoint sweep with data-efficiency fractions.
- `python benchmark_console.py --task mtbi --model lejepa --lejepa-ckpt-dir /path/to/ckpts --data-frac 1` evaluates all checkpoints in a folder.
- `python benchmark_console.py --all --reps 5` runs all models/tasks with repetitions.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and PEP 8-style naming.
- Use `snake_case` for functions/variables and `CamelCase` for classes.
- Keep implementations minimal and avoid redundant checks when inputs are already validated.

## Testing Guidelines
- There is no formal test suite defined; prefer running a small benchmark as a smoke test.
- For new tasks/models, run a single task with the new component before full sweeps.

## Configuration & Data Paths
- Configure dataset, cache, and checkpoint locations in `eeg_bench/config.json` (see `eeg_bench/config.py`).
- Ensure MNE download paths match your local storage layout.
 - Sweep metrics are appended to `results/sweep_results.csv`.

## Commit & Pull Request Guidelines
- Recent history uses short, lowercase, sentence-style commit subjects (e.g., “updates to running eeglejepa”). Follow that style.
- PRs should include:
  - A brief description of tasks/models affected.
  - Any new datasets or splits added.
  - Example command used to validate the change.
  - Notes about required data/config changes.
