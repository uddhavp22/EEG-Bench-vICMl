# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG-Bench is a standardized benchmark for evaluating classical and foundation models across clinical and BCI (Brain-Computer Interface) EEG decoding tasks. It supports 25 datasets including clinical classification tasks (epilepsy, Parkinson's, schizophrenia) and motor imagery paradigms (left/right hand, 5-finger decoding).

**This is a fork** of the original EEG-Bench repository. The primary contribution in this fork is **EEGLeJEPA**, a new model being integrated into the benchmark suite.

## Development Workflow

- **Do not modify** anything in `datasets/` - these are stable from upstream
- **Feature branches**: Create PRs to `sp-branch` for each new feature
- **Current focus**: Setting up EEGLeJEPA model integration properly
- The original benchmark code needs cleanup, but that's secondary to getting EEGLeJEPA working

## Commands

### Environment Setup
```bash
conda env create -f environment.yml
conda activate eeg_bench
```

### Running Benchmarks
```bash
# Run a specific task with a specific model
python benchmark_console.py --model labram --task left_right

# Run all tasks with all models
python benchmark_console.py --all

# Run with multiple repetitions for statistical assessment
python benchmark_console.py --model labram --task epilepsy --reps 5

# Disable WandB logging
python benchmark_console.py --model labram --task left_right --no-wandb
```

### Task Codes
Clinical: `parkinsons`, `schizophrenia`, `mtbi`, `ocd`, `epilepsy`, `abnormal`, `sleep_stages`, `seizure`, `binary_artifact`, `multiclass_artifact`

BCI (Motor Imagery): `left_right`, `right_feet`, `left_right_feet_tongue`, `5_fingers`

### Model Codes
`lda`, `svm`, `labram`, `bendr`, `neurogpt`, `reve`, `lejepa`

## Architecture

### Core Abstractions

**Tasks** (`eeg_bench/tasks/`): Central organizing principle. Each task defines datasets, train/test subject splits, target classes, and evaluation metrics.
- `AbstractBCITask` for motor imagery tasks
- `AbstractClinicalTask` for clinical diagnosis tasks
- Tasks return data via `get_data(Split.TRAIN)` or `get_data(Split.TEST)`

**Models** (`eeg_bench/models/`): All models implement `AbstractModel` with `fit()` and `predict()` methods.
- BCI models: CSP-based (LDA/SVM), LaBraM, BENDR, NeuroGPT, REVE, LeJEPA
- Clinical models: Brainfeatures (LDA/SVM), LaBraM, BENDR, NeuroGPT, REVE, LeJEPA
- Models are split between `bci/` and `clinical/` directories with separate implementations

**Datasets** (`eeg_bench/datasets/`): Dataset loaders inheriting from `AbstractDataset`.
- Must implement `load_data()` and `_download()` methods
- Data format: `(n_samples, n_channels, n_timepoints)` for BCI; `List[BaseRaw]` for clinical
- Meta dict must include `sampling_frequency`, `channel_names`, and `name`

### Data Flow
1. Task instantiated with predefined subject splits
2. Task's `get_data()` loads datasets for the specified split
3. Model's `fit()` receives `List[np.ndarray]` (one per dataset), labels, and metadata
4. Model's `predict()` returns predictions on test data

### Configuration
- `eeg_bench/config.json`: Paths for data, cache, checkpoints, logs, results
- Dataset-specific paths can be set to `null` to use defaults

### Multi-label Tasks
For multi-label tasks (artifacts, sleep stages), add task name to `get_multilabel_tasks()` in `eeg_bench/utils/utils.py` and handle channel requirements in `eeg_bench/models/clinical/brainfeatures/feature_extraction_2.py`.

## Known Issues

If you encounter `ModuleNotFoundError: No module named 'torch._six'`, delete line 18 in `deepspeed/runtime/utils.py` and line 9 in `deepspeed/runtime/zero/stage2.py` in the conda environment.
