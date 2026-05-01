# Suggested Commands

## Environment
```bash
conda activate eeg_bench
```

## Running benchmarks
```bash
python benchmark_console.py --model lejepa --task left_right --no-wandb
python benchmark_console.py --model lejepa --task left_right --linear-probe --lejepa-freeze-encoder --no-wandb
python run_lejepa_sweep.py --config sweep_configs/full_benchmark.yaml
```

## Lint / format / test
No dedicated test runner or linter config found in the repo. Use `python -m pytest` if tests are added.

## Key entry points
- `benchmark_console.py` — single run
- `run_lejepa_sweep.py` — sweep runner (reads YAML, expands matrix, calls benchmark_console.py)
