# Project Overview

EEG-Bench: standardized benchmark for classical and foundation EEG models across clinical (epilepsy, Parkinson's, etc.) and BCI (motor imagery) tasks. This fork adds EEGLeJEPA (Laya model).

## Key directories
- `eeg_bench/models/bci/EEGLeJEPA_model.py` — BCI LeJEPA model (frozen embedding + linear probe path)
- `eeg_bench/models/clinical/EEGLejepa_model.py` — Clinical LeJEPA model (chunking + streaming cache)
- `eeg_bench/config.py` — `LeJEPAConfig` dataclass + config loading
- `benchmark_console.py` — main CLI entry point
- `run_lejepa_sweep.py` — sweep runner with `ExperimentConfig` dataclass
- `sweep_configs/` — YAML sweep configurations
- `eeg_bench/utils/utils.py` — `save_results` function (result file naming)

## External dependency
`eegfmchallenge` package (not in repo) provides `EEGLEJEPAConfig` and `backbone.forward_downstream(x, channel_locations)` returning `{"cls_token": ..., "sequence_embeddings": ...}`. Imported dynamically via `_setup_eegfm_imports`.

## Cache system
Embeddings cached as `.npz` files. Cache key: `{prefix}_{task}_{ckpt_hash}_{dataset_hash}_{split}_{EMBED_CACHE_VERSION}.npz`. Clinical has memmap/sharded variants. `EMBED_CACHE_VERSION` constant controls invalidation.

## Result file naming
`{prefix}_{task}_{ModelClass}_ckpt_{ckpt_id}[_pctXX][_LP][_ATTN|_MLP]_{timestamp}.json`
Resume detection parses this filename pattern via regex in `get_completed_experiments`.
