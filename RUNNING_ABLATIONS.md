# Running rebuttal ablations (pretrain + eval)

Concise reference for the two-repo pipeline used for the NeurIPS rebuttal
ablations (Laya-EMA, Laya-MeanPool, etc.): pretrain in `eegfmchallenge` via
`baircondor`, evaluate here via `run_lejepa_sweep.py`. See
`eegfmchallenge/rebuttal/<name>/PLAN.md` for the write-up of each specific
ablation; this doc is just the generic "how do I run one" reference.

## 1. Pretrain (in `eegfmchallenge`)

Each ablation is a config file in `eegfmchallenge/configs/`, already written
with a docstring explaining the hypothesis/prediction/falsification
criterion (see any `configs/guess_*.py` or `configs/*_10pct_10k.py` for the
pattern) — read that before launching, don't just run it blind.

**Preflight — run this exact block before every submit, no need to
re-derive it or grep past sessions for the command:**

```bash
cd /REDLRADADM35839/home/spanchavati/eegfmchallenge
git status --short          # if dirty: commit what belongs to this run, or
                              # confirm the dirty file is unrelated (e.g. an
                              # analysis script) and safe to leave as-is
git pull --rebase            # another machine/session may have pushed
git push                     # REQUIRED if you have local commits — baircondor
                              # runs off this repo dir, which the target
                              # machine reads live over NFS; un-pushed local
                              # commits ARE visible (same filesystem), but
                              # push anyway so other sessions stay in sync.

# GPU headroom on the target machine. No SSH access to compute nodes — this
# is the only way to check. State=Unclaimed/Activity=Idle == free.
condor_status -constraint 'Machine=="REDLRADADM35840.ad.medctr.ucla.edu"' \
  -af Name State Activity GPUs_DeviceName AssignedGPUs
```

Then submit:

```bash
baircondor submit --machine REDLRADADM35840 \
  --scratch /REDLRADADM35839/home/$USER/condor-scratch \
  --project eegfm --jobname <short-name> \
  --gpus 2 --cpus 8 --mem 256G \
  --conda-env eeg2025 --conda-base /home/spanchavati/anaconda3 \
  -- python run_pretraining_from_config.py --config configs/<your_config>.py
```

If `baircondor` isn't on `PATH` in the current shell, use the full path:
`/raid/spanchavati/anaconda3/envs/eeg2025/bin/baircondor`.

Notes:
- `REDLRADADM35840` is the current default target — `REDLRADADM35839` has had
  intermittent CUDA ECC errors, and doesn't have `eegfm_data` mounted (fails
  with `ValueError: No training datasets available.` / `no loader raised, all
  splits were empty`). If 35840 dies with a `CUDA error: uncorrectable ECC
  error`, only then try the other machine.
- If the model has unused parameters under DDP (e.g. `n_local=0` branches),
  add `"strategy": "ddp_find_unused_parameters_true"` in the config's
  `trainer_config`, or plain `"ddp"` will crash on the first optimizer step.
- **Monitor** with:
  ```bash
  condor_q -constraint 'Owner=="spanchavati"' -af ClusterId JobStatus RemoteHost
  # JobStatus: 1=Idle 2=Running 5=Held
  tail -c 2000 <scratch>/condor-runs/spanchavati/eegfm/<jobname>/<run_id>/stdout.txt
  tail -c 2000 <scratch>/condor-runs/spanchavati/eegfm/<jobname>/<run_id>/stderr.txt
  ```
  Collapse detection raises `ValueError: Embedding collapse detected!` in
  stderr if `batch_std`/`seq_std` drops below 0.05 — check `metrics.csv` in
  the run's `lightning_logs_rebuttal/.../version_N/` dir for the trend.
- Checkpoints land in `eegfmchallenge/lightning_logs_rebuttal/<group>/version_N/checkpoints/`
  (the `group` is set in the config's `logger_config`). Note the `version_N` —
  you'll need it for the eval sweep config below.

## 2. Evaluate (here, in `EEG-Bench-vICMl`)

1. **Conda env matters**: `conda activate eeg_bench` (not `base` — the
   default `python3` on this box is base conda and is missing `mne` etc.,
   which fails instantly with `ModuleNotFoundError: No module named 'mne'`).

2. **Check `eeg_bench/config.json`**: `lejepa.eegfm_path` must be the
   fully-qualified cross-host path, `/REDLRADADM35839/home/spanchavati`, not
   the bare `/home/spanchavati`. The bare path only resolves when running
   literally on `REDLRADADM35839`; from any other box it fails with
   `ModuleNotFoundError: No module named 'eegfmchallenge'`. This is already
   fixed as of this writing — if it regresses, that's the fix.

3. **Add a sweep config** in `sweep_configs/`, copying an existing one (e.g.
   `sweep_configs/rebuttal_ema_noproj.yaml`) and updating `base_path`/`version`
   to point at the new checkpoint:

   ```yaml
   models:
     <model-name>:
       base_path: /REDLRADADM35839/home/spanchavati/eegfmchallenge/lightning_logs_rebuttal/<group>
       version: <N>
   checkpoints:
     steps: []
     include_last: true
   tasks:
     epoch_sweep: []
     final_only: [all]   # all 14 tasks
   training:
     linear_probe: true
     data_percentages: [1.0]
     seeds: [42, 123, 456]
   execution:
     gpus: 1
     workers_per_gpu: 1
     log_dir: rebuttal_logs/<model-name>
     results_dir: results_rebuttal/<model-name>
   ```

4. **Dry-run first** (cheap sanity check that paths/task list resolve):

   ```bash
   python run_lejepa_sweep.py --config sweep_configs/<your_config>.yaml --dry-run
   ```

5. **Run for real**, in the background (a full 14-task × 3-seed sweep takes
   ~1.5–3 hours depending on dataset sizes):

   ```bash
   conda activate eeg_bench
   CUDA_VISIBLE_DEVICES=<free_gpu> nohup python3 run_lejepa_sweep.py \
     --config sweep_configs/<your_config>.yaml \
     > rebuttal_logs/<name>_sweep_run.log 2>&1 &
   ```

   Check GPU availability first with `nvidia-smi`. Monitor with
   `tail -c 1000 rebuttal_logs/<name>_sweep_run.log` (progress bar) and
   `ls rebuttal_logs/<model-name>/*.log` (per-task/seed logs — check for
   `Return code: 0` vs tracebacks).

6. **Results** land in `results_rebuttal/<model-name>/raw/*.json` (raw
   predictions + metadata) and `results_rebuttal/<model-name>/classification_results_*.txt`
   (human-readable per-task metrics, incl. balanced accuracy). To aggregate
   into BCI/Clinical/Overall means, match each raw JSON's `timestamp` to the
   closest `classification_results_*.txt` by embedded timestamp (the sweep
   doesn't write one combined summary file) — see the per-task tables in
   `eegfmchallenge/rebuttal/ema/PLAN.md` or `rebuttal/channel_mixer/PLAN.md`
   for worked examples.
