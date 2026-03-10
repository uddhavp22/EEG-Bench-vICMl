# plot_noise_runs.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score

RAW_DIR = Path("results/raw")

def load_noisy_runs(raw_dir: Path) -> list[dict]:
    runs = []
    for p in raw_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if data.get("eval_noise_types"):  # only noisy runs
            data["_path"] = str(p)
            runs.append(data)
    return runs

def compute_metrics(run: dict) -> dict:
    # Assumes one model per file (true for your command)
    y_true_ds = run["y_test"][0]
    y_pred_ds = run["results"][0]

    y_true = np.concatenate([np.array(y) for y in y_true_ds])
    y_pred = np.concatenate([np.array(y) for y in y_pred_ds])

    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    y_pred_enc = le.transform(y_pred)

    return {
        "task": run.get("task_name"),
        "model": run.get("models_names", ["model"])[0],
        "noise_type": "+".join(run.get("eval_noise_types") or []),
        "snr_db": float(run.get("eval_noise_snr_db")),
        "acc": accuracy_score(y_true_enc, y_pred_enc),
        "bal_acc": balanced_accuracy_score(y_true_enc, y_pred_enc),
        "path": run.get("_path"),
    }

def main():
    runs = load_noisy_runs(RAW_DIR)
    if not runs:
        raise SystemExit(f"No noisy runs found in {RAW_DIR}")

    rows = [compute_metrics(r) for r in runs]
    df = pd.DataFrame(rows).sort_values(["noise_type", "snr_db"])

    print(df[["noise_type", "snr_db", "acc", "bal_acc"]])

    plt.figure(figsize=(8, 5))
    for noise_type, g in df.groupby("noise_type"):
        g = g.sort_values("snr_db")
        plt.plot(g["snr_db"], g["bal_acc"], marker="o", label=noise_type)

    plt.gca().invert_xaxis()  # optional: show 30 -> 0 left-to-right
    plt.xlabel("SNR (dB)")
    plt.ylabel("Balanced Accuracy")
    plt.title("Noise Robustness (Noisy Runs Only)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("noise_robustness_noisy_runs.png", dpi=300)

if __name__ == "__main__":
    main()
