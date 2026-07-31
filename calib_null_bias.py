#!/usr/bin/env python3
"""Why quantify_pca_changepoint.py uses a cross-chunk null and a MEAN margin.

Run: python calib_null_bias.py    (no data needed, pure synthetic, ~20 s)

THE PROBLEM
-----------
M4 and M7 compare a statistic on the true labels against the same statistic on
null labels. The old null, `_relocate`, drew cut positions uniformly. It
therefore preserved the transition COUNT but not the class fraction, the run
lengths, or the boundary-position distribution. eta2 and the M4 AUC are both
maximised near a balanced split, so if the real boundaries are distributed
differently from uniform, the null is systematically easier or harder than the
truth for reasons that have nothing to do with the embedding.

WHAT THIS SCRIPT SHOWS
----------------------
On a trajectory that is a PURE CLOCK -- a temporal ramp plus noise, carrying no
state information at all, so the correct margin is exactly 0.0 -- the old null
manufactures a margin whose SIGN depends on where the boundaries sit:

    boundaries spread over the chunk   +0.018
    off-centre (short late bursts)     -0.019      <- TUAR artifact looks like this
    centred (seizure-like)             +0.080      <- CHB-MIT seizure looks like this

That is the whole explanation for two numbers this project reported before
2026-07-30: artifact Laya at -0.0475, and seizure Laya at +0.0280. The seizure
"result" is a quarter of the size of the bias that produces it on a clock.

The cross-chunk null draws whole real annotations from OTHER chunks, matched on
transition count, so class fraction and run lengths come from the empirical
distribution by construction. It reads +0.0008, +0.0005 and +0.0000 in the same
three regimes, and +0.088 when a genuine state step is planted, so it is both
unbiased and powerful.

SECOND FINDING: REPORT THE MEAN MARGIN, NOT THE MEDIAN. eta2 is nonlinear in
boundary position, so median[eta2 - mean(nulls)] is biased even under the
correct null (+0.0184 on a clock) while the mean is not (+0.0008).
"""
import numpy as np

import quantify_pca_changepoint as Q

Q.PC_NORM = "plot"
N_TOK, N_SEC, DIM = 160, 16, 8


def trial(bmin, bmax, kind, null, n=600, seed=0, amp=0.6, guard=0):
    """One regime. Returns (median margin, mean margin, median rank)."""
    rng  = np.random.default_rng(seed)
    labs = []
    for _ in range(n):
        b = int(rng.integers(bmin, bmax))
        l = np.zeros(N_SEC, int); l[b:] = 1
        labs.append(l)
    pool = Q.LabelPool(labs)
    M, R = [], []
    for i, ls in enumerate(labs):
        lt = Q.tok_labels_repeat(ls, N_TOK)
        t  = np.linspace(-1, 1, N_TOK)[:, None]
        Z  = t * rng.normal(size=(1, DIM)) + 0.25 * rng.normal(size=(N_TOK, DIM))
        if kind == "state":
            Z = Z + amp * lt[:, None] * rng.normal(size=(1, DIM))
        nr = np.random.default_rng(1000 + i)
        ns = (Q.sample_null_labels(pool, i, 1, 40, nr) if null == "chunk"
              else [Q._relocate(ls, nr) for _ in range(40)])
        r = Q.within_state_consistency(
            Z, lt, [Q.tok_labels_repeat(s, N_TOK) for s in ns], guard)
        if np.isfinite(r["eta2"]) and np.isfinite(r["eta2n"]):
            M.append(r["eta2"] - r["eta2n"]); R.append(r["eta2rank"])
    return np.median(M), np.mean(M), np.median(R)


REGIMES = [("spread  (b in 2..14)", 2, 15),
           ("off-centre (b in 11..14, TUAR-like)", 11, 15),
           ("centred (b in 6..10, seizure-like)",  6, 11)]

print("=" * 84)
print("PURE CLOCK: no state information, so the correct margin is exactly 0.0000")
print("=" * 84)
print(f"{'boundary regime':<38}{'null':<9}{'medMargin':>11}{'meanMargin':>12}{'rank':>8}")
for name, lo, hi in REGIMES:
    for null in ("uniform", "chunk"):
        md, mn, rk = trial(lo, hi, "clock", null)
        flag = "  <== BIAS" if abs(mn) > 0.01 else ""
        print(f"{name:<38}{null:<9}{md:>+11.4f}{mn:>+12.4f}{rk:>8.3f}{flag}")

print()
print("=" * 84)
print("PLANTED STATE STEP: a working statistic must read clearly positive")
print("=" * 84)
print(f"{'boundary regime':<38}{'null':<9}{'medMargin':>11}{'meanMargin':>12}{'rank':>8}")
for name, lo, hi in REGIMES:
    for null in ("uniform", "chunk"):
        md, mn, rk = trial(lo, hi, "state", null)
        print(f"{name:<38}{null:<9}{md:>+11.4f}{mn:>+12.4f}{rk:>8.3f}")

print()
print("=" * 84)
print("GUARD BAND on a step smeared by a 10-token moving average (LaBraM's 1 s")
print("window at 0.1 s stride). Without a guard the smeared ramp is counted as")
print("within-state jitter, penalising the baseline for its own receptive field.")
print("=" * 84)


def smeared(guard, width=10, n=300, seed=3):
    rng = np.random.default_rng(seed)
    js, fs = [], []
    for i in range(n):
        b  = int(rng.integers(5, 12))
        ls = np.zeros(N_SEC, int); ls[b:] = 1
        lt = Q.tok_labels_repeat(ls, N_TOK)
        t  = np.linspace(-1, 1, N_TOK)[:, None]
        Z  = (t * rng.normal(size=(1, DIM)) + 0.25 * rng.normal(size=(N_TOK, DIM))
              + 0.6 * lt[:, None] * rng.normal(size=(1, DIM)))
        k  = np.ones(width) / width
        Z  = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, Z)
        r  = Q.within_state_consistency(Z, lt, [], guard)
        if np.isfinite(r["jit"]):
            js.append(r["jit"]); fs.append(r["jitfree"])
    return np.median(js), np.median(fs)


for g in (0, 5):
    j, f = smeared(g)
    print(f"  guard={g/10:.1f}s   jit={j:.4f}   jitfree={f:.4f}")
print("  The effect is real but modest (~7%): the guard is a fairness fix, not")
print("  a rescue, and it does not come close to closing a 9x jitter gap.")
