#!/usr/bin/env python3
"""
Quantify the Figure 2 PCA claim, over a whole split instead of the 5-11 chunks
that get plotted. Headless version of analyze_embeddings_rebuttals.ipynb.

WHAT IT MEASURES
----------------
The paper's own sentence (§4.2) is the spec, and it is a change point claim:

    "Laya's embeddings show clear temporal structure aligned with seizure onset
     and termination: at a state change, the colors shift distinctly. In
     contrast, LaBraM's embeddings fluctuate throughout the recording without
     clear correspondence to the clinical event."

So both metrics run on one curve, over per-dimension standardised patch
embeddings:

    c[t] = || mean(Z[t-w:t]) - mean(Z[t:t+w]) ||

That curve IS the picture: how much the embedding differs before vs after t.

  M1  percentile of c at the annotated transition, within the chunk's own c.
      Uniform on [0,1] under indifference -> one-sided Wilcoxon against 0.5.
  M2  |argmax of c - nearest annotated transition|, in SECONDS, against a
      chance level from drawing the argmax uniformly. This is the number for
      the rebuttal prose.

Both are rank / distance statistics on a within-chunk curve, so they are
invariant to embedding width, PCA sign flips and per-chunk rescaling. Laya at
D=384 and LaBraM at D=200 are directly comparable. (A multivariate Cohen's d
between states is NOT: on pure noise with uninformative labels it reads 3.31 at
160x384 and 2.47 at 151x200, a +0.84 gap from dimensionality alone. Not used.)

BOTH DIRECTIONS OF TRANSITION ARE SCORED. Seizure chunks routinely begin
mid-event, so the only annotated boundary inside a 16 s window can be the
termination. Scoring 0->1 only would silently drop those chunks from N.

TOKEN GEOMETRY
--------------
Laya   : (n_patches, D) straight from the cache. The V2 mixer collapses
         channels before tokenisation, so the token axis is already pure time,
         160 tokens per 16 s chunk at 10 Hz.
LaBraM : forward_features returns (1, C*n_patch, D), CHANNEL-MAJOR, from
         `rearrange(x, 'B N A T -> B (N A) T')` at modeling_finetune.py:257.
         It is fed sliding 1 s windows at 0.1 s stride with exactly ONE time
         patch each, so the token axis has length C and mean(dim=1) is
         unambiguously a channel mean, leaving one row per time step. Same
         approach as _labram_patch_embs_raw_segment in analyze_embeddings_bci.

CAVEATS THIS SCRIPT DOES NOT HIDE
---------------------------------
1. get_input_chans (LaBraM/utils.py:723) requires every channel name to be in
   standard_1020. CHB-MIT bipolar names (T8-P8) are not, so it falls back to
   `range(C+1)`, attaching pretrained electrode position embeddings to the
   wrong electrodes. --bipolar-map first maps T8-P8 -> T8 as a robustness
   check. The default reproduces the notebook exactly, warnings and all.
2. LaBraM is fed the LeJEPA-preprocessed EEG (same H5 both models read), not
   its own pipeline's. That is the controlled comparison (same input, different
   model) but it is not what LaBraM's own eval uses.

USAGE
-----
    python quantify_pca_changepoint.py --task binary_artifact_clinical
    python quantify_pca_changepoint.py --task seizure_clinical --split test
    python quantify_pca_changepoint.py --task seizure_clinical \
        --bipolar-map first --out figures/cp_seizure_bipolarfix

Writes <out>.csv (one row per chunk), <out>.png, and prints the aggregate.
"""

import argparse
import glob
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# The checkpoint reported in the paper. NOTE: the notebook's CKPT_PATH carries a
# stray leading space; md5 hashes the PATH STRING, so " /radraid2/..." and
# "/radraid2/..." give different cache filenames. The published cache
# (433c0df19392) is the hash of the path WITHOUT the space, so we strip.
# ---------------------------------------------------------------------------
DEFAULT_CKPT = ("/radraid2/spanchavati/eegfm/lightning_logs_rebuttal/"
                "fullds_02sig_20k/version_0/checkpoints/last.ckpt")

LABRAM_SFREQ          = 200
LABRAM_PATCH_SZ       = 200   # 1 s at 200 Hz
LABRAM_STRIDE_SAMPLES = 20    # 0.1 s stride, matches Laya's 0.1 s patches

TASK_ALIASES = {"sleep_staging": "sleep_stages"}


# ===========================================================================
# metrics
# ===========================================================================

def change_score(Z, w):
    """c[t] = || mean(Z[t-w:t]) - mean(Z[t:t+w]) ||, on per-dim standardised Z.

    Averaging over w tokens makes this respond to a SUSTAINED state change
    rather than a single-token spike, which is what "the colors shift
    distinctly" means. Standardised Euclidean rather than cosine: cosine
    degenerates when the window means have near-zero norm, saturating near 1
    regardless of the true shift. Only the within-chunk rank of c is used, so
    any monotone rescaling is irrelevant anyway.

    NaN in the first and last w positions.
    """
    n = len(Z)
    c = np.full(n, np.nan)
    if n < 2 * w + 1:
        return c
    sd = Z.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Zs = (Z - Z.mean(axis=0)) / sd
    cs = np.vstack([np.zeros(Zs.shape[1]), np.cumsum(Zs, axis=0)])
    for t in range(w, n - w + 1):
        c[t] = float(np.linalg.norm((cs[t] - cs[t - w]) / w -
                                    (cs[t + w] - cs[t]) / w))
    return c


def transitions(lab):
    """Indices where the label changes, BOTH directions. See module docstring."""
    lab = np.asarray(lab)
    return np.where(lab[1:] != lab[:-1])[0] + 1


def m1_transition_percentile(Z, lab, w, tol):
    """Percentile of the change score at each annotated transition, within the
    chunk's own change scores. Uniform on [0,1] under indifference.

    The reference is a distribution of ROLLING MAXIMA of width 2*tol+1, not of
    raw scores. Scoring the transition by a max over a window but comparing
    against raw scores inflates the null percentile to ~(2*tol+1)/(2*tol+2),
    measured at 0.819 for tol=4, which would have looked like a result.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    c = change_score(Z, w)
    if (~np.isnan(c)).sum() < 10:
        return []
    width = 2 * tol + 1
    cf    = np.where(np.isnan(c), -np.inf, c)
    roll  = sliding_window_view(cf, width).max(axis=1) if len(cf) >= width else cf
    ref   = roll[np.isfinite(roll)]
    if ref.size < 10:
        return []
    out = []
    for t in transitions(lab):
        seg = c[max(0, t - tol):min(len(c), t + tol + 1)]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out.append(float((ref < seg.max()).mean()))
    return out


def m1_ceiling(n_tokens, w, tol):
    """M1's upper bound, and it is TIGHT and it is not 1.0.

    The transition is scored by max(c) over a (2*tol+1)-wide window; every
    rolling maximum containing that argmax is >= it, so `width` reference
    entries can never count as strictly less.

        160 tokens, w=5, tol=5 -> 0.9267  (measured 0.927 on a planted step)
        151 tokens, w=5, tol=5 -> 0.9220

    A "fraction of events above 0.95" statistic is therefore arithmetically
    unattainable and would read as a null result for a perfect detector.
    """
    width = 2 * tol + 1
    n_ref = max(min(n_tokens - width, n_tokens - w) - max(0, w - width + 1) + 1, 1)
    return float(1.0 - width / n_ref)


def m2_localisation(Z, lab, w, token_hz, n_null=2000, seed=0):
    """|argmax of the change score - nearest annotated transition|, in seconds.

    Returns (err_s, chance_s). chance_s draws the argmax uniformly over the
    valid range of c, holding the annotation and the chunk length fixed, so it
    only removes the link to the embedding. Report err_s against chance_s, not
    alone: a chunk whose transition sits mid-window has a low chance error by
    construction.
    """
    c  = change_score(Z, w)
    ok = np.where(~np.isnan(c))[0]
    tr = transitions(lab)
    if ok.size < 3 or tr.size == 0:
        return float("nan"), float("nan")
    t_hat  = ok[int(np.nanargmax(c[ok]))]
    err    = float(np.abs(tr - t_hat).min() / token_hz)
    rng    = np.random.default_rng(seed)
    draws  = rng.choice(ok, size=n_null, replace=True)
    chance = float(np.mean([np.abs(tr - d).min() for d in draws]) / token_hz)
    return err, chance


def top3(Z):
    """The 3 PCs the panel actually colours with. If the effect is in the full
    embedding but absent here, the number and the picture are measuring
    different things and the rebuttal must say which it quotes."""
    Zc = Z - Z.mean(axis=0)
    return Zc @ np.linalg.svd(Zc, full_matrices=False)[2][:3].T


def _zscore(Z):
    sd = Z.std(axis=0)
    return (Z - Z.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def _auc(scores, y):
    """Rank AUC. Chance is exactly 0.5 at any dimension and any class balance."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.argsort(np.argsort(np.asarray(scores, float))) + 1.0
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def state_auc(Zs, labs, n_folds=5, seed=0, shift_null=False):
    """M3. Do event tokens occupy a different region than non-event tokens?

    This is what the PCA panel actually shows: not a sharp edge at the
    boundary, but a colour that HOLDS for the duration of the event. M1 cannot
    see that. A panel full of high-frequency flicker can have a perfectly real
    sustained level shift whose change score never wins the argmax.

    The contrast direction is fit on OTHER chunks and applied to held-out ones,
    so nothing is fit and scored on the same data and a wider embedding cannot
    buy accuracy. AUC is a rank statistic, so chance is exactly 0.5 at D=384
    and at D=200. That is what makes Laya and LaBraM directly comparable here,
    and what a multivariate Cohen's d could not deliver.

    shift_null circularly shifts each chunk's labels first, preserving the run
    structure of a temporal annotation while removing the label-state link.

    Returns (pooled_auc, per_chunk_auc array).
    """
    rng   = np.random.default_rng(seed)
    n     = len(Zs)
    if n < n_folds:
        return float("nan"), np.array([])
    labs  = [np.asarray(l) for l in labs]
    if shift_null:
        labs = [np.roll(l, int(rng.integers(1, max(len(l), 2)))) for l in labs]
    fold  = rng.permutation(n) % n_folds
    per, ps, py = [], [], []
    for f in range(n_folds):
        u, cnt = np.zeros(Zs[0].shape[1]), 0
        for i in range(n):
            if fold[i] == f:
                continue
            l = labs[i]
            if l.min() == l.max():
                continue
            u   += Zs[i][l == 1].mean(axis=0) - Zs[i][l == 0].mean(axis=0)
            cnt += 1
        if cnt == 0:
            continue
        u = u / (np.linalg.norm(u) + 1e-12)
        for i in range(n):
            if fold[i] != f:
                continue
            sc = Zs[i] @ u
            per.append(_auc(sc, labs[i]))
            ps.append(sc)
            py.append(labs[i])
    if not ps:
        return float("nan"), np.array([])
    return (_auc(np.concatenate(ps), np.concatenate(py)),
            np.array([v for v in per if np.isfinite(v)], dtype=float))


def _relocate(lab, rng):
    """Null label with the SAME number of transitions, at random positions.

    A circular shift is the wrong null here. Rolling a single step wraps it
    into two segments, which a time-like axis separates worse than one clean
    step, so the null lands too low and a pure clock scores a large false
    margin. Relocating instead keeps the temporal shape and moves only the
    location, so any axis that separates ANY step equally well -- which is what
    a clock is -- scores at the null by construction, and only a code that
    tracks the TRUE boundary clears it.
    """
    lab = np.asarray(lab)
    n   = len(lab)
    k   = int((lab[1:] != lab[:-1]).sum())
    if k < 1 or n < k + 2:
        return lab
    cuts = np.sort(rng.choice(np.arange(1, n), size=k, replace=False))
    out  = np.empty(n, dtype=int)
    v    = int(rng.integers(0, 2))
    prev = 0
    for c in cuts:
        out[prev:c] = v; v = 1 - v; prev = int(c)
    out[prev:] = v
    return out


WAVE_FEATS = ["rms", "linelen", "delta", "theta", "alpha", "beta", "gamma"]


def waveform_features(eeg, sfreq, centers_s, win_s=1.0):
    """Low-level signal descriptors on a 1 s window centred at each token.

    Identical definition and identical window length for both models, sampled
    at whatever token centres each one has, so the only thing that differs is
    the embedding being asked to predict them.
    """
    eeg  = np.asarray(eeg, dtype=np.float64)
    C, T = eeg.shape
    w    = int(round(win_s * sfreq))
    idx  = np.clip(np.round(np.asarray(centers_s) * sfreq).astype(int) - w // 2,
                   0, max(T - w, 0))
    seg  = np.stack([eeg[:, i:i + w] for i in idx], axis=0)      # (n, C, w)
    seg  = seg - seg.mean(axis=2, keepdims=True)

    rms  = np.sqrt((seg ** 2).mean(axis=2)).mean(axis=1)
    ll   = np.abs(np.diff(seg, axis=2)).mean(axis=2).mean(axis=1)
    P    = np.abs(np.fft.rfft(seg * np.hanning(w), axis=2)) ** 2
    fr   = np.fft.rfftfreq(w, 1.0 / sfreq)
    bp   = [P[:, :, (fr >= lo) & (fr < hi)].sum(axis=2).mean(axis=1)
            for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]]
    return np.log(np.vstack([rms, ll] + bp).T + 1e-12)


def tracking_scores(Zs, Ws, ls, k=32, n_folds=5, seed=0):
    """M6. Waveform tracking vs state tracking, at equal dimensionality.

    Both models are projected onto their own top-k global PCs fit on the
    TRAINING chunks, so Laya's 384 and LaBraM's 200 both become k and neither
    can win on width. Then, held out:

      waveform R2  OLS from the k PCs to the 1 s signal descriptors
      state AUC    mean-difference direction in the k PCs, scored by rank AUC

    Everything is per-chunk centred on both sides, so this measures tracking of
    WITHIN-chunk fluctuation, not recognition of which chunk you are in.

    Returns (state_auc, per_feature_r2).
    """
    rng  = np.random.default_rng(seed)
    n    = len(Zs)
    if n < n_folds:
        return float("nan"), np.full(len(WAVE_FEATS), np.nan)
    fold = rng.permutation(n) % n_folds
    r2s, ps, py = [], [], []
    for f in range(n_folds):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] == f]
        if not tr or not te:
            continue
        Xtr = np.concatenate([Zs[i] for i in tr]).astype(np.float64)
        mu  = Xtr.mean(axis=0)
        Vt  = np.linalg.svd(Xtr - mu, full_matrices=False)[2][:k]
        Ptr = (Xtr - mu) @ Vt.T
        Wtr = np.concatenate([Ws[i] for i in tr]).astype(np.float64)

        A = np.hstack([Ptr, np.ones((len(Ptr), 1))])
        beta, *_ = np.linalg.lstsq(A, Wtr, rcond=None)

        ytr = np.concatenate([ls[i] for i in tr]).astype(bool)
        if ytr.all() or not ytr.any():
            continue
        u = Ptr[ytr].mean(axis=0) - Ptr[~ytr].mean(axis=0)
        u = u / (np.linalg.norm(u) + 1e-12)

        Pte  = np.concatenate([(np.asarray(Zs[i], np.float64) - mu) @ Vt.T
                               for i in te])
        Wte  = np.concatenate([Ws[i] for i in te]).astype(np.float64)
        pred = np.hstack([Pte, np.ones((len(Pte), 1))]) @ beta
        sst  = ((Wte - Wte.mean(axis=0)) ** 2).sum(axis=0)
        r2s.append(1.0 - ((Wte - pred) ** 2).sum(axis=0) /
                   np.where(sst > 0, sst, 1.0))
        ps.append(Pte @ u)
        py.append(np.concatenate([ls[i] for i in te]))
    if not r2s:
        return float("nan"), np.full(len(WAVE_FEATS), np.nan)
    return (_auc(np.concatenate(ps), np.concatenate(py)),
            np.mean(r2s, axis=0))


def within_state_consistency(Z, lab, n_null=40, seed=0):
    """M7. Does each state get a CONSISTENT colour for its whole duration?

    AUC ranks tokens along one direction, so a representation that jumps around
    inside a state pays nothing as long as that one projection still separates.
    This is what "LaBraM changes at the boundary but is all over the place
    within the seizure" means, and no other metric here sees it.

      eta2    fraction of 3-PC variance that is BETWEEN states rather than
              within them. High = each state holds one colour.
      jitter  mean token-to-token step taken WITHIN a state, divided by the
              distance between the two state centroids. High = flickering.

    eta2 gets a relocated-label null since a drifting trajectory scores high
    against any split; jitter needs none, as it never looks at the labels
    except to exclude the steps that cross a boundary.

    Returns (eta2, eta2_null, jitter).
    """
    P   = top3(Z)
    lab = np.asarray(lab)

    def _eta2(l):
        if l.min() == l.max():
            return np.nan
        sst = float(((P - P.mean(axis=0)) ** 2).sum())
        if sst <= 0:
            return np.nan
        ssw = sum(float(((P[l == v] - P[l == v].mean(axis=0)) ** 2).sum())
                  for v in (0, 1))
        return 1.0 - ssw / sst

    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    same = lab[1:] == lab[:-1]
    sep  = (float(np.linalg.norm(P[lab == 1].mean(axis=0) - P[lab == 0].mean(axis=0)))
            if lab.min() != lab.max() else np.nan)
    jit  = (float(step[same].mean() / sep)
            if np.isfinite(sep) and sep > 0 and same.any() else np.nan)

    rng = np.random.default_rng(seed)
    nl  = [_eta2(_relocate(lab, rng)) for _ in range(n_null)]
    return (_eta2(lab),
            float(np.nanmean(nl)) if np.any(np.isfinite(nl)) else np.nan,
            jit)


def cusum_split(P):
    """Best single split of a token trajectory, by the standard multivariate
    CUSUM statistic  sqrt(t(n-t)/n) * ||mean(P[:t]) - mean(P[t:])||.

    Global, unlike M1's local w=5 difference, so a transition that takes
    seconds still registers. The sqrt weight is what stops the argmax being
    dragged to the edges where one block is tiny and its mean is noisy.

    Returns the argmax index.
    """
    n  = len(P)
    cs = np.vstack([np.zeros(P.shape[1]), np.cumsum(P, axis=0)])
    t  = np.arange(1, n)
    m1 = cs[1:n] / t[:, None]
    m2 = (cs[n] - cs[1:n]) / (n - t)[:, None]
    d  = np.sqrt(t * (n - t) / n) * np.linalg.norm(m1 - m2, axis=1)
    return int(np.argmax(d)) + 1


def m5_split_error(Z, lab, token_hz):
    """M5. Does the dominant split of the 3 PCs the figure colours with land on
    the annotated boundary?

    Separability saturates on a smooth trajectory -- a random boundary scores
    ~0.96 on Laya -- so M4 cannot discriminate. Location does not saturate:
    smoothness says nothing about WHERE the largest split sits. This is M2's
    question asked with a global segmentation instead of a local window.

    The chance level CANNOT be a uniform draw. The sqrt weight makes the argmax
    centre-biased, and real event boundaries are centre-biased too, so two
    independent centre biases manufacture agreement: a smooth random walk with
    no relation to the label beats a uniform baseline on 99.7% of chunks. The
    null is instead built across chunks in aggregate(), pairing this chunk's
    split against OTHER chunks' boundaries, which keeps both biases and
    destroys only their correspondence.

    Returns (err_s, split_s, trans_s) with the transitions ";"-joined.
    """
    tr = transitions(lab)
    if len(tr) == 0:
        return np.nan, np.nan, ""
    t = cusum_split(top3(Z))
    return (float(np.min(np.abs(tr - t))) / token_hz,
            t / token_hz,
            ";".join(f"{x / token_hz:.4f}" for x in tr))


def pca_state_auc(Z, lab, n_null=40, seed=0):
    """M4. On the 3 PCs the figure actually colours with: are event tokens
    separable from non-event tokens WITHIN the chunk?

    The direction is fit inside the same chunk it is scored on, which is
    circular and inflates the value. That is fine, because the null is built by
    the identical procedure on relocated labels, so the inflation cancels and
    only the margin over the null counts. See _relocate for why the null moves
    the boundary rather than circularly shifting it.

    This is the only place a within-chunk fit is legitimate: 3 dims against
    ~160 tokens cannot separate arbitrary labels, whereas 384 dims would
    separate anything and drive both the real and null values to 1.0.

    Returns (auc, null_auc).
    """
    P   = top3(Z)
    lab = np.asarray(lab)

    def _fit(l):
        if l.min() == l.max():
            return np.nan
        u = P[l == 1].mean(axis=0) - P[l == 0].mean(axis=0)
        nu = float(np.linalg.norm(u))
        return _auc(P @ (u / nu), l) if nu > 0 else np.nan

    rng   = np.random.default_rng(seed)
    nulls = [_fit(_relocate(lab, rng)) for _ in range(n_null)]
    return _fit(lab), float(np.nanmean(nulls)) if np.any(np.isfinite(nulls)) else np.nan


def positional_stats(Z):
    """How much of this embedding is just a clock?

    A representation whose PCA is dominated by "where am I in the chunk" will
    produce a smooth left-to-right colour sweep that looks like semantic
    organisation but is independent of the annotation. It also has a nearly
    flat change score, so M1 sits on the null and the argmax drifts to the
    chunk edges, which reads as WORSE than chance localisation.

    Returns
      pc1_t : |corr(PC1, token index)|, 1.0 = PC1 is a pure ramp
      evr1  : fraction of variance in PC1
      t_r2  : fraction of TOTAL embedding variance explained by a cubic in
              token index alone. This is the decisive number.
    """
    n  = len(Z)
    Zc = Z - Z.mean(axis=0)
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    t   = np.linspace(-1.0, 1.0, n)
    pc1 = Zc @ Vt[0]
    X   = np.vstack([np.ones(n), t, t ** 2, t ** 3]).T
    beta, *_ = np.linalg.lstsq(X, Zc, rcond=None)
    resid = Zc - X @ beta
    denom = float((Zc ** 2).sum())
    return (float(abs(np.corrcoef(pc1, t)[0, 1])),
            float(S[0] ** 2 / (S ** 2).sum()),
            float(1.0 - (resid ** 2).sum() / denom) if denom > 0 else np.nan)


def detrend_tokens(Z, order=3):
    """Regress a per-chunk polynomial in token index out of every dimension.

    If M1 rises above the null only after this, the state structure is present
    but masked by a global temporal drift, which is a defensible thing to say.
    If it stays on the null, there is nothing there to unmask.
    """
    n = len(Z)
    t = np.linspace(-1.0, 1.0, n)
    X = np.vstack([t ** k for k in range(order + 1)]).T
    beta, *_ = np.linalg.lstsq(X, Z, rcond=None)
    return Z - X @ beta


def tok_labels_repeat(lab, n_tok):
    """1 Hz chunk labels -> one per token, for a grid that tiles the chunk."""
    lab = np.asarray(lab)
    assert n_tok % len(lab) == 0, (
        f"{n_tok} tokens does not divide {len(lab)} labels; rounding here would "
        f"drift the transition by a growing offset and destroy M1")
    return np.repeat(lab, n_tok // len(lab))


def tok_labels_windows(lab, starts, win, sfreq_out):
    """1 Hz chunk labels -> one per SLIDING window, by window centre."""
    centres = (np.asarray(starts) + win / 2.0) / sfreq_out
    return np.asarray(lab)[np.clip(centres.astype(int), 0, len(lab) - 1)]


# ===========================================================================
# cache / data plumbing (mirrors analyze_embeddings_rebuttals.ipynb cell 2)
# ===========================================================================

def find_lejepa_index(task, ckpt_path, split, attentive=True):
    from eeg_bench.config import get_config_value
    from eeg_bench.models.clinical.EEGLejepa_model import EMBED_CACHE_VERSION
    task = TASK_ALIASES.get(task, task)
    cache_dir = Path(get_config_value("cache", ".cache")) / "lejepa_embeddings"
    h = hashlib.md5(str(ckpt_path).encode()).hexdigest()[:12]
    tag = "_seq" if attentive else ""
    pattern = f"{task}_{h}_*_{split}{tag}_{EMBED_CACHE_VERSION}.index.json"
    hits = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise FileNotFoundError(
            f"no cache index for {cache_dir / pattern}\n"
            f"  ckpt hash {h} is md5 of the PATH STRING, so a stray leading "
            f"space or trailing slash changes it. Path used: {ckpt_path!r}")
    return hits[0]


def find_h5(task, model_tag="LeJEPAClinical"):
    from eeg_bench.config import get_config_value
    task = TASK_ALIASES.get(task, task)
    root = get_config_value("make_dataset") or os.path.join(
        get_config_value("data"), "make_dataset")
    hits = sorted(glob.glob(os.path.join(root, f"{task}_{model_tag}_*.h5")),
                  key=os.path.getsize, reverse=True)
    if not hits:
        raise FileNotFoundError(f"no H5 for {root}/{task}_{model_tag}_*.h5")
    return hits[0]


def iter_chunks(index, limit=None):
    """Stream (seq_id, labels, emb) for chunks carrying a label transition.

    Streams rather than materialising `mixed`: a full split is thousands of
    chunks at 160x384 float32 each.
    """
    n = 0
    for shard in index["shards"]:
        labels  = np.load(shard["labels"], mmap_mode="r")
        embs    = np.load(shard["embeddings"], mmap_mode="r")
        seq_ids = shard["sequence_ids"]
        for i in range(len(labels)):
            lab = np.asarray(labels[i])
            if transitions(lab).size == 0:
                continue
            yield seq_ids[i], lab.copy(), np.asarray(embs[i]).squeeze()
            n += 1
            if limit and n >= limit:
                return


def decode_names(ch_names):
    return [(n.decode() if isinstance(n, (bytes, np.bytes_)) else str(n))
            for n in ch_names]


def map_channel_names(names, mode):
    """`none` reproduces the notebook exactly. `first` maps a bipolar pair to
    its first electrode (T8-P8 -> T8) so get_input_chans finds real positions
    instead of falling back to range(C+1)."""
    names = decode_names(names)          # idempotent, tolerates bytes or str
    if mode == "first":
        return [n.upper().replace("EEG ", "").strip().split("-")[0] for n in names]
    return [n.upper() for n in names]


# ===========================================================================
# LaBraM
# ===========================================================================

def load_labram(device):
    import torch
    from timm.models import create_model
    from eeg_bench.models.clinical.labram_model import check_and_download_pretrained_model
    from eeg_bench.models.clinical.LaBraM import modeling_finetune  # noqa: F401

    ck = torch.load(check_and_download_pretrained_model(), map_location="cpu",
                    weights_only=False)
    sd = {k[len("student."):]: v for k, v in ck["model"].items()
          if k.startswith("student.")}
    model = create_model(
        "labram_base_patch200_200", qkv_bias=False, rel_pos_bias=True,
        num_classes=0, drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
        drop_block_rate=None, use_mean_pooling=True, init_scale=0.001,
        use_rel_pos_bias=True, use_abs_pos_emb=True, init_values=0.1)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def labram_embeddings(model, device, eeg, ch_names, orig_sfreq, bipolar_map,
                      batch_size=128):
    """Dense sliding-window LaBraM embeddings on a real TIME axis.

    Each window holds exactly ONE time patch, so forward_features returns
    (n_win, C*1, D) and mean(dim=1) is a pure channel mean. Without this the
    token axis is C*n_patch in channel-major order and plotting or scoring it
    as time is meaningless.
    """
    import torch
    from scipy.signal import resample
    from eeg_bench.models.clinical.LaBraM import utils as labram_utils

    if orig_sfreq != LABRAM_SFREQ:
        new_T  = int(round(eeg.shape[1] * LABRAM_SFREQ / orig_sfreq))
        eeg_rs = resample(eeg, new_T, axis=1)
    else:
        eeg_rs = np.asarray(eeg)
    new_T = eeg_rs.shape[1]
    if new_T < LABRAM_PATCH_SZ:
        raise ValueError(f"segment shorter than one 1 s window: {new_T} samples")

    names       = map_channel_names(decode_names(ch_names), bipolar_map)
    input_chans = labram_utils.get_input_chans(names)

    starts  = np.arange(0, new_T - LABRAM_PATCH_SZ + 1, LABRAM_STRIDE_SAMPLES,
                        dtype=int)
    windows = np.stack([eeg_rs[:, s:s + LABRAM_PATCH_SZ] for s in starts], axis=0)

    out = []
    with torch.no_grad():
        for b0 in range(0, len(windows), batch_size):
            b = windows[b0:b0 + batch_size]
            tok = (torch.from_numpy(np.ascontiguousarray(b)).float()
                     .reshape(b.shape[0], b.shape[1], 1, LABRAM_PATCH_SZ) / 100.0)
            pe = model.forward_features(tok.to(device), input_chans=input_chans,
                                        return_patch_tokens=True)
            assert pe.shape[1] == b.shape[1], (
                f"token axis {pe.shape[1]} != channel count {b.shape[1]}; "
                f"channels and input_chans disagree")
            out.append(pe.mean(dim=1).cpu().numpy())
    return np.concatenate(out, axis=0), starts


# ===========================================================================
# main
# ===========================================================================

def aggregate(df, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    def vs_uniform(col, name, ceil=None):
        v = df[col].dropna().values
        if len(v) < 5:
            print(f"  {name:26s} n={len(v)}, too few"); return
        _, p = wilcoxon(v - 0.5, alternative="greater")
        c = f"  ceiling={df[ceil].iloc[0]:.4f}" if ceil else ""
        print(f"  {name:26s} n={len(v):5d}  median={np.median(v):.4f}{c}  "
              f"p_vs_0.5={p:.3e}")

    def paired(ca, cb, name, greater=True):
        d = df[[ca, cb]].dropna()
        if len(d) < 5:
            print(f"  {name:26s} n={len(d)}, too few"); return
        a, b = d[ca].values, d[cb].values
        _, p = wilcoxon(a, b, alternative="greater" if greater else "less")
        win = (a > b) if greater else (a < b)
        print(f"  {name:26s} n={len(d):5d}  Laya {np.median(a):7.3f}  "
              f"LaBraM {np.median(b):7.3f}  Laya better on {100*win.mean():5.1f}%  "
              f"p={p:.3e}")

    print("\nM1  change-score percentile at annotated transition (uniform null 0.5)")
    vs_uniform("m1_laya",       "Laya   full embedding", "ceil_laya")
    vs_uniform("m1_labram",     "LaBraM full embedding", "ceil_labram")
    vs_uniform("m1_laya_pc3",   "Laya   3 PCs shown")
    vs_uniform("m1_labram_pc3", "LaBraM 3 PCs shown")
    paired("m1_laya", "m1_labram", "paired, full embedding")
    paired("m1_laya_pc3", "m1_labram_pc3", "paired, 3 PCs")

    print("\nPOSITIONAL CODE  is the embedding just a clock?")
    for t, n in [("laya", "Laya  "), ("labram", "LaBraM")]:
        if f"timer2_{t}" not in df:
            continue
        print(f"  {n:26s} |corr(PC1, t)|={df[f'pc1t_{t}'].median():.3f}  "
              f"PC1 var={df[f'evr1_{t}'].median():.3f}  "
              f"var explained by cubic in t={df[f'timer2_{t}'].median():.3f}")
    if "m1_laya_dt" in df:
        print("\nM1 after removing a per-chunk polynomial in token index")
        vs_uniform("m1_laya_dt",     "Laya   detrended")
        vs_uniform("m1_labram_dt",   "LaBraM detrended")
        vs_uniform("m1_laya_dtpc3",  "Laya   detrended, 3 PCs")
        paired("m1_laya_dt", "m1_labram_dt", "paired, detrended")

    print("\nM2  change-point localisation error, seconds (lower is better)")
    for t, n in [("laya", "Laya  "), ("labram", "LaBraM")]:
        d = df[[f"err_{t}", f"chance_{t}"]].dropna()
        if len(d) < 5:
            continue
        _, p = wilcoxon(d[f"err_{t}"].values, d[f"chance_{t}"].values,
                        alternative="less")
        print(f"  {n:26s} n={len(d):5d}  median={d[f'err_{t}'].median():.3f}s  "
              f"chance={d[f'chance_{t}'].median():.3f}s  p_vs_chance={p:.3e}")
    paired("err_laya", "err_labram", "paired", greater=False)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].hist([df.m1_laya.dropna(), df.m1_labram.dropna()], bins=20,
               label=["Laya", "LaBraM"], color=["C0", "C3"])
    ax[0].axvline(0.5, color="k", ls="--")
    ax[0].legend(); ax[0].set_xlabel("M1 percentile at transition")
    ax[0].set_ylabel("chunks"); ax[0].set_title("M1, uniform null 0.5")

    d = df[["m1_laya", "m1_labram"]].dropna()
    ax[1].scatter(d.m1_labram, d.m1_laya, s=7, alpha=.3, c="k")
    ax[1].plot([0, 1], [0, 1], "r--")
    ax[1].set_xlabel("LaBraM"); ax[1].set_ylabel("Laya")
    ax[1].set_title(f"M1 paired, Laya higher on "
                    f"{100*(d.m1_laya > d.m1_labram).mean():.0f}% of n={len(d)}")

    e = df[["err_laya", "err_labram"]].dropna()
    hi = float(e.max().max()) if len(e) else 1.0
    ax[2].scatter(e.err_labram, e.err_laya, s=7, alpha=.3, c="k")
    ax[2].plot([0, hi], [0, hi], "r--")
    ax[2].set_xlabel("LaBraM error (s)"); ax[2].set_ylabel("Laya error (s)")
    ax[2].set_title(f"localisation, Laya closer on "
                    f"{100*(e.err_laya < e.err_labram).mean():.0f}%")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nfigure -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="binary_artifact_clinical")
    ap.add_argument("--split", default="train")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="checkpoint PATH; md5 of this string names the cache")
    ap.add_argument("--index", default=None, help="explicit .index.json, skips lookup")
    ap.add_argument("--h5", default=None, help="explicit recordings H5")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--window", type=int, default=5, help="w, in Laya tokens (0.5 s)")
    ap.add_argument("--tol", type=int, default=5, help="+/- tokens around annotation")
    ap.add_argument("--bipolar-map", choices=["none", "first"], default="none",
                    help="'first' maps T8-P8 -> T8 so LaBraM gets real electrode "
                         "positions instead of get_input_chans' range(C+1) fallback")
    ap.add_argument("--detrend", type=int, default=0, metavar="ORDER",
                    help="also score after regressing an order-N polynomial in "
                         "token index out of each chunk (try 3). Separates real "
                         "state structure from a global temporal ramp.")
    ap.add_argument("--smooth", type=int, default=0, metavar="K",
                    help="moving-average the token series by K before the change "
                         "score. M1 is driven by the largest spike, so a real but "
                         "gradual transition can lose to token-level flicker.")
    ap.add_argument("--state-auc", action="store_true",
                    help="also run M3, held-out state occupancy AUC. This is what "
                         "the PCA panel shows when the colour HOLDS across the "
                         "event rather than spiking at its edge.")
    ap.add_argument("--tracking", action="store_true",
                    help="also run M6: does the embedding track the WAVEFORM or "
                         "the STATE? Equal-dimension, held out over chunks.")
    ap.add_argument("--auc-max-chunks", type=int, default=2000,
                    help="cap on chunks retained in RAM for M3")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test N chunks")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from scipy import stats

    ckpt = str(args.ckpt).strip()
    if ckpt != args.ckpt:
        print(f"NOTE: stripped whitespace from --ckpt; md5 hashes the path string")

    index_path = args.index or find_lejepa_index(args.task, ckpt, args.split)
    h5_path    = args.h5 or find_h5(args.task)
    out        = args.out or f"figures/cp_{args.task}_{args.split}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"index  {index_path}")
    print(f"h5     {h5_path}")
    print(f"ckpt   {ckpt}")
    with open(index_path) as f:
        index = json.load(f)
    print(f"meta   {index.get('meta')}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model  = load_labram(device)
    print(f"device {device}\n")

    rows, t0, warned = [], time.time(), False
    keep = {"laya": [], "labram": []}
    keep_lab = {"laya": [], "labram": []}
    keep_w   = {"laya": [], "labram": []}
    keep_tr  = {"laya": [], "labram": [], "W": [], "l": []}
    with h5py.File(h5_path, "r") as hf:
        for ci, (seq_id, lab, emb) in enumerate(iter_chunks(index, args.limit)):
            eeg      = hf[f"/recordings/{seq_id}/data"][:]
            ch_names = hf[f"/recordings/{seq_id}/channels"][:]

            Za = np.asarray(emb, dtype=np.float64)          # (160, 384)
            la = tok_labels_repeat(lab, len(Za))
            Zb, starts = labram_embeddings(model, device, eeg, ch_names,
                                           args.sfreq, args.bipolar_map,
                                           args.batch_size)
            Zb = Zb.astype(np.float64)                      # (151, 200)
            lb = tok_labels_windows(lab, starts, LABRAM_PATCH_SZ, LABRAM_SFREQ)

            if not warned:
                print(f"  shapes: Laya {Za.shape} @ {len(Za)/16:.1f} Hz, "
                      f"LaBraM {Zb.shape} @ {len(Zb)/16:.1f} Hz, "
                      f"{len(decode_names(ch_names))} channels")
                warned = True

            if args.tracking and len(keep_tr["W"]) < args.auc_max_chunks:
                # One row per SECOND, which is also the native label rate. Each
                # model gets exactly the same 1 s of signal: Laya's 10 tokens of
                # 0.1 s concatenated, LaBraM's single 1 s window. Without this
                # LaBraM predicts descriptors of the very samples it saw while
                # Laya is asked about 10x more signal than its token covers, and
                # the R2 gap is receptive field rather than training objective.
                nsec = len(lab)
                oky  = len(Za) == nsec * 10
                idxb = np.searchsorted(starts, np.arange(nsec) * LABRAM_SFREQ)
                okb  = idxb.max() < len(Zb)
                if oky and okb:
                    keep_tr["laya"].append(
                        Za.reshape(nsec, -1).astype(np.float32))
                    keep_tr["labram"].append(Zb[idxb].astype(np.float32))
                    keep_tr["W"].append(
                        _zscore(waveform_features(
                            eeg, args.sfreq, np.arange(nsec) + 0.5)).astype(np.float32))
                    keep_tr["l"].append(np.asarray(lab))

            row = dict(seq_id=seq_id, n_trans=int(transitions(lab).size),
                       n_pos=int((lab == 1).sum()))
            for tag, Z, l in [("laya", Za, la), ("labram", Zb, lb)]:
                hz = len(Z) / 16.0
                (row[f"pc1t_{tag}"], row[f"evr1_{tag}"],
                 row[f"timer2_{tag}"]) = positional_stats(Z)
                if ((args.state_auc or args.tracking)
                        and len(keep[tag]) < args.auc_max_chunks):
                    keep[tag].append(_zscore(Z).astype(np.float32))
                    keep_lab[tag].append(np.asarray(l))
                    if args.tracking:
                        ctr = ((np.arange(len(Z)) + 0.5) / hz if tag == "laya"
                               else (starts + LABRAM_PATCH_SZ / 2) / LABRAM_SFREQ)
                        keep_w[tag].append(
                            _zscore(waveform_features(eeg, args.sfreq, ctr))
                            .astype(np.float32))
                if args.smooth > 1:
                    ker = np.ones(args.smooth) / args.smooth
                    Z = np.apply_along_axis(
                        lambda v: np.convolve(v, ker, mode="same"), 0, Z)
                a, a0 = pca_state_auc(Z, l, seed=ci)
                q7, q7n, q7j = within_state_consistency(Z, l, seed=ci)
                row[f"eta2_{tag}"], row[f"eta2n_{tag}"] = q7, q7n
                row[f"jit_{tag}"] = q7j
                e5, s5, tr5 = m5_split_error(Z, l, hz)
                row[f"m5err_{tag}"]   = e5
                row[f"m5split_{tag}"] = s5
                row[f"m5trans_{tag}"] = tr5
                row[f"pcauc_{tag}"], row[f"pcauc0_{tag}"] = a, a0
                variants = [("", Z), ("_pc3", top3(Z))]
                if args.detrend:
                    Zd = detrend_tokens(Z, args.detrend)
                    variants += [("_dt", Zd), ("_dtpc3", top3(Zd))]
                for sp, Zs in variants:
                    p = m1_transition_percentile(Zs, l, args.window, args.tol)
                    row[f"m1_{tag}{sp}"] = float(np.mean(p)) if p else np.nan
                    e, c = m2_localisation(Zs, l, args.window, hz)
                    row[f"err_{tag}{sp}"], row[f"chance_{tag}{sp}"] = e, c
                row[f"ceil_{tag}"] = m1_ceiling(len(Z), args.window, args.tol)
            rows.append(row)

            if (ci + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  {ci+1} chunks  {el:.0f}s  ({el/(ci+1):.2f} s/chunk)")

    if args.state_auc:
        print("\nM3  held-out state-occupancy AUC (chance exactly 0.500)")
        print("    Does the embedding sit somewhere DIFFERENT during the event?")
        print("    The TIME ONLY row is the clock's own score: what you get from")
        print("    token index alone, knowing nothing about the EEG. In these")
        print("    Read it as a control, not a subtraction: if it is near 0.5 the")
        print("    clock explains nothing and the minus-clock rows are a floor only,")
        print("    since removing a cubic also eats a genuine step.")

        def _timebasis(l):
            t = np.linspace(-1.0, 1.0, len(l))
            return np.vstack([t, t ** 2, t ** 3]).T.astype(np.float32)

        arms = [("Laya", keep["laya"], keep_lab["laya"], False),
                ("LaBraM", keep["labram"], keep_lab["labram"], False),
                ("TIME ONLY (clock)",
                 [_timebasis(l) for l in keep_lab["laya"]], keep_lab["laya"], False),
                ("Laya  minus clock", keep["laya"], keep_lab["laya"], True),
                ("LaBraM minus clock", keep["labram"], keep_lab["labram"], True)]
        for nm, Zs, ls, dt in arms:
            if not Zs:
                continue
            Za = [detrend_tokens(Z, 3).astype(np.float32) for Z in Zs] if dt else Zs
            pooled, per = state_auc(Za, ls)
            nulls = [state_auc(Za, ls, seed=s_, shift_null=True)[0] for s_ in range(5)]
            print(f"  {nm:20s} n={len(Za):5d}  pooled AUC={pooled:.4f}  "
                  f"per-chunk median={np.median(per) if per.size else float('nan'):.4f}  "
                  f"shift-null={np.median(nulls):.4f}")

    df = pd.DataFrame(rows)
    if args.tracking:
        print("\nM6  waveform tracking vs state tracking, both at 32 PCs")
        print("    One row per second. Each model sees the SAME 1 s of signal:")
        print("    Laya's 10 x 0.1 s tokens concatenated, LaBraM's one 1 s window.")
        print("    R2 = how much of the 1 s descriptors the embedding predicts;")
        print("    AUC = how well it separates the annotated state.")
        print(f"    {'':18s} {'stateAUC':>9s} " +
              " ".join(f"{f:>7s}" for f in WAVE_FEATS) + f" {'meanR2':>7s}")
        n_tr = len(keep_tr["W"])
        for tag, nm in [("laya", "Laya"), ("labram", "LaBraM")]:
            if n_tr < 5:
                print(f"    {nm}: too few aligned chunks ({n_tr})"); continue
            auc, r2 = tracking_scores(keep_tr[tag], keep_tr["W"], keep_tr["l"])
            print(f"    {nm:18s} {auc:9.4f} " +
                  " ".join(f"{v:7.3f}" for v in r2) + f" {np.mean(r2):7.3f}")
        print(f"    (n={n_tr} chunks x {len(keep_tr['l'][0]) if n_tr else 0} s)")

    print("\nM7  is each state held at a CONSISTENT colour? (the 3 PCs shown)")
    for tag, nm in [("laya", "Laya  "), ("labram", "LaBraM")]:
        e  = df[f"eta2_{tag}"].to_numpy(float)
        en = df[f"eta2n_{tag}"].to_numpy(float)
        j  = df[f"jit_{tag}"].to_numpy(float)
        m  = np.isfinite(e) & np.isfinite(en) & np.isfinite(j)
        print(f"  {nm:26s} n={int(m.sum()):5d}  between-state var={np.median(e[m]):.4f}  "
              f"(null {np.median(en[m]):.4f})  within-state jitter={np.median(j[m]):.4f}")
    mm = np.isfinite(df["jit_laya"]) & np.isfinite(df["jit_labram"])
    jl, jb = df["jit_laya"][mm].to_numpy(float), df["jit_labram"][mm].to_numpy(float)
    if mm.sum() > 5:
        st = stats.wilcoxon(jl, jb, alternative="less")
        print(f"  {'paired jitter':26s} n={int(mm.sum()):5d}  Laya {np.median(jl):.4f}  "
              f"LaBraM {np.median(jb):.4f}  Laya steadier on {100*np.mean(jl<jb):5.1f}%  "
              f"p={st.pvalue:.3e}")

    print("\nM5  does the dominant split of the 3 PCs land on the boundary? (s)")
    print("    Null pairs each chunk's split with OTHER chunks' boundaries, so")
    print("    the shared centre bias of both is preserved and only the")
    print("    correspondence is broken.")
    for tag, nm in [("laya", "Laya  "), ("labram", "LaBraM")]:
        e  = df[f"m5err_{tag}"].to_numpy(float)
        sp = df[f"m5split_{tag}"].to_numpy(float)
        trs = [np.fromstring(x, sep=";") if isinstance(x, str) and x else np.array([])
               for x in df[f"m5trans_{tag}"]]
        m  = np.isfinite(e) & np.isfinite(sp)
        idx = np.where(m)[0]
        rng = np.random.default_rng(0)
        perm = []
        for _ in range(200):
            j = rng.permutation(idx)
            perm.append(np.median([np.min(np.abs(trs[b] - sp[a]))
                                   for a, b in zip(idx, j) if trs[b].size]))
        perm = np.array(perm)
        obs  = np.median(e[m])
        print(f"  {nm:26s} n={int(m.sum()):5d}  err={obs:.3f}s  "
              f"null={np.median(perm):.3f}s  "
              f"p={(1 + np.sum(perm <= obs)) / (len(perm) + 1):.3e}")
    mm = np.isfinite(df["m5err_laya"]) & np.isfinite(df["m5err_labram"])
    el, eb = df["m5err_laya"][mm].to_numpy(float), df["m5err_labram"][mm].to_numpy(float)
    if mm.sum() > 5:
        st = stats.wilcoxon(el, eb, alternative="less")
        print(f"  {'paired':26s} n={int(mm.sum()):5d}  Laya {np.median(el):.3f}s  "
              f"LaBraM {np.median(eb):.3f}s  Laya better on {100*np.mean(el<eb):5.1f}%  "
              f"p={st.pvalue:.3e}")

    print("\nM4  state separability on the 3 PCs the figure colours with")
    print("    Circular by design; the null is built the same way, so read the margin.")
    for tag, nm in [("laya", "Laya  "), ("labram", "LaBraM")]:
        a  = df[f"pcauc_{tag}"].to_numpy(float)
        a0 = df[f"pcauc0_{tag}"].to_numpy(float)
        m  = np.isfinite(a) & np.isfinite(a0)
        st = stats.wilcoxon(a[m], a0[m], alternative="greater") if m.sum() > 5 else None
        print(f"  {nm:26s} n={int(m.sum()):5d}  AUC={np.median(a[m]):.4f}  "
              f"null={np.median(a0[m]):.4f}  "
              f"margin={np.median(a[m] - a0[m]):+.4f}  "
              f"p={st.pvalue:.3e}" if st else f"  {nm}: too few")
    ml = np.isfinite(df["pcauc_laya"]) & np.isfinite(df["pcauc_labram"])
    dl = (df["pcauc_laya"] - df["pcauc0_laya"])[ml].to_numpy(float)
    db = (df["pcauc_labram"] - df["pcauc0_labram"])[ml].to_numpy(float)
    if ml.sum() > 5:
        st = stats.wilcoxon(dl, db, alternative="greater")
        print(f"  {'paired margin':26s} n={int(ml.sum()):5d}  Laya {np.median(dl):+.4f}  "
              f"LaBraM {np.median(db):+.4f}  Laya better on {100*np.mean(dl>db):5.1f}%  "
              f"p={st.pvalue:.3e}")

    df.to_csv(f"{out}.csv", index=False)
    print(f"\n{len(df)} chunks with a label transition -> {out}.csv "
          f"({time.time()-t0:.0f}s)")
    aggregate(df, f"{out}.png")


if __name__ == "__main__":
    main()
