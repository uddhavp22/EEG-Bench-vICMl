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
2. The EEG is read from the LaBraMModel H5 (200 Hz, microvolts), NOT from the
   LeJEPAClinical H5. The latter applies defossez scaling at write time
   (utils_2.py:516) and CLIPS TO +/-20, which flattens exactly the
   high-amplitude ictal and artifact segments these metrics are about. Every
   result produced before 2026-07-30 used the clipped H5 for the LaBraM arm and
   should be discarded. Laya's arm is unaffected: its embeddings come from the
   .index.json cache, not from the H5. `signal_fingerprint` prints the units and
   the clip fraction on the first chunk so this cannot recur silently.
3. The index and the H5 are independent dataset builds that each number
   recordings from scratch, so seq_id agreement is assumed, not guaranteed. The
   per-chunk H5 label is compared against the index label and the run ABORTS on
   any mismatch rather than reporting misaligned pairs.

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


# Set from --pc-norm in main(). "plot" reproduces the figure's colour space and
# is the default, because the figure is what the rebuttal defends.
PC_NORM = "plot"


def plot_norm(P):
    """The figure's exact per-component normalisation, from
    plot_chunk_with_embeddings in analyze_embeddings_rebuttals.ipynb:

        p_low, p_high = percentile(P, 2, axis=0), percentile(P, 98, axis=0)
        P = clip((P - p_low) / (p_high - p_low), 0, 1)

    This rescales EVERY component to the same [0,1] span before it becomes an
    RGB channel, so PC3 at 5% of the variance carries the same visual weight as
    PC1 at 50%. Metrics on RAW PC coordinates are dominated by PC1 and therefore
    do not measure what the panel shows. With Laya's PC1 at |corr(PC1,t)|=0.815,
    raw-space change-point statistics track the temporal drift in PC1 rather
    than a state step living in a lower-variance component.

    The 2/98 clip also caps single-token excursions, at ~3 tokens per end of a
    160-token chunk. Applied identically to both models.
    """
    lo  = np.percentile(P, 2, axis=0, keepdims=True)
    hi  = np.percentile(P, 98, axis=0, keepdims=True)
    rng = hi - lo
    rng[rng == 0] = 1e-8
    return np.clip((P - lo) / rng, 0.0, 1.0)


def top3(Z):
    """The 3 PCs the panel actually colours with, in the panel's own colour
    space (see plot_norm and --pc-norm). If the effect is in the full embedding
    but absent here, the number and the picture are measuring different things
    and the rebuttal must say which it quotes."""
    Zc = Z - Z.mean(axis=0)
    P  = Zc @ np.linalg.svd(Zc, full_matrices=False)[2][:3].T
    return plot_norm(P) if PC_NORM == "plot" else P


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
    """DEPRECATED, kept only behind --null uniform for reproducing old numbers.

    Draws cut positions uniformly. Its docstring used to claim it "keeps the
    temporal shape and moves only the location", and that is FALSE: for k=1 it
    does not preserve the class fraction, and for k>1 it does not preserve the
    run lengths. eta2 is maximised near a balanced split, so if real boundaries
    sit off-centre -- short artifact bursts, seizures that run to the chunk end
    -- a uniform relocation is more balanced on average than the truth and wins
    for purely combinatorial reasons. Every negative eta2 margin this script
    reported before 2026-07-30 (artifact Laya -0.0637 raw, -0.0475 plot) is
    suspect for exactly this reason. Use `sample_null_labels` instead.
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


def sample_null_labels(pool, self_idx, k, n_null, rng):
    """Null labels drawn from OTHER chunks' REAL annotations.

    This is the same construction M5's null already uses, and it is the only
    one that preserves the label geometry eta2 and AUC are sensitive to. A
    synthesised null has to get class fraction, run-length distribution,
    boundary position and transition direction all right simultaneously; a real
    annotation from another chunk has them right by construction, and breaks
    only the correspondence between the labels and THIS chunk's signal, which is
    precisely the thing being tested.

    Restricted to patterns with the same transition count where the pool allows,
    so the null matches this chunk's k as well as the empirical run lengths. The
    chunk's own pattern is excluded, which matters at n=216 where a self-draw is
    otherwise ~0.5% of the null.

    Returns a list of n_null second-resolution label arrays.
    """
    cand = pool.by_k.get(int(k))
    if cand is None or len(cand) < 5:
        cand = pool.all_idx
    cand = [j for j in cand if j != self_idx]
    if not cand:
        return []
    draw = rng.choice(len(cand), size=n_null, replace=True)
    return [pool.labels[cand[d]] for d in draw]


class LabelPool:
    """Every real label pattern in the split, indexed by transition count.

    Built in a cheap pre-pass over the index that touches only the labels .npy,
    never the embeddings, so it costs a fraction of a second even at n=5058.
    """

    def __init__(self, labels):
        self.labels  = labels
        self.all_idx = list(range(len(labels)))
        self.by_k    = {}
        for j, l in enumerate(labels):
            self.by_k.setdefault(int(transitions(l).size), []).append(j)

    def __len__(self):
        return len(self.labels)

    def summary(self):
        ks   = sorted(self.by_k)
        frac = np.array([float(np.mean(np.asarray(l) == 1)) for l in self.labels])
        return (f"{len(self.labels)} real label patterns; "
                f"transition counts {{{', '.join(f'{k}:{len(self.by_k[k])}' for k in ks)}}}; "
                f"positive fraction median={np.median(frac):.3f} "
                f"IQR=[{np.percentile(frac, 25):.3f}, {np.percentile(frac, 75):.3f}]")


def collect_label_pool(index, limit=None):
    """Pre-pass for LabelPool. Reads only shard['labels'], never embeddings."""
    out = []
    for shard in index["shards"]:
        labels = np.load(shard["labels"], mmap_mode="r")
        for i in range(len(labels)):
            lab = np.asarray(labels[i])
            if transitions(lab).size == 0:
                continue
            out.append(lab.copy())
            if limit and len(out) >= limit:
                return LabelPool(out)
    return LabelPool(out)


def rec_id(seq_id):
    """Recording identity from a seq_id.

    utils_2.py:240 writes `recording_{idx:04d}_{i:03d}`, so chunks of one
    recording share the first field. Chunk-level p-values treat all 5058
    artifact chunks as independent when they come from far fewer recordings;
    everything reported at chunk level is pseudoreplicated without this.
    """
    s = str(seq_id)
    p = s.split("_")
    return "_".join(p[:2]) if len(p) >= 3 and p[0] == "recording" else s


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


def _far_from_transitions(lab, guard):
    """Mask of tokens at least `guard` tokens away from every label change.

    LaBraM's 1 s window at 0.1 s stride means a true step is smeared across
    ~10 consecutive windows. Excluding only the single label-crossing step
    leaves that entire ramp counted as WITHIN-state variation and also drags
    the two centroids toward each other, so the un-guarded jitter penalises the
    baseline for its own receptive field. Guarding both models identically in
    seconds removes that.
    """
    lab = np.asarray(lab)
    if guard <= 0:
        return np.ones(len(lab), dtype=bool)
    keep = np.ones(len(lab), dtype=bool)
    for t in transitions(lab):
        keep[max(0, t - guard):min(len(lab), t + guard)] = False
    return keep


def within_state_consistency(Z, lab, null_labs=(), guard=0):
    """M7. Does each state get a CONSISTENT colour for its whole duration?

    AUC ranks tokens along one direction, so a representation that jumps around
    inside a state pays nothing as long as that one projection still separates.
    This is what "LaBraM changes at the boundary but is all over the place
    within the seizure" means, and no other metric here sees it.

      eta2     fraction of 3-PC variance that is BETWEEN states rather than
               within them. High = each state holds one colour.
      jit      mean within-state token step / distance between state centroids.
      jitfree  mean within-state token step / the chunk's own 3-PC RMS radius.

    WHY BOTH JITTER FORMS ARE REPORTED. `jit`'s denominator is defined by the
    labels, so it conflates two different things: how smooth the trajectory is,
    and how far apart the labelled states sit. A pure temporal ramp gets a
    flattering `jit` for free, because when the labels form early/late blocks
    the centroid distance is large no matter what the embedding encodes. Worse,
    the obvious control does not catch it: for a linear ramp split at fraction
    f the two centroids sit at f/2 and (1+f)/2, so the separation is 1/2 for
    EVERY f and a relocated-label jitter is identical to the real one. `jitfree`
    replaces the denominator with a label-free scale, which makes it an honest
    smoothness statistic that claims nothing about states. Quote `jitfree` for
    "Laya's trajectory is smoother"; quote `jit` only alongside it.

    eta2 gets a null drawn from other chunks' real annotations (see
    sample_null_labels). The nulls are pre-converted to this model's token grid
    by the caller, since Laya and LaBraM have different token counts.

    Returns a dict.
    """
    P    = top3(Z)
    lab  = np.asarray(lab)
    out  = dict(eta2=np.nan, eta2n=np.nan, eta2rank=np.nan, jit=np.nan,
                jitfree=np.nan, step=np.nan, sep=np.nan, spread=np.nan,
                n_used=0)

    def _stats(l):
        l    = np.asarray(l)
        keep = _far_from_transitions(l, guard)
        if keep.sum() < 4:
            return None
        Pk, lk = P[keep], l[keep]
        if lk.min() == lk.max():
            return None
        sst = float(((Pk - Pk.mean(axis=0)) ** 2).sum())
        if sst <= 0:
            return None
        ssw  = sum(float(((Pk[lk == v] - Pk[lk == v].mean(axis=0)) ** 2).sum())
                   for v in (0, 1))
        eta2 = 1.0 - ssw / sst
        sep  = float(np.linalg.norm(Pk[lk == 1].mean(axis=0) -
                                    Pk[lk == 0].mean(axis=0)))
        return eta2, sep, keep, lk

    real = _stats(lab)
    if real is None:
        return out
    eta2, sep, keep, _ = real

    # A step counts only when BOTH endpoints survive the guard and share a
    # label, so no step spanning a transition or its ramp is ever included.
    ok   = keep[1:] & keep[:-1] & (lab[1:] == lab[:-1])
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if not ok.any():
        return out
    step_mean = float(step[ok].mean())
    spread    = float(np.sqrt(((P[keep] - P[keep].mean(axis=0)) ** 2)
                              .sum(axis=1).mean()))

    nl = []
    for nlab in null_labs:
        s = _stats(nlab)
        if s is not None:
            nl.append(s[0])

    out.update(eta2=eta2,
               eta2n=float(np.mean(nl)) if nl else np.nan,
               # Rank of the truth among the nulls, uniform on [0,1] under the
               # null. Immune to the ceiling compression that makes the MARGIN
               # unreadable on a smooth trajectory: when every relocated split
               # already scores near the maximum, a true effect has almost no
               # headroom left to show as a difference, but it can still sit
               # above nearly all of the nulls. Aggregate this against 0.5.
               eta2rank=float(np.mean(np.asarray(nl) < eta2)) if nl else np.nan,
               jit=step_mean / sep if sep > 0 else np.nan,
               jitfree=step_mean / spread if spread > 0 else np.nan,
               step=step_mean, sep=sep, spread=spread, n_used=int(keep.sum()))
    return out


def _group_folds(recs, n_folds, rng):
    """Folds that never split a recording across train and test."""
    uniq = sorted(set(recs))
    rng.shuffle(uniq)
    assign = {r: i % n_folds for i, r in enumerate(uniq)}
    return np.array([assign[r] for r in recs]), len(uniq)


def onset_termination_reversal(Zs, labs, recs, win=20, n_folds=5, seed=0):
    """M8. Does the embedding REVERSE at a termination, or keep going?

    This is the one cheap test that a clock cannot pass. Learn the direction the
    embedding moves at seizure ONSET, on one set of recordings. Then look at
    what it does at seizure TERMINATION on recordings never seen during the fit.

      a state representation  moves back toward the interictal region, so the
                              projection onto the onset direction is NEGATIVE
      a clock                 moves forward in time at both events, so the
                              projection stays POSITIVE

    Chance is 50% negative. Nothing about smoothness, PC1 variance share or
    boundary sharpness can produce reversal; only a code whose position depends
    on the STATE rather than on elapsed time can.

    Deltas are unit-normalised before averaging, so a few high-amplitude events
    cannot define the direction, and the fit is on the full embedding in its
    native space -- NOT per-chunk PCA, whose component signs and axis order are
    arbitrary from chunk to chunk and would make cross-chunk averaging
    meaningless.

    The onset->onset row is the control: if held-out onsets do not align with
    the training onset direction, there is no consistent direction to speak of
    and the reversal row carries no information either way.

    `win` is clamped to n/2 so it can never span the whole chunk. In native mode
    a chunk is only 16 tokens, so an unclamped win=20 made every delta
    mean(everything after) - mean(everything before). That does NOT bias the
    test toward the clock answer -- a pure state code still gives -1.0, since a
    termination's delta is exactly the negative of an onset's -- but it does mix
    in the far side of a second transition when a chunk contains both, and it
    makes the printed window size a lie.

    Returns {name: (frac_positive, mean_projection, n_events, per_recording
    mean projections)}. The per-recording list is what the p-value is computed
    on: several events from one recording share a patient and a montage, so an
    event-level sign test is pseudoreplicated in exactly the way M7 was.
    """
    rng = np.random.default_rng(seed)

    events = {"onset": [], "term": []}          # (rec, unit delta)
    for Z, lab, rc in zip(Zs, labs, recs):
        Z   = np.asarray(Z, dtype=np.float64)
        lab = np.asarray(lab)
        n   = len(Z)
        w   = max(2, min(win, n // 2))
        for t in transitions(lab):
            if t < 2 or t > n - 2:
                continue
            a, b = Z[max(0, t - w):t], Z[t:min(n, t + w)]
            if len(a) < 2 or len(b) < 2:
                continue
            d  = b.mean(axis=0) - a.mean(axis=0)
            nd = float(np.linalg.norm(d))
            if nd <= 0:
                continue
            kind = "onset" if lab[t] == 1 else "term"
            events[kind].append((rc, d / nd))

    if len(events["onset"]) < 10 or len(events["term"]) < 10:
        return {}

    all_recs = [r for k in events for r, _ in events[k]]
    fold_of  = {}
    uniq     = sorted(set(all_recs))
    rng.shuffle(uniq)
    for i, r in enumerate(uniq):
        fold_of[r] = i % n_folds

    out = {}
    for fit_on, test_on in [("onset", "onset"), ("onset", "term"),
                            ("term", "onset")]:
        proj, by_rec = [], {}
        for f in range(n_folds):
            tr = [d for r, d in events[fit_on] if fold_of[r] != f]
            te = [(r, d) for r, d in events[test_on] if fold_of[r] == f]
            if len(tr) < 5 or not te:
                continue
            u  = np.mean(tr, axis=0)
            nu = float(np.linalg.norm(u))
            if nu <= 0:
                continue
            u = u / nu
            for r, d in te:
                v = float(d @ u)
                proj.append(v)
                by_rec.setdefault(r, []).append(v)
        if proj:
            p = np.array(proj)
            out[f"{fit_on}->{test_on}"] = (
                float(np.mean(p > 0)), float(np.mean(p)), len(p),
                [float(np.mean(v)) for v in by_rec.values()])
    return out


def step_profile(Zs, labs, hz, max_s=6.0, n_bins=12):
    """M10. Step size as a function of distance from the annotated boundary.

    THIS IS THE TEST M8 CANNOT DO. M8 assumes the state sequence is
    interictal -> ictal -> interictal, so a state code must come back. If the
    real sequence is preictal -> ictal -> POSTICTAL, and postictal is its own
    clinical state (suppression, slowing), then a genuine state representation
    should NOT reverse either, and "no reversal" stops distinguishing a clock
    from a three-state progression.

    Speed does distinguish them:

      a clock                 advances at a constant rate, so the normalised
                              step is FLAT in distance from the boundary
      a state progression     moves quickly between states and slowly inside
                              one, so the step PEAKS at the boundary, even if
                              the transition is gradual and smeared over
                              seconds -- which is exactly the regime an
                              argmax test like M5 or M2 is blind to

    Steps are normalised by each chunk's own RMS radius before pooling, so a
    chunk with a large embedding excursion cannot dominate the profile.

    Returns (bin_centres_in_seconds, mean_normalised_step, n_per_bin).
    """
    edges = np.linspace(0.0, max_s, n_bins + 1)
    acc   = [[] for _ in range(n_bins)]
    for Z, lab in zip(Zs, labs):
        Z   = np.asarray(Z, dtype=np.float64)
        lab = np.asarray(lab)
        tr  = transitions(lab)
        if tr.size == 0 or len(Z) < 4:
            continue
        P   = Z - Z.mean(axis=0)
        rad = float(np.sqrt((P ** 2).sum(axis=1).mean()))
        if rad <= 0:
            continue
        step = np.linalg.norm(np.diff(P, axis=0), axis=1) / rad
        # distance from the MIDPOINT of each step to the nearest transition
        mid  = np.arange(len(step)) + 0.5
        dist = np.min(np.abs(mid[:, None] - tr[None, :]), axis=1) / hz
        idx  = np.digitize(dist, edges) - 1
        for b, s in zip(idx, step):
            if 0 <= b < n_bins:
                acc[b].append(float(s))
    centres = 0.5 * (edges[:-1] + edges[1:])
    means   = np.array([np.mean(a) if a else np.nan for a in acc])
    counts  = np.array([len(a) for a in acc])
    return centres, means, counts


def _ridge_auc(Xtr, ytr, Xte, yte, lam=1.0):
    """Held-out AUC from ridge regression onto +/-1 labels.

    Ridge rather than logistic to keep this dependency-free and deterministic;
    for a rank statistic the two are near-identical, and AUC is invariant to any
    monotone transform of the score.
    """
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    A  = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B  = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    t  = np.where(np.asarray(ytr).astype(bool), 1.0, -1.0)
    G  = A.T @ A + lam * np.eye(A.shape[1])
    w  = np.linalg.solve(G, A.T @ t)
    return _auc(B @ w, yte)


def conditional_state_decoder(Es, Ws, Ls, recs, k=32, n_folds=5, seed=0,
                              permute=False):
    """M9. Does the embedding carry state information BEYOND clock and waveform?

    This is the test that speaks to `results.tex:42` directly. Everything is at
    1 Hz, whole recordings are held out, and the four arms are nested:

        TIME              t, t^2, t^3 within the chunk -- the clock alone
        WAVE              the 7 signal descriptors -- "surface-level statistics"
        TIME+WAVE         both
        TIME+WAVE+EMB     both, plus the embedding at k global PCs

    The number that matters is the LAST ROW MINUS THE THIRD. If adding the
    embedding to a model that already knows elapsed time and the waveform does
    not improve held-out AUC, then whatever the embedding encodes about state is
    already available from the clock and from bandpower, and "organized around
    semantically meaningful brain states rather than surface-level signal
    statistics" is not supported. If it does improve, that increment is the
    quantitative claim, and it is immune to every confound in M1-M7 because
    those confounds are IN the baseline.

    Recording-level grouping is not optional here: chunks of one recording share
    a patient and an event, so a chunk-level split lets the model memorise the
    recording and every arm saturates.

    permute=True shuffles the EMBEDDING between chunks and leaves labels, time
    and waveform untouched. That is the right null for an INCREMENT: it holds
    the TIME+WAVE arm fixed and measures what adding 32 uninformative columns
    buys through overfitting alone. Permuting the labels instead would destroy
    the baseline as well, and the resulting "gain" would answer a different
    question.

    Returns {arm: auc}.
    """
    rng = np.random.default_rng(seed)
    Ls  = [np.asarray(l) for l in Ls]
    if permute:
        perm = rng.permutation(len(Es))
        Es   = [Es[j] if len(Es[j]) == len(Ls[i]) else Es[i]
                for i, j in enumerate(perm)]

    def tbasis(n):
        t = np.linspace(-1.0, 1.0, n)
        return np.vstack([t, t ** 2, t ** 3]).T

    fold, n_rec = _group_folds(list(recs), n_folds, rng)
    if n_rec < n_folds:
        return {}

    arms  = ["TIME", "WAVE", "TIME+WAVE", "TIME+WAVE+EMB"]
    score = {a: [] for a in arms}
    for f in range(n_folds):
        tr = [i for i in range(len(Ls)) if fold[i] != f]
        te = [i for i in range(len(Ls)) if fold[i] == f]
        if not tr or not te:
            continue
        ytr = np.concatenate([Ls[i] for i in tr])
        yte = np.concatenate([Ls[i] for i in te])
        if ytr.min() == ytr.max() or yte.min() == yte.max():
            continue

        Etr = np.concatenate([Es[i] for i in tr]).astype(np.float64)
        mu  = Etr.mean(axis=0)
        Vt  = np.linalg.svd(Etr - mu, full_matrices=False)[2][:k]
        Ptr = (Etr - mu) @ Vt.T
        Pte = np.concatenate([(np.asarray(Es[i], np.float64) - mu) @ Vt.T
                              for i in te])

        Ttr = np.concatenate([tbasis(len(Ls[i])) for i in tr])
        Tte = np.concatenate([tbasis(len(Ls[i])) for i in te])
        Wtr = np.concatenate([Ws[i] for i in tr]).astype(np.float64)
        Wte = np.concatenate([Ws[i] for i in te]).astype(np.float64)

        feats = {"TIME": (Ttr, Tte), "WAVE": (Wtr, Wte),
                 "TIME+WAVE": (np.hstack([Ttr, Wtr]), np.hstack([Tte, Wte])),
                 "TIME+WAVE+EMB": (np.hstack([Ttr, Wtr, Ptr]),
                                   np.hstack([Tte, Wte, Pte]))}
        for a in arms:
            X1, X2 = feats[a]
            v = _ridge_auc(X1, ytr, X2, yte)
            if np.isfinite(v):
                score[a].append(v)
    return {a: float(np.mean(v)) for a, v in score.items() if v}


def pc_dimension_sweep(Es, Ls, recs, ks=(1, 2, 3, 5, 8, 16, 32, 64, 128),
                       n_folds=5, seed=0):
    """M11. How many UNSUPERVISED dimensions does the state code live in?

    This is the metric that carries the word "organization", and it is not the
    linear probe restated. The probe is supervised with D free parameters: it
    says the state information IS PRESENT, which almost any competent encoder
    achieves, and says nothing about how the space is arranged. Here the basis
    is PCA fit on TRAIN embeddings with the labels never shown to it, so asking
    how well state decodes from the first k components asks whether state is
    aligned with the DOMINANT axes of variation.

    That is exactly the claim Figure 2 makes. The figure colours the top 3 PCs
    and the colour tracks state, so the quantitative version of the figure is
    "state decodes from the first 3 unsupervised PCs", measured over every chunk
    in the split instead of a handful of panels.

    Read the curve, not one number:

        AUC saturating at small k   state is a dominant axis -> organised
        AUC climbing until large k  state is present but spread thin across
                                    many low-variance directions -> decodable
                                    but NOT organised

    k95 is the smallest k reaching 95% of that model's own gain over 0.5, so it
    compares SHAPE and is not confounded by one model having a higher ceiling
    or a wider embedding (Laya D=384 vs LaBraM D=200).

    CAVEAT, and it must be quoted with the result: PCA is unsupervised but it is
    not assumption-free. It ranks directions by variance, so a model whose
    largest variance component is something irrelevant but huge (drift, a DC
    offset, an amplitude envelope) is penalised for reasons unrelated to state.
    That is a real confound for LaBraM, whose tokens are not centred per chunk
    here. It is a statement about the representation as it comes out of the
    model, which is the right unit for a claim about organisation, but it is not
    a statement that state is absent from the model.

    Returns {"auc": {k: auc}, "k95": int_or_nan, "kmax": int}.
    """
    rng = np.random.default_rng(seed)
    Ls  = [np.asarray(l) for l in Ls]
    fold, n_rec = _group_folds(list(recs), n_folds, rng)
    if n_rec < n_folds:
        return {}
    D    = Es[0].shape[1]
    ks   = [k for k in ks if k <= D]
    if not ks:
        return {}
    kmax = max(ks)
    acc  = {k: [] for k in ks}
    for f in range(n_folds):
        tr = [i for i in range(len(Ls)) if fold[i] != f]
        te = [i for i in range(len(Ls)) if fold[i] == f]
        if not tr or not te:
            continue
        ytr = np.concatenate([Ls[i] for i in tr])
        yte = np.concatenate([Ls[i] for i in te])
        if ytr.min() == ytr.max() or yte.min() == yte.max():
            continue
        Etr = np.concatenate([Es[i] for i in tr]).astype(np.float64)
        mu  = Etr.mean(axis=0)
        # one SVD per fold; every k is a prefix of the same basis, so the
        # nested structure is exact rather than k separate decompositions
        Vt  = np.linalg.svd(Etr - mu, full_matrices=False)[2][:kmax]
        Ptr = (Etr - mu) @ Vt.T
        Pte = np.concatenate([(np.asarray(Es[i], np.float64) - mu) @ Vt.T
                              for i in te])
        for k in ks:
            v = _ridge_auc(Ptr[:, :k], ytr, Pte[:, :k], yte)
            if np.isfinite(v):
                acc[k].append(v)
    auc = {k: float(np.mean(v)) for k, v in acc.items() if v}
    if not auc:
        return {}
    best = max(auc.values())
    k95  = next((k for k in sorted(auc)
                 if auc[k] - 0.5 >= 0.95 * (best - 0.5)), float("nan"))
    return {"auc": auc, "k95": k95, "kmax": max(auc)}


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


def pca_state_auc(Z, lab, null_labs=()):
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

    THE MARGIN IS CEILING-COMPRESSED AND THE RANK IS NOT. On a smooth monotone
    trajectory every contiguous split separates well, so the null itself sits
    near the maximum (Laya artifact null = 0.920 vs LaBraM 0.760) and there is
    almost no headroom left for a real effect to appear as a difference. Laya's
    higher absolute AUC and its much smaller margin are then the same fact --
    smoothness -- reported twice, and the margin cannot be read as "less
    state-specific than LaBraM". The rank of the truth among the nulls is
    uniform on [0,1] under the null at ANY ceiling, so it is the comparable
    statistic across two models with different smoothness.

    Returns (auc, null_auc, rank).
    """
    P   = top3(Z)
    lab = np.asarray(lab)

    def _fit(l):
        l = np.asarray(l)
        if l.min() == l.max():
            return np.nan
        u = P[l == 1].mean(axis=0) - P[l == 0].mean(axis=0)
        nu = float(np.linalg.norm(u))
        return _auc(P @ (u / nu), l) if nu > 0 else np.nan

    real  = _fit(lab)
    nulls = np.array([v for v in (_fit(l) for l in null_labs) if np.isfinite(v)])
    if nulls.size == 0 or not np.isfinite(real):
        return real, np.nan, np.nan
    return real, float(nulls.mean()), float(np.mean(nulls < real))


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


def find_h5(task, model_tag="LaBraMModel"):
    """Locate the recordings H5.

    Default is the LaBraMModel build, NOT LeJEPAClinical. The LeJEPAClinical
    build applies defossez scaling at write time (utils_2.py:516): median
    subtracted, divided by the global IQR, then CLIPPED TO +/-20. Feeding that
    to LaBraM squashes exactly the high-amplitude ictal and artifact segments
    the metrics are about, and no downstream norm can undo a clip. The
    LaBraMModel build (utils_2.py:435-444) is filtered to 200 Hz and kept in
    microvolts, which is what LaBraM was trained on.

    Laya's embeddings come from the .index.json, not from here, so switching
    the H5 does not touch the Laya arm. The EEG read here is used for LaBraM's
    forward pass and for the M6 waveform descriptors.
    """
    from eeg_bench.config import get_config_value
    task = TASK_ALIASES.get(task, task)
    root = get_config_value("make_dataset") or os.path.join(
        get_config_value("data"), "make_dataset")
    hits = sorted(glob.glob(os.path.join(root, f"{task}_{model_tag}_*.h5")),
                  key=os.path.getsize, reverse=True)
    if hits:
        return hits[0]

    # Say what IS there. `root` comes from eeg_bench config and can resolve to a
    # RELATIVE path, so a miss may mean "wrong directory" rather than "not built".
    msg = [f"no H5 matching {os.path.join(root, f'{task}_{model_tag}_*.h5')}",
           f"  root resolved to {os.path.abspath(root)}"
           f"{'' if os.path.isdir(root) else '   <-- DOES NOT EXIST'}"]
    same_task = sorted(glob.glob(os.path.join(root, f"{task}_*.h5")))
    if same_task:
        msg.append(f"  builds present for {task}:")
        msg += [f"    {os.path.basename(h)}" for h in same_task]
    else:
        any_h5 = sorted(glob.glob(os.path.join(root, "*.h5")))[:15]
        msg.append("  no H5 for this task in that directory" if not any_h5
                   else "  directory holds: " + ", ".join(os.path.basename(h) for h in any_h5))
    msg.append("  pass --h5 /abs/path.h5 to bypass the lookup")
    raise FileNotFoundError("\n".join(msg))


def infer_sfreq(eeg, chunk_len_s):
    """Sampling rate from the chunk itself. The H5 stores no sfreq attribute,
    so a --sfreq flag is a silent footgun the moment the H5 tag changes."""
    return float(eeg.shape[1]) / float(chunk_len_s)


def signal_fingerprint(eeg):
    """Detect defossez-scaled input before it reaches LaBraM.

    Microvolt EEG: median|x| of order 1-50, max|x| in the hundreds, nothing
    pinned to a bound. Defossez output: unit-IQR and hard-clipped at +/-20, so
    `clipped` is strictly positive on any chunk containing a real event.
    """
    a  = np.abs(eeg)
    mx = float(a.max())
    # A clip shows up as mass pinned AT the bound with nothing above it. The
    # fraction of samples ABOVE 20 is not the test: microvolt EEG has a median
    # around 35 uV, so it sits above 20 most of the time and that reads as a
    # false alarm. The discriminative fact is max|x|, 478 uV here vs exactly
    # 20.0 for defossez output.
    return dict(median=float(np.median(a)), p99=float(np.percentile(a, 99)),
                max=mx, at_bound=float(np.mean(a >= mx - 1e-6)),
                looks_clipped=bool(mx <= 20.0001))


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


def labram_embeddings_native(model, device, eeg, ch_names, orig_sfreq,
                             bipolar_map):
    """LaBraM the way LaBraM is meant to run: ONE forward pass over the 16 s.

    WHY THIS ARM EXISTS. The sliding-window function above runs 151 independent
    single-patch forward passes. That is not a neutral choice: it removes all
    across-time attention and pins every window's time embedding to slot 0, so
    the baseline is denied the mechanism that would let it hold a state. It is
    also exactly what notebook cell 9 does, which means the published Figure 2
    shows LaBraM in this crippled configuration -- so the sliding arm stays the
    DEFAULT here, because the point of this analysis is to characterise the
    figure we actually printed.

    This arm answers the separate and fair question: with full context, does
    LaBraM organise by state? The output is (16, D) at 1 Hz -- one token per
    second, channel-averaged -- and the caller pools Laya to the same 16 points
    so neither model is scored at a resolution the other cannot reach.

    forward_features returns (1, C*n_patch, D) in CHANNEL-MAJOR order, hence
    reshape(C, n_patch, D) then mean over the channel axis. Getting that order
    backwards would silently transpose time into channels.
    """
    import torch
    from scipy.signal import resample
    from eeg_bench.models.clinical.LaBraM import utils as labram_utils

    if orig_sfreq != LABRAM_SFREQ:
        new_T  = int(round(eeg.shape[1] * LABRAM_SFREQ / orig_sfreq))
        eeg_rs = resample(eeg, new_T, axis=1)
    else:
        eeg_rs = np.asarray(eeg)

    C, T    = eeg_rs.shape
    n_patch = T // LABRAM_PATCH_SZ
    if n_patch < 2:
        raise ValueError(f"need >=2 whole 1 s patches, got {n_patch}")
    x = eeg_rs[:, :n_patch * LABRAM_PATCH_SZ]

    names       = map_channel_names(decode_names(ch_names), bipolar_map)
    input_chans = labram_utils.get_input_chans(names)

    with torch.no_grad():
        tok = (torch.from_numpy(np.ascontiguousarray(x)).float()
                 .reshape(1, C, n_patch, LABRAM_PATCH_SZ) / 100.0)
        pe  = model.forward_features(tok.to(device), input_chans=input_chans,
                                     return_patch_tokens=True)
        assert pe.shape[1] == C * n_patch, (
            f"token axis {pe.shape[1]} != C*n_patch {C}*{n_patch}; the reshape "
            "below would mix channels into time")
        Z = pe.reshape(C, n_patch, -1).mean(dim=0).cpu().numpy()

    starts = np.arange(n_patch, dtype=int) * LABRAM_PATCH_SZ
    return Z, starts


# ===========================================================================
# main
# ===========================================================================

def paired_report(df, col, name, alternative="less", fmt="{:.4f}", width=20):
    """Paired Laya-vs-LaBraM test at BOTH chunk and recording level.

    Chunk-level p-values here are pseudoreplicated and should not be quoted.
    The 5058 artifact chunks come from far fewer recordings; chunks inside one
    recording share a patient, a montage, an amplifier and often a single
    clinical event, so they are nowhere near independent draws. A Wilcoxon over
    them answers "is this true of the average CHUNK", with an effective n far
    below 5058, which is how a p of 1e-300 appears. Collapsing each recording to
    its median first answers "is this true of the average RECORDING", which is
    the claim the rebuttal actually needs. Quote the recording-level row.

    The descriptive win rate is unaffected by this and stays quotable.
    """
    from scipy import stats
    a_col, b_col = f"{col}_laya", f"{col}_labram"
    if a_col not in df or b_col not in df:
        return
    d = (df[["rec", a_col, b_col]].replace([np.inf, -np.inf], np.nan).dropna())
    if len(d) < 6:
        print(f"  {name:<{width}} too few usable chunks ({len(d)})")
        return
    levels = [("chunks", d[[a_col, b_col]]),
              ("recordings", d.groupby("rec")[[a_col, b_col]].median())]
    for i, (lvl, dd) in enumerate(levels):
        a, b = dd[a_col].to_numpy(float), dd[b_col].to_numpy(float)
        win  = np.mean(a < b) if alternative == "less" else np.mean(a > b)
        try:
            p = stats.wilcoxon(a, b, alternative=alternative).pvalue
        except ValueError:
            p = np.nan
        print(f"  {(name if i == 0 else ''):<{width}} {lvl:<11s} n={len(a):5d}  "
              f"Laya {fmt.format(np.median(a))}  LaBraM {fmt.format(np.median(b))}  "
              f"Laya wins {100*win:5.1f}%  p={p:.3e}")


def rank_report(df, col, name, width=20):
    """Rank of the truth among its nulls, aggregated. Uniform on [0,1] under
    the null at any ceiling, so this is comparable across two models whose
    nulls sit at very different levels. Tested against 0.5."""
    from scipy import stats
    for tag, nm in [("laya", "Laya  "), ("labram", "LaBraM")]:
        c = f"{col}_{tag}"
        if c not in df:
            continue
        v = df[["rec", c]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) < 6:
            continue
        r = v.groupby("rec")[c].median().to_numpy(float)
        try:
            p = stats.wilcoxon(r - 0.5, alternative="greater").pvalue
        except ValueError:
            p = np.nan
        print(f"  {(name if tag == 'laya' else ''):<{width}} {nm} "
              f"recordings n={len(r):5d}  median rank={np.median(r):.4f}  "
              f"(0.5 = chance)  p={p:.3e}")


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
    ap.add_argument("--pc-norm", choices=["plot", "raw"], default="plot",
                    help="colour space for the 3-PC metrics. 'plot' applies the "
                         "figure's per-component 2/98 percentile stretch to [0,1], "
                         "so each PC weighs equally as it does in the RGB overlay. "
                         "'raw' leaves PC1 dominating every distance, which is what "
                         "every result before 2026-07-30 used.")
    ap.add_argument("--h5-tag", default="LaBraMModel",
                    help="H5 model tag. LaBraMModel is 200 Hz microvolts; "
                         "LeJEPAClinical is defossez-scaled and clipped to +/-20 "
                         "and must NOT be used for the LaBraM arm.")
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
    ap.add_argument("--labram-mode", choices=["sliding", "native"],
                    default="sliding",
                    help="sliding (DEFAULT) reproduces the published Figure 2: "
                         "151 independent single-patch passes, no across-time "
                         "attention, time embedding pinned to slot 0. native "
                         "runs one 16 s pass and returns 1 token/s with full "
                         "context, and pools Laya to the same 1 Hz grid. Quote "
                         "sliding when describing the figure, native when "
                         "claiming anything about LaBraM as a model.")
    ap.add_argument("--reversal", action="store_true",
                    help="run M8, the onset/termination sign-reversal test. The "
                         "one test here that a temporal ramp cannot pass: a clock "
                         "moves forward at BOTH events, a state code moves back at "
                         "the second one. Recording-held-out, chance is 50%%.")
    ap.add_argument("--decoder", action="store_true",
                    help="run M9, the recording-held-out conditional decoder. "
                         "Asks whether the embedding beats a baseline that already "
                         "has elapsed time and the 7 waveform descriptors. This is "
                         "the direct test of the results.tex:42 claim; implies "
                         "--tracking for the per-second features it needs.")
    ap.add_argument("--tracking-pool", choices=["concat", "mean", "both"],
                    default="both",
                    help="how Laya's 10 x 0.1 s tokens are combined into one 1 s "
                         "row. concat keeps within-second temporal structure, "
                         "which LaBraM's single vector cannot express and which "
                         "therefore favours Laya; mean removes it. Reporting both "
                         "brackets the answer.")
    ap.add_argument("--tracking", action="store_true",
                    help="also run M6: does the embedding track the WAVEFORM or "
                         "the STATE? Equal-dimension, held out over chunks.")
    ap.add_argument("--auc-max-chunks", type=int, default=2000,
                    help="cap on chunks retained in RAM for M3")
    ap.add_argument("--null", choices=["chunk", "uniform"], default="chunk",
                    help="chunk (default): null labels are OTHER chunks' real "
                         "annotations, matched on transition count, so class "
                         "fraction and run lengths come from the real "
                         "distribution. uniform: the old _relocate, which "
                         "preserved neither and is kept only to reproduce "
                         "pre-2026-07-30 numbers.")
    ap.add_argument("--n-null", type=int, default=40,
                    help="null draws per chunk for M4/M7")
    ap.add_argument("--guard", type=float, default=0.5, metavar="SEC",
                    help="exclude +/- this many SECONDS around every label "
                         "change from M7. LaBraM's 1 s window smears a true "
                         "step over ~10 tokens; without a guard that ramp is "
                         "counted as within-state jitter and also pulls the "
                         "centroids together, penalising the baseline for its "
                         "own receptive field. 0 reproduces the old behaviour.")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test N chunks")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    import torch
    from scipy import stats

    global PC_NORM
    PC_NORM = args.pc_norm
    if args.decoder:
        args.tracking = True          # M9 reads the same per-second features
    print(f"pc-norm {PC_NORM}"
          f"{'  (figure colour space: per-PC 2/98 stretch to [0,1])' if PC_NORM == 'plot' else '  (raw PCs, PC1 dominates)'}")

    ckpt = str(args.ckpt).strip()
    if ckpt != args.ckpt:
        print(f"NOTE: stripped whitespace from --ckpt; md5 hashes the path string")

    index_path = args.index or find_lejepa_index(args.task, ckpt, args.split)
    h5_path    = args.h5 or find_h5(args.task, args.h5_tag)
    out        = args.out or f"figures/cp_{args.task}_{args.split}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"index  {index_path}")
    print(f"h5     {h5_path}")
    print(f"ckpt   {ckpt}")
    with open(index_path) as f:
        index = json.load(f)
    print(f"meta   {index.get('meta')}")

    pool = collect_label_pool(index, args.limit)
    print(f"null   {args.null}: {pool.summary()}")
    if args.null == "chunk" and len(pool) < 20:
        print("  *** WARNING: pool too small for a cross-chunk null; "
              "the same few patterns will recur. ***")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model  = load_labram(device)
    print(f"device {device}\n")

    meta        = index.get("meta") or {}
    chunk_len_s = float(meta.get("chunk_len_s", 16))
    n_missing = n_checked = n_mismatch = n_shape_mismatch = 0
    rows, t0, warned = [], time.time(), False
    keep = {"laya": [], "labram": []}
    keep_lab = {"laya": [], "labram": []}
    keep_w   = {"laya": [], "labram": []}
    keep_tr  = {"laya": [], "laya_mean": [], "labram": [], "W": [], "l": [],
                "rec": []}
    # M8 needs the embedding in its NATIVE space, not the per-chunk z-scored
    # copy in `keep`: z-scoring divides each dimension by its own within-chunk
    # std, which differs from chunk to chunk, so a direction learned on one
    # chunk means nothing on another.
    keep_rev = {"laya": [], "labram": [], "laya_lab": [], "labram_lab": [],
                "rec": []}
    with h5py.File(h5_path, "r") as hf:
        for ci, (seq_id, lab, emb) in enumerate(iter_chunks(index, args.limit)):
            grp = hf.get(f"/recordings/{seq_id}")
            if grp is None:
                n_missing += 1
                continue
            eeg      = grp["data"][:]
            ch_names = grp["channels"][:]

            # The index and the H5 are separate dataset builds that each number
            # recordings from scratch. If either dropped a recording, seq_id
            # silently pairs Laya's chunk with someone else's EEG. The writer
            # stores per-chunk labels (utils_2.py:243-246), so compare them.
            if "label" in grp:
                h5_lab = np.asarray(grp["label"][:]).ravel()
                if h5_lab.shape == np.asarray(lab).shape:
                    n_checked += 1
                    if not np.array_equal(h5_lab, np.asarray(lab)):
                        n_mismatch += 1
                else:
                    n_shape_mismatch += 1

            sfreq = infer_sfreq(eeg, chunk_len_s)

            Za = np.asarray(emb, dtype=np.float64)          # (160, 384)
            if args.labram_mode == "native":
                Zb, starts = labram_embeddings_native(
                    model, device, eeg, ch_names, sfreq, args.bipolar_map)
                # Pool Laya to LaBraM's 1 Hz so the two arms are scored on the
                # same time grid. Without this LaBraM's smoothness advantage
                # would be nothing but its coarser sampling.
                nsec = len(starts)
                if len(Za) % nsec == 0:
                    Za = Za.reshape(nsec, len(Za) // nsec, -1).mean(axis=1)
            else:
                Zb, starts = labram_embeddings(model, device, eeg, ch_names,
                                               sfreq, args.bipolar_map,
                                               args.batch_size)
            la = tok_labels_repeat(lab, len(Za))
            Zb = Zb.astype(np.float64)                      # (151, 200)
            lb = tok_labels_windows(lab, starts, LABRAM_PATCH_SZ, LABRAM_SFREQ)

            if not warned:
                print(f"  shapes: Laya {Za.shape} @ {len(Za)/16:.1f} Hz, "
                      f"LaBraM {Zb.shape} @ {len(Zb)/16:.1f} Hz, "
                      f"{len(decode_names(ch_names))} channels")
                fp = signal_fingerprint(eeg)
                print(f"  sfreq  {sfreq:.1f} Hz (inferred from {eeg.shape[1]} "
                      f"samples / {chunk_len_s:g} s)")
                print(f"  signal median|x|={fp['median']:.3g} p99={fp['p99']:.3g} "
                      f"max|x|={fp['max']:.3g} at_bound={fp['at_bound']:.5f} "
                      f"({'MICROVOLTS ok' if not fp['looks_clipped'] else 'CLIPPED'})")
                if fp['looks_clipped']:
                    print("  *** WARNING: looks defossez-scaled and clipped to +/-20. "
                          "LaBraM expects microvolts; use the LaBraMModel H5. ***")
                warned = True

            if args.tracking and len(keep_tr["W"]) < args.auc_max_chunks:
                # One row per SECOND, which is also the native label rate. Each
                # model gets exactly the same 1 s of signal: Laya's 10 tokens of
                # 0.1 s concatenated, LaBraM's single 1 s window. Without this
                # LaBraM predicts descriptors of the very samples it saw while
                # Laya is asked about 10x more signal than its token covers, and
                # the R2 gap is receptive field rather than training objective.
                nsec = len(lab)
                # native mode has already pooled Laya to 1 token/s, so there is
                # nothing left to concatenate and both arms are the same rows.
                per_s = len(Za) // nsec if nsec else 0
                oky   = len(Za) == nsec * per_s and per_s >= 1
                idxb  = np.searchsorted(starts, np.arange(nsec) * LABRAM_SFREQ)
                okb   = idxb.max() < len(Zb)
                if oky and okb:
                    keep_tr["laya"].append(
                        Za.reshape(nsec, -1).astype(np.float32))
                    keep_tr["laya_mean"].append(
                        Za.reshape(nsec, per_s, -1).mean(axis=1).astype(np.float32))
                    keep_tr["labram"].append(Zb[idxb].astype(np.float32))
                    keep_tr["W"].append(
                        _zscore(waveform_features(
                            eeg, sfreq, np.arange(nsec) + 0.5)).astype(np.float32))
                    keep_tr["l"].append(np.asarray(lab))
                    keep_tr["rec"].append(rec_id(seq_id))

            if (args.reversal and transitions(lab).size
                    and len(keep_rev["rec"]) < args.auc_max_chunks):
                keep_rev["laya"].append(Za.astype(np.float32))
                keep_rev["labram"].append(Zb.astype(np.float32))
                keep_rev["laya_lab"].append(la)
                keep_rev["labram_lab"].append(lb)
                keep_rev["rec"].append(rec_id(seq_id))

            # Null annotations for THIS chunk, drawn once at second resolution
            # and converted onto each model's own token grid below, so both
            # models are scored against the identical set of null labels.
            nrng = np.random.default_rng(10_000 + ci)
            if args.null == "chunk":
                null_secs = sample_null_labels(pool, ci, int(transitions(lab).size),
                                               args.n_null, nrng)
            else:
                null_secs = [_relocate(lab, nrng) for _ in range(args.n_null)]

            row = dict(seq_id=seq_id, rec=rec_id(seq_id),
                       n_trans=int(transitions(lab).size),
                       n_pos=int((lab == 1).sum()),
                       pos_frac=float(np.mean(np.asarray(lab) == 1)))
            for tag, Z, l in [("laya", Za, la), ("labram", Zb, lb)]:
                # From the token GEOMETRY, not from the token count. LaBraM
                # produces 151 windows over 16 s, so len(Z)/16 = 9.4375 Hz,
                # but the windows are strided 20 samples at 200 Hz and their
                # true rate is 10 Hz. The old expression inflated LaBraM's M2
                # and M5 errors in seconds by 10/9.4375, about 6%, against the
                # baseline. Laya is unaffected: 160/16 is already exactly 10.
                hz = (len(Za) / chunk_len_s if tag == "laya"
                      else LABRAM_SFREQ / LABRAM_STRIDE_SAMPLES
                      if args.labram_mode == "sliding"
                      else len(Zb) / chunk_len_s)
                nulls_tok = [(tok_labels_repeat(s, len(Za)) if tag == "laya"
                              else tok_labels_windows(s, starts, LABRAM_PATCH_SZ,
                                                      LABRAM_SFREQ))
                             for s in null_secs]
                guard_tok = int(round(args.guard * hz))
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
                            _zscore(waveform_features(eeg, sfreq, ctr))
                            .astype(np.float32))
                if args.smooth > 1:
                    ker = np.ones(args.smooth) / args.smooth
                    Z = np.apply_along_axis(
                        lambda v: np.convolve(v, ker, mode="same"), 0, Z)
                a, a0, arank = pca_state_auc(Z, l, nulls_tok)
                q7 = within_state_consistency(Z, l, nulls_tok, guard_tok)
                for key in ("eta2", "eta2n", "eta2rank", "jit", "jitfree",
                            "step", "sep", "spread"):
                    row[f"{key}_{tag}"] = q7[key]
                e5, s5, tr5 = m5_split_error(Z, l, hz)
                row[f"m5err_{tag}"]   = e5
                row[f"m5split_{tag}"] = s5
                row[f"m5trans_{tag}"] = tr5
                row[f"pcauc_{tag}"], row[f"pcauc0_{tag}"] = a, a0
                row[f"pcaucrank_{tag}"] = arank
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

    print(f"\nalignment: {n_checked} chunks label-checked, {n_mismatch} mismatched, "
          f"{n_shape_mismatch} shape-mismatched, {n_missing} index seq_ids absent from H5")
    if n_mismatch or n_shape_mismatch:
        raise SystemExit(
            f"ABORT: {n_mismatch + n_shape_mismatch} chunks disagree on labels between the "
            f"index and {h5_path}. The two builds number recordings independently, so "
            f"seq_id is pairing Laya embeddings with the wrong EEG. Do not report these "
            f"numbers. Rebuild the H5 or add an explicit seq_id map.")
    if n_checked == 0:
        print("WARNING: no chunk carried an H5 label, so alignment is UNVERIFIED.")

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
        arms = [("laya", "Laya (concat)"), ("laya_mean", "Laya (mean-pool)"),
                ("labram", "LaBraM")]
        if args.tracking_pool == "concat":
            arms = [a for a in arms if a[0] != "laya_mean"]
        elif args.tracking_pool == "mean":
            arms = [a for a in arms if a[0] != "laya"]
        for tag, nm in arms:
            if n_tr < 5:
                print(f"    {nm}: too few aligned chunks ({n_tr})"); continue
            auc, r2 = tracking_scores(keep_tr[tag], keep_tr["W"], keep_tr["l"])
            print(f"    {nm:18s} {auc:9.4f} " +
                  " ".join(f"{v:7.3f}" for v in r2) + f" {np.mean(r2):7.3f}")
        print(f"    (n={n_tr} chunks x {len(keep_tr['l'][0]) if n_tr else 0} s)")

    if args.reversal:
        print("\nM8  does the embedding REVERSE at a termination, or keep going?")
        print("    Direction learned at ONSET on one set of recordings, applied to")
        print("    events in recordings held out of that fit. A state code moves")
        print("    back toward interictal at termination, so onset->term should be")
        print("    NEGATIVE (frac+ well below 0.50). A clock keeps moving forward")
        print("    and stays positive. No smoothness or boundary artifact can")
        print("    produce reversal. onset->onset is the control: if it is not")
        print("    clearly positive there is no consistent direction to test.")
        print(f"    {'':18s} {'test':>16s} {'frac+':>8s} {'meanProj':>9s} {'n':>6s}")
        n_rev = len(keep_rev["rec"])
        for tag, nm in [("laya", "Laya  "), ("labram", "LaBraM")]:
            if n_rev < 10:
                print(f"    {nm}: too few chunks with a transition ({n_rev})")
                continue
            res = onset_termination_reversal(
                keep_rev[tag], keep_rev[f"{tag}_lab"], keep_rev["rec"], win=20)
            if not res:
                print(f"    {nm}: too few onset or termination events")
                continue
            for k, (fp, mp, n, per_rec) in res.items():
                if len(per_rec) >= 6:
                    # SAME event type on both sides -> consistency, expect
                    # positive. DIFFERENT -> reversal, expect negative. Both
                    # onset->term and term->onset are reversal tests; the
                    # earlier rule only caught the first and tested the second
                    # against the wrong tail.
                    src, dst = k.split("->")
                    alt = "greater" if src == dst else "less"
                    p = f"{stats.wilcoxon(per_rec, alternative=alt).pvalue:.2g}"
                else:
                    p = f"n/a ({len(per_rec)} rec)"
                print(f"    {nm:18s} {k:>16s} {fp:8.3f} {mp:>+9.4f} {n:6d}"
                      f"   p={p}")
        print(f"    (n={n_rev} chunks, 2.0 s either side of each transition;")
        print("     p is a recording-level signed-rank test, not event-level)")
        print("    CAVEAT, and it is a real one: M8 assumes interictal -> ictal ->")
        print("    interictal, so that a state code has to come back. If the true")
        print("    sequence is preictal -> ictal -> POSTICTAL and postictal is its")
        print("    own state, a genuine state representation should NOT reverse")
        print("    either. 'No reversal' is therefore consistent with a clock AND")
        print("    with a real three-state progression. M8 cannot separate those.")
        print("    M10 below is the test that can.")

        print("\nM10 step size vs distance from the annotated boundary")
        print("    A clock advances at a constant rate -> FLAT profile. A state")
        print("    progression moves fast between states and slowly inside one ->")
        print("    a PEAK near 0, even if the transition is smeared over seconds,")
        print("    which is the regime M2/M5 argmax tests are blind to. Steps are")
        print("    normalised by each chunk's own radius before pooling.")
        for tag, nm in [("laya", "Laya  "), ("labram", "LaBraM")]:
            if n_rev < 10:
                break
            hz_t = len(keep_rev[tag][0]) / chunk_len_s
            ctr, mu, cnt = step_profile(keep_rev[tag], keep_rev[f"{tag}_lab"], hz_t)
            ok = np.isfinite(mu) & (cnt > 0)
            if ok.sum() < 3:
                print(f"    {nm}: too few steps to profile")
                continue
            near, far = mu[ok][0], np.nanmean(mu[ok][max(1, ok.sum() // 2):])
            print(f"    {nm} (hz={hz_t:g})  " +
                  " ".join(f"{c:.1f}s:{v:.3f}" for c, v in
                           zip(ctr[ok], mu[ok])))
            print(f"    {'':18s} nearest-bin/far-mean = {near / far:.3f}"
                  f"   (1.00 = flat = clock)")

    if args.decoder:
        print("\nM9  does the embedding add state information BEYOND clock+waveform?")
        print("    One row per second, whole RECORDINGS held out, nested arms.")
        print("    The number that matters is the last column minus TIME+WAVE. If")
        print("    it is ~0 the embedding tells you nothing the clock and the")
        print("    bandpower did not already say, which is exactly the claim in")
        print("    results.tex:42. perm = same model with label sequences swapped")
        print("    between recordings, i.e. the null for that increment.")
        n_tr, tw_max = len(keep_tr["W"]), np.nan
        hdr = ["TIME", "WAVE", "TIME+WAVE", "TIME+WAVE+EMB"]
        print(f"    {'':18s} " + " ".join(f"{h:>14s}" for h in hdr)
              + f" {'gain':>7s} {'permGain':>9s}")
        for tag, nm in [("laya_mean", "Laya (mean-pool)"), ("labram", "LaBraM")]:
            if n_tr < 10:
                print(f"    {nm}: too few aligned chunks ({n_tr})"); continue
            r = conditional_state_decoder(keep_tr[tag], keep_tr["W"],
                                          keep_tr["l"], keep_tr["rec"])
            if not r:
                print(f"    {nm}: too few recordings for grouped CV"); continue
            gains = []
            for s_ in range(5):
                rp = conditional_state_decoder(keep_tr[tag], keep_tr["W"],
                                               keep_tr["l"], keep_tr["rec"],
                                               seed=s_, permute=True)
                if rp.get("TIME+WAVE+EMB") and rp.get("TIME+WAVE"):
                    gains.append(rp["TIME+WAVE+EMB"] - rp["TIME+WAVE"])
            gain   = r.get("TIME+WAVE+EMB", np.nan) - r.get("TIME+WAVE", np.nan)
            tw_max = np.nanmax([tw_max, r.get("TIME+WAVE", np.nan)])
            print(f"    {nm:18s} "
                  + " ".join(f"{r.get(h, float('nan')):14.4f}" for h in hdr)
                  + f" {gain:>+7.4f} "
                  + (f"{np.mean(gains):>+9.4f}" if gains else f"{'--':>9s}"))
        print(f"    (n={n_tr} chunks, {len(set(keep_tr['rec']))} recordings, "
              f"5-fold grouped CV, 32 global PCs fit on train only)")
        if np.isfinite(tw_max) and tw_max > 0.95:
            print(f"    *** TIME+WAVE reaches {tw_max:.4f}. The baseline is at "
                  "ceiling, so the increment has no headroom and a gain near")
            print("    zero is UNINTERPRETABLE rather than negative evidence. "
                  "Report the ceiling, do not report the gain.")

        print("\nM11 how many UNSUPERVISED dimensions does the state code need?")
        print("    THIS is the quantitative version of Figure 2, and it is not")
        print("    the linear probe restated. The probe is supervised over all D")
        print("    dims and says the information is PRESENT. Here the basis is")
        print("    PCA fit on train embeddings with labels never shown to it, so")
        print("    'state decodes from the first 3 PCs' is a claim about how the")
        print("    space is ARRANGED -- which is what the figure shows and what")
        print("    balanced accuracy cannot say. Saturating at small k means")
        print("    state is a dominant axis; climbing to large k means state is")
        print("    present but spread thin, i.e. decodable but not organised.")
        for tag, nm in [("laya_mean", "Laya (mean-pool)"), ("labram", "LaBraM")]:
            if n_tr < 10:
                break
            r = pc_dimension_sweep(keep_tr[tag], keep_tr["l"], keep_tr["rec"])
            if not r:
                print(f"    {nm}: too few recordings for grouped CV"); continue
            print(f"    {nm:18s} " + " ".join(f"k{k}:{v:.3f}" for k, v in
                                              sorted(r["auc"].items())))
            print(f"    {'':18s} k95={r['k95']}  (smallest k at 95% of this "
                  f"model's own gain over 0.5)")
        print("    CAVEAT to quote alongside: PCA ranks by variance, so a model "
              "whose largest")
        print("    component is big but irrelevant (drift, DC offset, amplitude "
              "envelope) is")
        print("    penalised for reasons unrelated to state. This measures the "
              "representation")
        print("    as it leaves the model, not whether state exists inside it.")

    print("\nM7  is each state held at a CONSISTENT colour? (the 3 PCs shown)")
    print("    jitfree is the LABEL-FREE smoothness statistic (step / chunk 3-PC")
    print("    radius) and is the one to quote for 'smoother trajectory'. jit")
    print("    divides by the label-defined centroid distance instead, so a")
    print("    temporal ramp scores well on it for free; report it only next to")
    print("    jitfree. step and sep are jit's numerator and denominator.")
    for tag, nm in [("laya", "Laya  "), ("labram", "LaBraM")]:
        e  = df[f"eta2_{tag}"].to_numpy(float)
        en = df[f"eta2n_{tag}"].to_numpy(float)
        m  = np.isfinite(e) & np.isfinite(en)
        g = lambda c: np.nanmedian(df[f"{c}_{tag}"].to_numpy(float))
        # MEAN margin, not median. eta2 is nonlinear in boundary position, so
        # median[eta2 - mean(nulls)] is biased even under a correct null:
        # measured +0.0184 on a pure clock where the truth is 0, while the mean
        # reads +0.0008. See calib_null_bias.py.
        print(f"  {nm:8s} n={int(m.sum()):5d}  eta2={np.median(e[m]):.4f} "
              f"(null {np.median(en[m]):.4f}, mean margin {np.mean(e[m]-en[m]):+.4f})  "
              f"jit={g('jit'):.4f}  jitfree={g('jitfree'):.4f}  "
              f"step={g('step'):.4f}  sep={g('sep'):.4f}  spread={g('spread'):.4f}")
    paired_report(df, "jitfree", "paired jitfree")
    paired_report(df, "jit",     "paired jit")
    rank_report(df,  "eta2rank", "eta2 rank vs null")

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
    paired_report(df, "m5err", "paired", fmt="{:.3f}")

    print("\nM4  state separability on the 3 PCs the figure colours with")
    print("    Circular by design; the null is built the same way. READ THE RANK,")
    print("    not the margin: on a smooth trajectory every relocated split")
    print("    already scores near the maximum, so the margin is ceiling-")
    print("    compressed and a smoother model looks less state-specific purely")
    print("    for being smooth. The rank is uniform on [0,1] under the null at")
    print("    any ceiling and is therefore comparable between the two models.")
    for tag in ("laya", "labram"):
        df[f"m4margin_{tag}"] = df[f"pcauc_{tag}"] - df[f"pcauc0_{tag}"]
    for tag, nm in [("laya", "Laya  "), ("labram", "LaBraM")]:
        a  = df[f"pcauc_{tag}"].to_numpy(float)
        a0 = df[f"pcauc0_{tag}"].to_numpy(float)
        m  = np.isfinite(a) & np.isfinite(a0)
        if m.sum() <= 5:
            print(f"  {nm}: too few"); continue
        st = stats.wilcoxon(a[m], a0[m], alternative="greater")
        print(f"  {nm:8s} n={int(m.sum()):5d}  AUC={np.median(a[m]):.4f}  "
              f"null={np.median(a0[m]):.4f}  "
              f"mean margin={np.mean(a[m] - a0[m]):+.4f}  p={st.pvalue:.3e}")
    paired_report(df, "m4margin", "paired margin", alternative="greater",
                  fmt="{:+.4f}")
    rank_report(df, "pcaucrank", "M4 rank vs null")

    df.to_csv(f"{out}.csv", index=False)
    print(f"\n{len(df)} chunks with a label transition -> {out}.csv "
          f"({time.time()-t0:.0f}s)")
    aggregate(df, f"{out}.png")


if __name__ == "__main__":
    main()
