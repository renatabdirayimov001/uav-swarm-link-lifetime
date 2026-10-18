"""
Experiment 3 helpers: additional baselines and censoring-aware metrics.

- snr_trend          : heuristic baseline, extrapolates the linear trend of the
                       measured SNR over the observation window to the threshold
- gbt_features       : per-link tabular features for a gradient-boosted-tree baseline
- evaluate_ext       : censoring-aware metrics (MAE over all active links with a
                       one-sided error for censored links, Harrell's C-index) and
                       break detection for several horizons Delta
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def snr_trend(snr_meas, up, th_db, H, dt, W):
    """Time until the least-squares SNR trend over the last W slots reaches th_db."""
    T, M = snr_meas.shape
    k = np.arange(W) - (W - 1) / 2.0
    pred = np.full((T, M), H, dtype=float)
    for t in range(W - 1, T):
        win = snr_meas[t - W + 1:t + 1]                       # (W, M)
        slope = (k[:, None] * (win - win.mean(0))).sum(0) / (k ** 2).sum() / dt   # dB/s
        level = win.mean(0) + slope * dt * (W - 1) / 2.0      # trend value at slot t
        ttc = np.where(slope < 0, (th_db - level) / np.minimum(slope, -1e-9), H)
        ttc = np.where(level < th_db, 0.0, ttc)
        pred[t] = np.clip(ttc, 0.0, H)
    pred[~up] = 0.0
    return pred


def gbt_features(nf, ef, iu, ju, W):
    """Rows (t, m) for t >= W-1: current edge features, endpoint node features
    (sum, |diff|), and edge features 3 and 7 slots earlier. Returns X (T, M, F)."""
    a, b = nf[:, iu], nf[:, ju]
    pair = np.concatenate([a + b, np.abs(a - b)], -1)
    lag = lambda L: np.concatenate([np.zeros_like(ef[:L]), ef[:-L]], 0)
    return np.concatenate([ef, pair, lag(3), lag(W - 1)], -1).astype(np.float32)


def cindex(pred, life, cens, n=4000, rng=None):
    """Harrell's C-index on a random subsample of n active samples."""
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(pred), size=min(n, len(pred)), replace=False)
    p, y, c = pred[idx], life[idx], cens[idx]
    comp = (y[:, None] < y[None, :]) & (~c)[:, None]          # i has event before j
    conc = (p[:, None] < p[None, :]) + 0.5 * (p[:, None] == p[None, :])
    return float((conc * comp).sum() / max(comp.sum(), 1))


def evaluate_ext(pred, life, cens, up, H, deltas=(2.0, 5.0, 10.0)):
    """pred, life, cens, up: (T, M) arrays of one episode (slots t >= W-1)."""
    p, y, c = pred[up], life[up], cens[up]
    err = np.where(c, np.maximum(0.0, H - p), np.abs(p - y))
    out = dict(MAE_all=float(err.mean()), Cindex=cindex(p, y, c))
    for d in deltas:
        lab = (~c) & (y <= d)
        out[f"AUC_{d:g}"] = roc_auc_score(lab, -p)
        out[f"AUPRC_{d:g}"] = average_precision_score(lab, -p)
        out[f"pos_{d:g}"] = float(lab.mean())
    return out
