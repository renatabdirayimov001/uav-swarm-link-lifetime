"""
Third-revision baselines and feature studies (code v6, CPU only).

- let_snr          : SNR-corrected LET. The shadowing of each link is estimated from every Hello received in the
                     last W slots, X^ = mean over those Hellos of [Pt - PL(d^) - N0 - gamma] (d^ from the GNSS
                     positions of that slot; the mean of the Rician fading in dB is removed), and the LET is
                     computed with the effective range
                     R_eff = R * 10^(-X^ / (10 eta)) at which the large-scale SNR reaches the threshold.
                     Uses only Hello information and no training.
- nbr_features     : neighbourhood summaries of the other active links of both endpoints (count, mean SNR,
                     mean SNR change, mean LET, share of weak links), combined symmetrically. Each endpoint knows
                     its own summary; the other endpoint's summary would be 5 numbers carried in its Hello.
- GROUPS / columns : column groups of extra.gbt_features for the feature ablation of the GBT.
"""
import numpy as np
import uav_swarm_sim as sim
import data_torch as dt

# edge feature order (data_torch.episode_features): dist, radial, rel, snr, dsnr, cap, LET, up, miss
EDGE_K, EDGE_S, EDGE_UP, EDGE_M = [0, 1, 2, 6], [3, 4, 5], [7], [8]
NE = 9                       # edge features per block (current, lag 3, lag W-1)
PAIR0 = NE                   # pair node features: sum (x, y, z, vx, vy, vz, deg), |diff| (same)
PAIR_ABS_POS = [PAIR0 + 0, PAIR0 + 1, PAIR0 + 2]          # sums of absolute coordinates (= link midpoint)
PAIR_ALL = list(range(PAIR0, PAIR0 + 14))
LAG0 = PAIR0 + 14            # lag blocks start here


def _edge_cols(idx):
    return [b + i for b in (0, LAG0, LAG0 + NE) for i in idx]


def columns(variant, n_base=LAG0 + 2 * NE, n_nbr=0):
    """Column indices of the GBT input for an ablation variant.
    K = kinematics (distance, radial and relative speed, LET, endpoint node features), S = measured SNR
    (SNR, its change, spectral efficiency), M = missed-Hello flag."""
    K = _edge_cols(EDGE_K) + PAIR_ALL
    S, M, UP = _edge_cols(EDGE_S), _edge_cols(EDGE_M), _edge_cols(EDGE_UP)
    allc = list(range(n_base))
    sel = {
        "all": allc,
        "K": K + UP,
        "K+M": K + UP + M,
        "S": S + UP,
        "S+M": S + UP + M,
        "K+S": K + S + UP,
        "noAbsPos": [c for c in allc if c not in PAIR_ABS_POS],
        "all+nbr": allc + list(range(n_base, n_base + n_nbr)),
    }[variant]
    return sorted(sel)


VARIANTS = ["all", "K", "K+M", "S", "S+M", "K+S", "noAbsPos", "all+nbr"]


def nbr_features(ef, up, iu, ju, n, snr_th_db):
    """ef (T, M, 9) edge features (held, zero when down), up (T, M) bool -> (T, M, 10) features."""
    u = up.astype(np.float64)
    inc = np.zeros((len(iu), n))
    inc[np.arange(len(iu)), iu] = 1.0
    inc[np.arange(len(iu)), ju] = 1.0
    snr_db = ef[..., 3] * 30.0
    qs = [ef[..., 3], ef[..., 4], ef[..., 6], ((snr_db < snr_th_db + 3.0) & up).astype(np.float64)]
    cnt = u @ inc                                                  # (T, N) active links per node
    out = []
    ci, cj = cnt[:, iu] - u, cnt[:, ju] - u                        # other links of each endpoint
    out += [ci + cj, np.abs(ci - cj)]
    for q in qs:
        s = (q * u) @ inc
        mi = np.where(ci > 0, (s[:, iu] - q * u) / np.maximum(ci, 1), 0.0)
        mj = np.where(cj > 0, (s[:, ju] - q * u) / np.maximum(cj, 1), 0.0)
        out += [mi + mj, np.abs(mi - mj)]
    F = np.stack(out, -1)
    F[~up] = 0.0
    return F.astype(np.float32)


def rician_db_mean(k_db, n=2_000_000):
    """E[10 log10 |h|^2] of the unit-power Rician gain (the bias of an average of measured SNRs in dB)."""
    rng = np.random.default_rng(0)
    K = 10 ** (k_db / 10)
    h = np.sqrt(K / (K + 1)) + np.sqrt(1 / (2 * (K + 1))) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    return float(np.mean(10 * np.log10(np.abs(h) ** 2)))


def let_snr(pos_obs, vel_obs, snr_meas, rx, up, miss, iu, ju, cfg, W, H, decay=False):
    """SNR-corrected LET for one episode. rx (T, M): active link with a received Hello in that slot."""
    T = snr_meas.shape[0]
    R = sim.comm_range(cfg)
    fade = rician_db_mean(cfg["rician_k_db"])
    dp, dv = pos_obs[:, iu] - pos_obs[:, ju], vel_obs[:, iu] - vel_obs[:, ju]
    d = np.maximum(np.linalg.norm(dp, axis=2), 1.0)
    # shadowing seen by each received Hello (its own distance), averaged over the Hellos of the window
    x1 = cfg["pt_dbm"] - sim.path_loss_db(d, cfg) - sim.noise_dbm(cfg) - (snr_meas - fade)
    s = np.where(rx, x1, 0.0)
    cs = np.vstack([np.zeros((1, s.shape[1])), np.cumsum(s, 0)])
    cn = np.vstack([np.zeros((1, s.shape[1])), np.cumsum(rx, 0)])
    lo = np.maximum(np.arange(T) - W + 1, 0)
    S, N = cs[1:] - cs[lo], cn[1:] - cn[lo]
    xhat = np.where(N > 0, S / np.maximum(N, 1), 0.0)
    if decay:
        # the shadowing estimate decays towards zero as the link distance changes (Gudmundson, eq. (4)):
        # the break distance r >= d solves 10 eta log10(r / R) + X^ exp(-(r - d) / d_corr) = 0 (bisection)
        eta, dc = cfg["pl_exp"], cfg["shadow_dcorr"]
        g = lambda r: 10 * eta * np.log10(r / R) + xhat * np.exp(-(r - d) / dc)
        lo_r, hi_r = d.copy(), np.maximum(d, R) * 4.0
        for _ in range(40):
            mid = 0.5 * (lo_r + hi_r)
            pos = g(mid) > 0
            hi_r = np.where(pos, mid, hi_r); lo_r = np.where(pos, lo_r, mid)
        reff = np.where(g(d) > 0, d * 0.999, 0.5 * (lo_r + hi_r))
    else:
        reff = R * 10 ** (-xhat / (10 * cfg["pl_exp"]))
    a = (dv * dv).sum(-1)
    b = 2 * (dp * dv).sum(-1)
    c = (dp * dp).sum(-1) - reff ** 2
    disc = np.maximum(b * b - 4 * a * c, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = (-b + np.sqrt(disc)) / (2 * a)
    tau = np.where(a < 1e-9, H, tau)
    tau = np.where(c > 0, 0.0, tau)
    tau = np.clip(np.nan_to_num(tau, nan=H), 0, H)
    tau = np.maximum(dt.hold_prev(tau, miss) - dt.miss_age(miss) * cfg["dt"], 0.0)   # held like the LET
    tau[~up] = 0.0
    return tau.astype(np.float32), reff
