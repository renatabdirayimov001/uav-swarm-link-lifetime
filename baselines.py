"""
Analytical baselines and evaluation metrics for residual link lifetime.

- LET    : constant-velocity link expiration time (Su, Lee & Gerla, 2001)
- KF-LET : Kalman filter with a Singer (manoeuvring-target) acceleration model,
           as in Kalman-assisted FANET routing (MP-QGRD, KFQR); the filtered
           trajectories are extrapolated and the first time the distance
           exceeds the range R is the predicted lifetime.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Kalman filter (Singer model), identical for every UAV and axis -> vectorised
# ---------------------------------------------------------------------------
def singer_matrices(dt, tau, q):
    a = 1.0 / tau
    e = np.exp(-a * dt)
    F = np.array([[1, dt, (a * dt - 1 + e) / a**2],
                  [0, 1, (1 - e) / a],
                  [0, 0, e]])
    Q = q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6],
                      [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                      [dt**3 / 6, dt**2 / 2, dt]])
    return F, Q


def kalman_singer(P_obs, V_obs, cfg, tau=3.0, q=2.0):
    """Filter noisy positions/velocities. Inputs (T, N, 3).
    Returns filtered position, velocity, acceleration (T, N, 3)."""
    T, N, _ = P_obs.shape
    dt = cfg["dt"]
    F, Q = singer_matrices(dt, tau, q)
    H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    R = np.diag([cfg["obs_pos_std"]**2, cfg["obs_vel_std"]**2])
    x = np.zeros((N, 3, 3))                     # (uav, axis, state)
    x[:, :, 0], x[:, :, 1] = P_obs[0], V_obs[0]
    Pc = np.diag([cfg["obs_pos_std"]**2, cfg["obs_vel_std"]**2, 4.0])
    out = np.zeros((T, N, 3, 3))
    for t in range(T):
        if t > 0:                               # predict
            x = x @ F.T
            Pc = F @ Pc @ F.T + Q
        S = H @ Pc @ H.T + R                    # update (same gain for all)
        K = Pc @ H.T @ np.linalg.inv(S)
        z = np.stack([P_obs[t], V_obs[t]], -1)  # (N, 3, 2)
        x = x + (z - x @ H.T) @ K.T
        Pc = (np.eye(3) - K @ H) @ Pc
        out[t] = x
    return out[..., 0], out[..., 1], out[..., 2]


def kf_let(P_obs, V_obs, iu, ju, R, cfg, tau=3.0, q=2.0):
    """Predicted lifetime from Singer-model extrapolation of both UAVs."""
    p, v, a = kalman_singer(P_obs, V_obs, cfg, tau, q)
    H, dt = cfg["horizon"], cfg["dt"]
    tk = np.arange(dt, H + dt / 2, dt)                      # prediction grid
    al = 1.0 / tau
    g = (np.exp(-al * tk) - 1 + al * tk) / al**2            # Singer position gain
    dp, dv, da = p[:, iu] - p[:, ju], v[:, iu] - v[:, ju], a[:, iu] - a[:, ju]
    T, M, _ = dp.shape
    life = np.full((T, M), H)
    for t in range(T):
        traj = dp[t][None] + tk[:, None, None] * dv[t][None] + g[:, None, None] * da[t][None]
        out = np.linalg.norm(traj, axis=2) > R                # (K, M)
        first = np.where(out.any(0), out.argmax(0), -1)
        life[t] = np.where(first >= 0, tk[np.maximum(first, 0)], H)
        life[t][np.linalg.norm(dp[t], axis=1) > R] = 0.0
    return np.clip(life, 0, H)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(pred, life, censored, up, delta=5.0):
    """Regression metrics on uncensored active links + break-within-delta
    classification on all active links (censored links never break early)."""
    m = up & ~censored
    y, yh = life[m], pred[m]
    err = yh - y
    mae, rmse = np.abs(err).mean(), np.sqrt((err**2).mean())
    r2 = 1 - (err**2).sum() / ((y - y.mean())**2).sum()
    lab = (up & ~censored & (life <= delta))[up]
    score = -pred[up]
    return dict(MAE=mae, RMSE=rmse, R2=r2,
                AUC=roc_auc_score(lab, score), AUPRC=average_precision_score(lab, score),
                n=int(m.sum()), pos_rate=lab.mean())
