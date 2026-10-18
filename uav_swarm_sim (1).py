"""
UAV Swarm Link-Lifetime Dataset Generator
=========================================
Paper: Learning-Based Residual Link Lifetime Prediction for UAV Swarms
Author: Renat Abdirayimov (TATU)
Version: 2 (causal link state, fully observed lifetime labels)

What this module does
---------------------
1. Moves N UAVs in 3D using three mobility models:
   - Gauss-Markov (GM): each UAV flies independently, smooth random motion
   - RPGM (Reference Point Group Mobility): UAVs fly in groups (swarm squads)
   - RWP (3D random waypoint): straight flights between random waypoints
2. Computes the air-to-air (A2A) channel for every UAV pair:
   path loss + time-correlated shadowing + Rician small-scale fading -> SNR
3. Decides which links are UP (large-scale SNR >= threshold) -> a dynamic graph.
   The link state is CAUSAL: it depends only on the current and past slots.
4. Labels every UP link with its TRUE residual lifetime (seconds until it breaks).
   The episode is simulated for T + H seconds, but samples are taken only in
   [0, T), so every label has a full H-second follow-up.
5. Computes the classic analytical baseline: Link Expiration Time (LET)
"""
import numpy as np

# ----------------------------------------------------------------------------
# 1. Simulation parameters (these go into Table I of the paper)
# ----------------------------------------------------------------------------
DEFAULT_CFG = dict(
    n_uav=30,              # number of UAVs
    area=(1500.0, 1500.0), # x, y size [m]
    alt=(50.0, 150.0),     # min/max altitude [m]
    dt=0.5,                # time step [s]
    T=600.0,               # episode length [s]
    mobility="rpgm",       # "gm", "rpgm" or "rwp"
    group_size=5,          # UAVs per group (RPGM)
    v_mean=12.0,           # mean speed [m/s]
    v_max=35.0,            # max UAV speed [m/s]
    gm_alpha=0.85,         # Gauss-Markov memory (0=random walk, 1=straight line)
    group_radius=250.0,    # max deviation of a member from its group centre [m]
    member_alpha=0.95,     # memory of a member's own motion inside the group
    member_noise=2.0,      # random acceleration of members [m/s per step]
    member_pull=0.002,     # spring force pulling members back to the centre
    # ---- channel (A2A link) ----
    fc=2.4e9,              # carrier frequency [Hz]
    pt_dbm=10.0,           # transmit power [dBm]
    bw=20e6,               # bandwidth [Hz]
    nf_db=7.0,             # receiver noise figure [dB]
    pl_exp=2.2,            # A2A path-loss exponent
    shadow_std=2.0,        # shadowing std [dB]
    shadow_dcorr=50.0,     # shadowing decorrelation distance [m]
    shadow_model="link",   # "link": independent per link (Gudmundson in distance change)
                           # "spatial": static 2-D shadowing field shared by all links
    field_corr=100.0,      # correlation length of the spatial shadowing field [m]
    rician_k_db=7.6,       # Rician K-factor, measured A2A value (Ad Hoc Networks 2025)
    snr_th_db=10.0,        # link is UP if large-scale SNR >= this [dB]
    break_steps=2,         # link is broken only if down for >= 2 steps (1 s)
    horizon=60.0,          # residual lifetime is capped (censored) here [s]
    channel="a2a",         # "a2a" (SNR threshold, shadowing + Rician) or "disk"
    rwp_pause_max=2.0,     # RWP pause time at a waypoint, U(0, max) [s]
    obs_pos_std=2.0,       # GPS position error in Hello messages [m]
    obs_vel_std=0.3,       # velocity error in Hello messages [m/s]
    seed=0,
)

C = 3e8


def noise_dbm(cfg):
    return -174.0 + 10 * np.log10(cfg["bw"]) + cfg["nf_db"]


def path_loss_db(d, cfg):
    """Log-distance path loss with free-space reference at 1 m."""
    pl0 = 20 * np.log10(4 * np.pi * cfg["fc"] / C)  # FSPL at 1 m
    return pl0 + 10 * cfg["pl_exp"] * np.log10(np.maximum(d, 1.0))


def comm_range(cfg):
    """Distance where mean SNR (no shadowing) equals the threshold."""
    pl_max = cfg["pt_dbm"] - noise_dbm(cfg) - cfg["snr_th_db"]
    pl0 = 20 * np.log10(4 * np.pi * cfg["fc"] / C)
    return 10 ** ((pl_max - pl0) / (10 * cfg["pl_exp"]))


# ----------------------------------------------------------------------------
# 2. Mobility models
# ----------------------------------------------------------------------------
def _reflect(pos, vel, mean_dir, lo, hi):
    """Bounce UAVs off the borders of the area (velocity and mean direction flip)."""
    for k in range(3):
        below, above = pos[:, k] < lo[k], pos[:, k] > hi[k]
        pos[below, k] = 2 * lo[k] - pos[below, k]
        pos[above, k] = 2 * hi[k] - pos[above, k]
        vel[below | above, k] *= -1
        mean_dir[below | above, k] *= -1
    return pos, vel, mean_dir


def _gauss_markov_step(vel, mean_dir, cfg, rng):
    a = cfg["gm_alpha"]
    noise = rng.normal(0, 1, vel.shape) * cfg["v_mean"] * 0.5
    noise[:, 2] *= 0.2  # vertical motion is smaller
    return a * vel + (1 - a) * mean_dir + np.sqrt(1 - a ** 2) * noise


def simulate_mobility(cfg, rng):
    """Returns positions and velocities, shape (n_steps, n_uav, 3)."""
    n, dt = cfg["n_uav"], cfg["dt"]
    steps = int(cfg["T"] / dt)
    lo = np.array([0, 0, cfg["alt"][0]])
    hi = np.array([cfg["area"][0], cfg["area"][1], cfg["alt"][1]])

    def rand_heading(m):
        th = rng.uniform(0, 2 * np.pi, m)
        v = np.stack([np.cos(th), np.sin(th), np.zeros(m)], 1) * cfg["v_mean"]
        return v

    P = np.zeros((steps, n, 3))
    V = np.zeros((steps, n, 3))

    if cfg["mobility"] == "gm":
        pos = rng.uniform(lo, hi, (n, 3))
        vel = rand_heading(n)
        mean_dir = vel.copy()
        for t in range(steps):
            P[t], V[t] = pos, vel
            vel = _gauss_markov_step(vel, mean_dir, cfg, rng)
            pos = pos + vel * dt
            pos, vel, mean_dir = _reflect(pos, vel, mean_dir, lo, hi)
    elif cfg["mobility"] == "rpgm":
        g = cfg["group_size"]
        n_groups = int(np.ceil(n / g))
        member_group = np.arange(n) // g
        margin = np.array([200.0, 200.0, 0.0])
        c_pos = rng.uniform(lo + margin, hi - margin, (n_groups, 3))
        c_pos[:, 2] = rng.uniform(*cfg["alt"], n_groups)
        c_vel = rand_heading(n_groups)
        c_mean = c_vel.copy()
        # each member keeps a random offset that slowly drifts (Gauss-Markov)
        offset = rng.normal(0, cfg["group_radius"] / 2, (n, 3))
        offset[:, 2] *= 0.3
        off_vel = np.zeros((n, 3))
        prev = c_pos[member_group] + offset
        for t in range(steps):
            pos = np.clip(c_pos[member_group] + offset, lo, hi)
            if t > 0:  # physical speed limit (removes jumps at the borders)
                step = pos - prev
                sp = np.linalg.norm(step, axis=1, keepdims=True) / dt
                pos = prev + np.where(sp > cfg["v_max"], step * cfg["v_max"] / np.maximum(sp, 1e-9), step)
                vel = (pos - prev) / dt
            else:
                vel = c_vel[member_group].copy()
            P[t], V[t] = pos, vel
            prev = pos
            # group centres move with Gauss-Markov
            c_vel = _gauss_markov_step(c_vel, c_mean, cfg, rng)
            c_pos = c_pos + c_vel * dt
            c_pos, c_vel, c_mean = _reflect(c_pos, c_vel, c_mean, lo, hi)
            # members wander inside the group (pulled back to the centre)
            off_vel = cfg["member_alpha"] * off_vel + rng.normal(0, cfg["member_noise"], (n, 3)) \
                - cfg["member_pull"] * offset
            off_vel[:, 2] *= 0.5
            offset = offset + off_vel * dt
            r = np.linalg.norm(offset, axis=1, keepdims=True)
            offset = np.where(r > cfg["group_radius"], offset * cfg["group_radius"] / r, offset)
    elif cfg["mobility"] == "rwp":
        # 3-D random waypoint: fly straight to a random waypoint at a random
        # speed, pause, pick the next waypoint
        pos = rng.uniform(lo, hi, (n, 3))
        wp = rng.uniform(lo, hi, (n, 3))
        speed = rng.uniform(0.5, 1.5, n) * cfg["v_mean"]
        pause = np.zeros(n)
        for t in range(steps):
            d = wp - pos
            dist = np.linalg.norm(d, axis=1)
            moving = pause <= 0
            vel = np.where(moving[:, None], d / np.maximum(dist, 1e-9)[:, None] * speed[:, None], 0.0)
            arrive = moving & (dist <= speed * dt)
            vel[arrive] = d[arrive] / dt
            P[t], V[t] = pos, vel
            pos = pos + vel * dt
            pause = pause - dt
            # arrivals: pause, then new waypoint and speed
            pause[arrive] = rng.uniform(0, cfg["rwp_pause_max"], arrive.sum())
            wp[arrive] = rng.uniform(lo, hi, (arrive.sum(), 3))
            speed[arrive] = rng.uniform(0.5, 1.5, arrive.sum()) * cfg["v_mean"]
    else:
        raise ValueError("mobility must be 'gm', 'rpgm' or 'rwp'")
    return P, V


# ----------------------------------------------------------------------------
# 3. A2A channel
# ----------------------------------------------------------------------------
def simulate_channel(P, cfg, rng):
    """Returns distance, large-scale SNR and measured (faded) SNR for all pairs."""
    steps, n, _ = P.shape
    iu, ju = np.triu_indices(n, 1)
    diff = P[:, iu, :] - P[:, ju, :]
    dist = np.linalg.norm(diff, axis=2)                       # (steps, n_pairs)

    if cfg.get("shadow_model", "link") == "spatial":
        shadow = spatial_shadowing(P, iu, ju, cfg, rng)
        return _finish_channel(dist, shadow, iu, ju, cfg, rng)
    # time-correlated shadowing (Gudmundson model): rho = exp(-moved / d_corr)
    shadow = np.zeros_like(dist)
    shadow[0] = rng.normal(0, cfg["shadow_std"], dist.shape[1])
    for t in range(1, steps):
        moved = np.abs(dist[t] - dist[t - 1]) + 1e-3
        rho = np.exp(-moved / cfg["shadow_dcorr"])
        shadow[t] = rho * shadow[t - 1] + np.sqrt(1 - rho ** 2) * rng.normal(
            0, cfg["shadow_std"], dist.shape[1])

    return _finish_channel(dist, shadow, iu, ju, cfg, rng)


def spatial_shadowing(P, iu, ju, cfg, rng, res=10.0, n_pts=8):
    """Spatially correlated shadowing: a static 2-D Gaussian random field G(x, y)
    with correlation length `field_corr` (buildings / terrain). The shadowing of
    link (i, j) is the field averaged along the ground projection of the segment
    i-j, rescaled to std `shadow_std`. Links that cross the same region are
    therefore correlated, and a link's shadowing changes as its endpoints move."""
    from scipy.ndimage import gaussian_filter
    nx = int(cfg["area"][0] / res) + 1
    ny = int(cfg["area"][1] / res) + 1
    G = gaussian_filter(rng.normal(size=(ny, nx)), sigma=cfg["field_corr"] / res / 2, mode="wrap")
    G = (G - G.mean()) / G.std()
    s = np.linspace(0.0, 1.0, n_pts)
    pi, pj = P[:, iu, :2], P[:, ju, :2]                          # (T, M, 2)
    pts = pi[:, :, None, :] + s[None, None, :, None] * (pj - pi)[:, :, None, :]
    ix = np.clip(np.rint(pts[..., 0] / res).astype(int), 0, nx - 1)
    iy = np.clip(np.rint(pts[..., 1] / res).astype(int), 0, ny - 1)
    L = G[iy, ix].mean(-1)                                       # (T, M)
    return cfg["shadow_std"] * L / L.std()


def _finish_channel(dist, shadow, iu, ju, cfg, rng):
    snr_ls = cfg["pt_dbm"] - path_loss_db(dist, cfg) - shadow - noise_dbm(cfg)

    # Rician fading power |h|^2 with unit mean
    K = 10 ** (cfg["rician_k_db"] / 10)
    los = np.sqrt(K / (K + 1))
    nlos = np.sqrt(1 / (2 * (K + 1)))
    h = los + nlos * (rng.normal(size=dist.shape) + 1j * rng.normal(size=dist.shape))
    snr_meas = snr_ls + 10 * np.log10(np.abs(h) ** 2)
    return dist, snr_ls, snr_meas, iu, ju


# ----------------------------------------------------------------------------
# 4. Labels: true residual link lifetime + LET baseline
# ----------------------------------------------------------------------------
def debounce(up_raw, k):
    """CAUSAL link state machine (uses only the current and past slots).
    An active link is declared BROKEN at the k-th consecutive slot whose
    large-scale SNR is below the threshold (k missed Hello messages); during
    the first k-1 low slots it is still active. A broken link becomes active
    again as soon as the threshold is exceeded. Initial state = up_raw[0]."""
    if k <= 1:
        return up_raw.copy()
    out = np.zeros_like(up_raw, dtype=bool)
    state = up_raw[0].copy()
    miss = np.where(up_raw[0], 0, k).astype(int)
    out[0] = state
    for t in range(1, up_raw.shape[0]):
        miss = np.where(up_raw[t], 0, miss + 1)
        state = np.where(up_raw[t], True, state & (miss < k))
        out[t] = state
    return out


def residual_lifetime(up, cfg):
    """For every (t, pair) that is UP, seconds until the link first goes DOWN.
    Capped at `horizon` (right-censored). Returns (lifetime, censored_flag)."""
    steps, m = up.shape
    dt, H = cfg["dt"], cfg["horizon"]
    life = np.zeros((steps, m))
    nxt = np.full(m, np.inf)  # steps until next DOWN, scanning backwards
    for t in range(steps - 1, -1, -1):
        nxt = np.where(up[t], nxt + 1, 0)
        life[t] = nxt
    life = life * dt
    censored = life >= H
    life = np.minimum(life, H)
    life[~up] = 0.0
    return life, censored & up


def link_expiration_time(P, V, iu, ju, r, cfg):
    """Classic LET (Su, Lee & Gerla, 2001) extended to 3D: assumes both UAVs
    keep their current velocity. Solves |dp + dv*tau| = r for tau > 0."""
    dp = P[:, iu, :] - P[:, ju, :]
    dv = V[:, iu, :] - V[:, ju, :]
    a = np.sum(dv * dv, -1)
    b = 2 * np.sum(dp * dv, -1)
    c = np.sum(dp * dp, -1) - r ** 2
    disc = np.maximum(b ** 2 - 4 * a * c, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = (-b + np.sqrt(disc)) / (2 * a)
    tau = np.where(a < 1e-9, cfg["horizon"], tau)
    tau = np.where(c > 0, 0.0, tau)  # already out of range
    return np.clip(tau, 0, cfg["horizon"])


# ----------------------------------------------------------------------------
# 5. One full episode -> dictionary of arrays
# ----------------------------------------------------------------------------
def generate_episode(cfg=None, **overrides):
    cfg = {**DEFAULT_CFG, **(cfg or {}), **overrides}
    rng = np.random.default_rng(cfg["seed"])
    n_keep = int(round(cfg["T"] / cfg["dt"]))            # sample slots [0, T)
    sim_cfg = {**cfg, "T": cfg["T"] + cfg["horizon"] + cfg["dt"]}   # + follow-up
    P, V = simulate_mobility(sim_cfg, rng)
    dist, snr_ls, snr_meas, iu, ju = simulate_channel(P, cfg, rng)
    if cfg["channel"] == "disk":          # classical distance-threshold link
        up_raw = dist <= comm_range(cfg)
    elif cfg["channel"] == "a2a":         # SNR threshold with shadowing
        up_raw = snr_ls >= cfg["snr_th_db"]
    else:
        raise ValueError("channel must be 'a2a' or 'disk'")
    up = debounce(up_raw, cfg["break_steps"])
    life, cens = residual_lifetime(up, cfg)
    # Hello-message observations: GPS position/velocity errors
    P_obs = P + rng.normal(0, cfg["obs_pos_std"], P.shape)
    V_obs = V + rng.normal(0, cfg["obs_vel_std"], V.shape)
    let = link_expiration_time(P_obs, V_obs, iu, ju, comm_range(cfg), cfg)
    # every kept label has a full horizon of follow-up inside the simulation
    assert up.shape[0] >= n_keep + int(round(cfg["horizon"] / cfg["dt"])) + 1
    cut = lambda x: x[:n_keep]
    P, V, P_obs, V_obs, dist, snr_ls, snr_meas, up, life, cens, let = map(
        cut, (P, V, P_obs, V_obs, dist, snr_ls, snr_meas, up, life, cens, let))
    assert np.isfinite(life).all()
    return dict(cfg=cfg, pos=P, vel=V, pos_obs=P_obs, vel_obs=V_obs, dist=dist,
                snr_ls=snr_ls, snr_meas=snr_meas, up=up, life=life, censored=cens,
                let=let, iu=iu, ju=ju)


def self_test(verbose=True):
    """Unit tests for the two properties checked in the code audit."""
    k = 2
    f = lambda seq: debounce(np.array(seq, bool)[:, None], k)[:, 0].astype(int).tolist()
    assert f([1, 0, 1]) == [1, 1, 1]          # 1-slot dip: still active
    assert f([1, 0, 0]) == [1, 1, 0]          # break declared at the 2nd miss
    assert f([1, 0, 0, 1]) == [1, 1, 0, 1]    # reconnection
    assert f([0, 0, 1]) == [0, 0, 1]          # starts down
    # causality: same history, different futures -> identical states so far
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = rng.random((40, 5)) > 0.3
        b = a.copy(); t = rng.integers(1, 39); b[t + 1:] = rng.random((39 - t, 5)) > 0.3
        assert (debounce(a, k)[:t + 1] == debounce(b, k)[:t + 1]).all()
    # labels: every sample has a full follow-up; censored <=> life == H
    ep = generate_episode(mobility="gm", seed=123, T=60.0)
    H = ep["cfg"]["horizon"]
    assert ep["up"].shape[0] == 120
    assert ((ep["life"] >= H) == ep["censored"])[ep["up"]].all()
    # a label at t must not depend on anything after t + H (finite follow-up)
    cfg = {**DEFAULT_CFG, "dt": 0.5, "horizon": 5.0}
    nH = 10
    for _ in range(50):
        u = rng.random((80, 4)) > 0.1
        full, _ = residual_lifetime(u, cfg)
        for t in rng.integers(0, 80 - nH - 1, 5):
            part, _ = residual_lifetime(u[:t + nH + 1], cfg)
            assert np.allclose(full[t], part[t])
    if verbose:
        print("self_test passed: causal link state, full-horizon labels")


if __name__ == "__main__":
    ep = generate_episode()
    cfg = ep["cfg"]
    up = ep["up"]
    print(f"Comm range R = {comm_range(cfg):.1f} m, noise = {noise_dbm(cfg):.1f} dBm")
    print(f"Steps = {up.shape[0]}, pairs = {up.shape[1]}")
    print(f"Avg active links per snapshot = {up.sum(1).mean():.1f}")
    print(f"Avg node degree = {2*up.sum(1).mean()/cfg['n_uav']:.2f}")
    m = up & ~ep["censored"]
    print(f"Uncensored samples = {m.sum()}, mean lifetime = {ep['life'][m].mean():.1f} s")
    err = np.abs(ep["let"][m] - ep["life"][m])
    print(f"LET baseline MAE = {err.mean():.2f} s")
