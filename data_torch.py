"""Feature construction and batching (v4).

v4 change (Hello information model): pair features are refreshed only in slots in
which a Hello message is received (link available). In a slot in which the link is
still active but the Hello is missed (the first k-1 unavailable slots), the pair
features of the previous slot are held, the LET is aged by one slot, and a
'missed Hello' flag is set. With k = 2 a missed slot is always preceded by a slot
with a received Hello, so 'held' always means 'value of slot t-1'.
"""
import numpy as np
try:
    import torch
except ImportError:          # numpy-only use (GBT, baselines)
    torch = None
from uav_swarm_sim import generate_episode, comm_range

FN, FE = 7, 9   # node / edge feature sizes (v4: + missed-Hello flag)


def held_mask(ep):
    """(T, M) bool: link active but Hello missed in this slot."""
    return ep["up"] & ~ep["hello"]


def hold_prev(x, miss):
    """Replace x[t, m] by x[t-1, m] where miss[t, m] (x: (T, M, ...))."""
    y = x.copy()
    t, m = np.nonzero(miss)
    assert (t > 0).all()
    y[t, m] = x[t - 1, m]
    return y


def held_let(let, ep):
    """LET as known to the receiver: in a missed slot, the previous value minus dt."""
    miss = held_mask(ep)
    out = hold_prev(let, miss)
    out[miss] = np.maximum(out[miss] - ep["cfg"]["dt"], 0.0)
    return out


def episode_features(ep, hold=True):
    cfg = ep["cfg"]
    R, H = comm_range(cfg), cfg["horizon"]
    p, v, iu, ju = ep["pos_obs"], ep["vel_obs"], ep["iu"], ep["ju"]
    up = ep["up"]
    T, N, _ = p.shape
    dp, dv = p[:, iu] - p[:, ju], v[:, iu] - v[:, ju]
    dist = np.linalg.norm(dp, axis=2)
    radial = (dp * dv).sum(2) / np.maximum(dist, 1.0)
    rel = np.linalg.norm(dv, axis=2)
    snr = ep["snr_meas"]
    dsnr = np.diff(snr, axis=0, prepend=snr[:1])
    cap = np.log2(1 + 10 ** (np.clip(snr, -20, 60) / 10))
    miss = held_mask(ep) if hold else np.zeros_like(up)
    ef = np.stack([dist / R, radial / 35, rel / 35, snr / 30, dsnr / 10, cap / 10,
                   ep["let"] / H, up.astype(float), np.zeros_like(dist)], -1)
    if hold:
        ef = hold_prev(ef, miss)
        ef[..., 6] = held_let(ep["let"], ep) / H
        ef[..., 7] = up
        ef[..., 8] = miss
    ef[~up] = 0.0
    inc = np.zeros((len(iu), N))
    inc[np.arange(len(iu)), iu] = 1.0
    inc[np.arange(len(iu)), ju] = 1.0
    deg = up.astype(float) @ inc                     # node degree (T, N)
    nf = np.concatenate([p[..., :2] / cfg["area"][0], (p[..., 2:] - cfg["alt"][0]) / 100,
                         v / 35, deg[..., None] / N], -1)
    return nf.astype(np.float32), ef.astype(np.float32), miss


class EpisodeBank:
    """Generates episodes (same seeds as Experiment 1) and stores features on device."""

    def __init__(self, mobility, episodes, channel="a2a", device="cuda", seed0=10_000, hold=True, **cfg_over):
        nodes, edges, ups, hels, lifes, cens, lets = [], [], [], [], [], [], []
        self.raw = {}
        for s in episodes:
            ep = generate_episode(mobility=mobility, channel=channel, seed=seed0 + s, **cfg_over)
            nf, ef, miss = episode_features(ep, hold)
            nodes.append(nf); edges.append(ef); ups.append(ep["up"]); hels.append(ep["up"] & ~miss)
            lifes.append(ep["life"]); cens.append(ep["censored"])
            lets.append(held_let(ep["let"], ep) if hold else ep["let"])
            self.raw[s] = dict(pos_obs=ep["pos_obs"], vel_obs=ep["vel_obs"], cfg=ep["cfg"],
                               snr_meas=ep["snr_meas"], miss=miss)
        self.cfg = ep["cfg"]
        self.H = float(self.cfg["horizon"])
        self.episodes = list(episodes)
        self.iu = torch.as_tensor(ep["iu"], device=device)
        self.ju = torch.as_tensor(ep["ju"], device=device)
        t = lambda x, dt: torch.as_tensor(np.stack(x), dtype=dt, device=device)
        self.node = t(nodes, torch.float32)          # (E, T, N, FN)
        self.edge = t(edges, torch.float16)          # (E, T, M, FE)  half precision to save memory
        self.up = t(ups, torch.bool)                 # (E, T, M) link state (active)
        self.hel = t(hels, torch.bool)               # (E, T, M) active and Hello received
        self.life = t(lifes, torch.float32)
        self.cens = t(cens, torch.bool)
        self.let = np.stack(lets)
        self.T, self.N, self.M = self.node.shape[1], self.node.shape[2], self.edge.shape[2]

    def index(self, W, stride=1):
        e = np.repeat(np.arange(len(self.episodes)), len(range(W - 1, self.T, stride)))
        t = np.tile(np.arange(W - 1, self.T, stride), len(self.episodes))
        return np.stack([e, t], 1)

    def batch(self, idx, W):
        """node: W+1 slots (t-W..t) so that a model can use the previous slot for held pairs;
        edge / up / hel: W slots (t-W+1..t)."""
        dev = self.node.device
        e = torch.as_tensor(idx[:, 0], device=dev)
        t = torch.as_tensor(idx[:, 1], device=dev)
        ts = t[:, None] - torch.arange(W - 1, -1, -1, device=dev)[None]
        tn = (t[:, None] - torch.arange(W, -1, -1, device=dev)[None]).clamp(min=0)
        ee = e[:, None].expand_as(ts)
        en = e[:, None].expand_as(tn)
        return (self.node[en, tn], self.edge[ee, ts].float(), self.up[ee, ts], self.hel[ee, ts],
                self.life[e, t], self.cens[e, t], self.up[e, t])
