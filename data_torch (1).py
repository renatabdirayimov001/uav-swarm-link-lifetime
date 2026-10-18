"""Feature construction and batching for Experiment 2 (all tensors kept on the GPU)."""
import numpy as np
import torch
from uav_swarm_sim import generate_episode, comm_range

FN, FE = 7, 8   # node / edge feature sizes


def episode_features(ep):
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
    ef = np.stack([dist / R, radial / 35, rel / 35, snr / 30, dsnr / 10, cap / 10,
                   ep["let"] / H, up.astype(float)], -1)
    ef[~up] = 0.0
    inc = np.zeros((len(iu), N))
    inc[np.arange(len(iu)), iu] = 1.0
    inc[np.arange(len(iu)), ju] = 1.0
    deg = up.astype(float) @ inc                     # node degree (T, N)
    nf = np.concatenate([p[..., :2] / cfg["area"][0], (p[..., 2:] - cfg["alt"][0]) / 100,
                         v / 35, deg[..., None] / N], -1)
    return nf.astype(np.float32), ef.astype(np.float32)


class EpisodeBank:
    """Generates episodes (same seeds as Experiment 1) and stores features on device."""

    def __init__(self, mobility, episodes, channel="a2a", device="cuda", seed0=10_000, **cfg_over):
        nodes, edges, ups, lifes, cens, lets = [], [], [], [], [], []
        self.raw = {}
        for s in episodes:
            ep = generate_episode(mobility=mobility, channel=channel, seed=seed0 + s, **cfg_over)
            nf, ef = episode_features(ep)
            nodes.append(nf); edges.append(ef); ups.append(ep["up"])
            lifes.append(ep["life"]); cens.append(ep["censored"]); lets.append(ep["let"])
            self.raw[s] = dict(pos_obs=ep["pos_obs"], vel_obs=ep["vel_obs"], cfg=ep["cfg"])
        self.cfg = ep["cfg"]
        self.H = float(self.cfg["horizon"])
        self.episodes = list(episodes)
        self.iu = torch.as_tensor(ep["iu"], device=device)
        self.ju = torch.as_tensor(ep["ju"], device=device)
        t = lambda x, dt: torch.as_tensor(np.stack(x), dtype=dt, device=device)
        self.node = t(nodes, torch.float32)          # (E, T, N, FN)
        self.edge = t(edges, torch.float16)          # (E, T, M, FE)  half precision to save memory
        self.up = t(ups, torch.bool)                 # (E, T, M)
        self.life = t(lifes, torch.float32)
        self.cens = t(cens, torch.bool)
        self.let = np.stack(lets)
        self.T, self.N, self.M = self.node.shape[1], self.node.shape[2], self.edge.shape[2]

    def index(self, W, stride=1):
        e = np.repeat(np.arange(len(self.episodes)), len(range(W - 1, self.T, stride)))
        t = np.tile(np.arange(W - 1, self.T, stride), len(self.episodes))
        return np.stack([e, t], 1)

    def batch(self, idx, W):
        dev = self.node.device
        e = torch.as_tensor(idx[:, 0], device=dev)
        t = torch.as_tensor(idx[:, 1], device=dev)
        ts = t[:, None] - torch.arange(W - 1, -1, -1, device=dev)[None]
        ee = e[:, None].expand_as(ts)
        return (self.node[ee, ts], self.edge[ee, ts].float(), self.up[ee, ts],
                self.life[e, t], self.cens[e, t], self.up[e, t])
