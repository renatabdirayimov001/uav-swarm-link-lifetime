"""v5 analyses on the main A2A configuration (test episodes):
  * dataset statistics: episodes, link intervals and break events (length bias)
  * event-level metrics: interval-weighted MAE, break recall at 2 s / 5 s lead, false-alarm rate
  * route-level (protocol-oriented) metrics: widest-path routing on predicted lifetimes,
    route breaks per minute, realised route lifetime and hop count
Predictions: LET, LET-cal, KF-LET, GBT (seed 0), MLP/LSTM/StaticGNN/StaticGNN-noagg/ST-GNN (seed 0, CPU inference),
oracle (true residual lifetime) and min-hop routing (no lifetime information)."""
import os, sys, pickle, heapq, itertools, numpy as np, pandas as pd, torch
os.environ.setdefault("OUT", "./out_v4")
sys.argv = [sys.argv[0], "none"]
import run_v4 as R
from data_torch import EpisodeBank
import uav_swarm_sim as sim
torch.set_num_threads(2)
W, H, DT = 8, 60.0, 0.5
CK = {"cpu": "./out_v4/ckpt", "nn": "./colab_out/ckpt"}
OUTD = "./v5_analysis"; os.makedirs(OUTD, exist_ok=True)


def predictions(mob):
    f = f"{OUTD}/pred_{mob}.npz"
    if os.path.exists(f):
        return dict(np.load(f))
    tr = EpisodeBank(mob, R.TRAIN, device="cpu"); te = EpisodeBank(mob, R.TEST, device="cpu")
    kt = pd.read_csv(f"{R.OUT}/runs/main_{mob}_analytical.csv"); kt = kt[kt.method == "KF-LET"].iloc[0]
    P = {"LET": te.let.astype(np.float32), "LET-cal": R.let_cal(tr, te)[0], "KF-LET": R.kf_pred(te, kt.tau, kt.q)}
    del tr
    P["GBT"] = R.gbt_pred(pickle.load(open(f"{CK['cpu']}/main_{mob}_GBT_s0.pkl", "rb")), te)
    for name in ["MLP", "LSTM", "StaticGNN", "StaticGNN-noagg", "STGNN"]:
        m = R.build(name, te)
        sd = torch.load(f"{CK['nn']}/main_{mob}_{name}_s0.pt", map_location="cpu")
        m.load_state_dict({k: v for k, v in sd.items() if k not in ("iu", "ju")}, strict=False)
        P[name.replace("STGNN", "ST-GNN")] = R.predict(m, te, bs=32)
        print("  ", mob, name, flush=True)
    P["Oracle"] = te.life.numpy()
    P["_up"] = te.up.numpy(); P["_cens"] = te.cens.numpy(); P["_iu"] = te.iu.numpy(); P["_ju"] = te.ju.numpy()
    np.savez_compressed(f, **P)
    return P


def intervals(up_col):
    """(start, end) slot index of maximal runs of up (end exclusive)."""
    d = np.diff(np.concatenate([[0], up_col.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def event_metrics(P, mob):
    up, cens, life = P["_up"], P["_cens"], P["Oracle"]
    E, T, M = up.shape
    rows = []
    for name in [k for k in P if not k.startswith("_") and k != "Oracle"]:
        p = P[name]
        act = up.copy(); act[:, :W - 1] = False
        neg = act & ((life > 10.0) | cens)
        thr = np.quantile(p[neg], 0.05)          # operating point with a 5% false-alarm rate
        int_err, rec2, rec5, nev = [], [], [], 0
        for e, m in itertools.product(range(E), range(M)):
            for a, b in intervals(up[e, :, m]):
                a2 = max(a, W - 1)
                if b >= T or a2 >= b or cens[e, a2, m]:   # interval ends after the sampled window or censored
                    continue
                int_err.append(np.abs(p[e, a2:b, m] - life[e, a2:b, m]).mean())
                nev += 1
                for lead, store in ((2.0, rec2), (5.0, rec5)):
                    t = b - int(lead / DT)
                    if t >= W - 1 and t >= a:
                        store.append(p[e, t, m] <= thr)
        rows.append(dict(mobility=mob, method=name, breaks=nev, MAE_interval=np.mean(int_err),
                         recall_lead2=np.mean(rec2), recall_lead5=np.mean(rec5), threshold=thr))
    return rows


def widest_path(n, adj, w, s, d):
    """Max-bottleneck path, ties broken by fewer hops. adj: list of neighbour lists, w[(i,j)] weight."""
    best = {s: (np.inf, 0)}; prev = {}
    pq = [(-np.inf, 0, s)]
    while pq:
        nb, hops, u = heapq.heappop(pq); bw = -nb
        if (bw, -hops) < (best[u][0], -best[u][1]): continue
        if u == d: break
        for v in adj[u]:
            c = (min(bw, w[(u, v)]), hops + 1)
            if v not in best or (c[0], -c[1]) > (best[v][0], -best[v][1]):
                best[v] = c; prev[v] = u; heapq.heappush(pq, (-c[0], c[1], v))
    if d not in best: return None
    path = [d]
    while path[-1] != s: path.append(prev[path[-1]])
    return path[::-1]


def route_metrics(P, mob, n_pairs=30, seed=0):
    up, cens, life = P["_up"], P["_cens"], P["Oracle"]
    iu, ju = P["_iu"], P["_ju"]
    E, T, M = up.shape; N = int(max(iu.max(), ju.max())) + 1
    kidx = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(iu, ju))}
    rng = np.random.default_rng(seed)
    pairs = [tuple(rng.choice(N, 2, replace=False)) for _ in range(n_pairs)]
    methods = [k for k in P if not k.startswith("_")] + ["MinHop"]
    rows = []
    for name in methods:
        breaks, conn_time, lifes, hops = 0, 0.0, [], []
        for e in range(E):
            for s, d in pairs:
                t = W - 1
                while t < T:
                    m_act = np.where(up[e, t])[0]
                    adj = [[] for _ in range(N)]; w = {}
                    pw = (np.ones(M) if name == "MinHop" else P[name][e, t])
                    for k in m_act:
                        i, j = iu[k], ju[k]; adj[i].append(j); adj[j].append(i)
                        w[(i, j)] = w[(j, i)] = (1.0 if name == "MinHop" else float(pw[k]))
                    if name == "MinHop":
                        for a in adj: rng.shuffle(a)
                    path = widest_path(N, adj, w, s, d)
                    if path is None:
                        t += 1; continue
                    ks = [kidx[(min(a, b), max(a, b))] for a, b in zip(path[:-1], path[1:])]
                    L = min(life[e, t, k] for k in ks); censored = L >= H
                    steps = max(1, int(round(L / DT)))
                    steps = min(steps, T - t)
                    conn_time += steps * DT; lifes.append(L); hops.append(len(path) - 1)
                    if not censored and t + steps < T: breaks += 1
                    t += steps
        rows.append(dict(mobility=mob, method=name, route_breaks_per_min=60 * breaks / conn_time,
                         mean_route_life=np.mean(lifes), mean_hops=np.mean(hops)))
        print("   route", mob, name, round(60 * breaks / conn_time, 3), flush=True)
    return rows


def dataset_counts():
    rows = []
    for mob, ch in itertools.product(["rpgm", "gm", "rwp"], ["a2a", "disk"]):
        ints, brk, samp = 0, 0, 0
        for s in range(23):
            ep = sim.generate_episode(mobility=mob, channel=ch, seed=10000 + s)
            up = ep["up"]; T = up.shape[0]
            for m in range(up.shape[1]):
                for a, b in intervals(up[:, m]):
                    ints += 1; brk += b < T
            samp += int(up.sum())
        rows.append(dict(mobility=mob, channel=ch, episodes=23, link_intervals=ints, break_events=brk, samples=samp,
                         samples_per_interval=samp / ints))
        print("  counts", mob, ch, ints, brk, flush=True)
    return rows


if __name__ == "__main__":
    what = os.environ.get("WHAT", "all")
    if what in ("all", "counts"):
        pd.DataFrame(dataset_counts()).to_csv(f"{OUTD}/dataset_counts.csv", index=False)
    ev, rt = [], []
    if what in ("all", "events", "routes"):
        for mob in ["rpgm", "gm", "rwp"]:
            P = predictions(mob)
            if what in ("all", "events"): ev += event_metrics(P, mob)
            if what in ("all", "routes"): rt += route_metrics(P, mob)
        if ev: pd.DataFrame(ev).to_csv(f"{OUTD}/event_metrics.csv", index=False)
        if rt: pd.DataFrame(rt).to_csv(f"{OUTD}/route_metrics.csv", index=False)
    print("ANALYSIS_DONE")
