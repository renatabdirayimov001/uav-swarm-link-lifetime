"""
Experiments of the revised paper (code v4).

  python run_v4.py main      # A2A, link shadowing: all predictors, censoring-aware metrics, no-aggregation ablation
  python run_v4.py motion    # alternative shadowing (displacement-driven correlation), all mobility models
  python run_v4.py general   # trained models tested on N = 20 / 50 and shadowing std 4 dB, without retraining
  python run_v4.py timing    # single-slot inference time of all predictors on one device

Every finished run is written to OUT/runs/*.csv, so the script can be restarted after an interruption.
Change in v4: Hello information model (pair features are held during a missed Hello), see data_torch.py.
"""
import sys, os, time, itertools, json, glob, copy
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingRegressor
import uav_swarm_sim as sim, baselines as bl, extra as ex, data_torch as dt
from models import MLPModel, LSTMModel, STGNN, plain_huber
from data_torch import EpisodeBank, FN, FE

MODE = sys.argv[1] if len(sys.argv) > 1 else "main"
QUICK = os.environ.get("QUICK", "0") == "1"
NEURAL_ONLY = os.environ.get("NEURAL_ONLY", "0") == "1"     # Colab GPU part: neural predictors only
NO_NEURAL = os.environ.get("NO_NEURAL", "0") == "1"         # CPU part: analytical, SNR trend and GBT
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
OUT = os.environ.get("OUT", "out_v4")
DEV = "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
torch.set_num_threads(max(1, os.cpu_count()))
W, H = 8, 60.0
EPOCHS, PATIENCE, BATCH, LR = 30, 6, 32, 1e-3
SEEDS = [0, 1, 2]
TRAIN, VAL, TEST = list(range(0, 14)), list(range(14, 17)), list(range(17, 23))
MOBILITY = ["rpgm", "gm", "rwp"]
KF_GRID = list(itertools.product([2, 3, 5, 10], [0.5, 1.0, 2.0]))
if QUICK:
    EPOCHS, SEEDS, TRAIN, VAL, TEST, MOBILITY = 1, [0], [0], [14], [17], ["gm"]
    KF_GRID = [(3, 1.0)]
for d in ("", "/runs", "/ckpt", "/logs"):
    os.makedirs(OUT + d, exist_ok=True)
print("device", DEV, "| mode", MODE, "| out", OUT, flush=True)


# ------------------------------------------------------------------ helpers
def build(name, bank):
    if name == "MLP":            return MLPModel(FN, FE, H)
    if name == "LSTM":           return LSTMModel(FN, FE, H)
    if name == "StaticGNN":      return STGNN(FN, FE, H, bank.N, bank.iu, bank.ju, temporal=False)
    if name == "StaticGNN-noagg": return STGNN(FN, FE, H, bank.N, bank.iu, bank.ju, temporal=False, agg=False)
    if name == "STGNN":          return STGNN(FN, FE, H, bank.N, bank.iu, bank.ju, temporal=True)
    raise ValueError(name)


@torch.no_grad()
def predict(model, bank, bs=64):
    model.eval()
    idx = bank.index(W)
    pred = np.zeros((len(bank.episodes), bank.T, bank.M), np.float32)
    for k in range(0, len(idx), bs):
        b = idx[k:k + bs]
        node, edge, up, hel, *_ = bank.batch(b, W)
        pred[b[:, 0], b[:, 1]] = model(node, edge, up, hel, bank.iu, bank.ju).float().cpu().numpy()
    return pred


def val_mae(model, bank):
    pred = predict(model, bank)
    life, cens, up = bank.life.cpu().numpy(), bank.cens.cpu().numpy(), bank.up.cpu().numpy()
    m = up[:, W - 1:] & ~cens[:, W - 1:]
    return float(np.abs(pred[:, W - 1:][m] - life[:, W - 1:][m]).mean())


def train(name, seed, tr, va, log):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(name, tr).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    idx = tr.index(W)
    best, best_state, bad, hist = np.inf, None, 0, []
    for epoch in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0.0
        perm = np.random.permutation(len(idx))
        for k in range(0, len(perm), BATCH):
            node, edge, up, hel, y, c, m = tr.batch(idx[perm[k:k + BATCH]], W)
            loss = plain_huber(model(node, edge, up, hel, tr.iu, tr.ju), y, c, m, tr.H)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        v = val_mae(model, va); sched.step(v)
        hist.append(dict(epoch=epoch, train_loss=tot / max(1, len(perm) // BATCH), val_MAE=v, sec=time.time() - t0))
        print(f"    {log} ep{epoch:02d} loss {hist[-1]['train_loss']:.3f} val MAE {v:.3f} ({hist[-1]['sec']:.0f}s)", flush=True)
        if v < best - 1e-3:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    pd.DataFrame(hist).to_csv(f"{OUT}/logs/{log}.csv", index=False)
    model.load_state_dict(best_state)
    return model


def score(pred, bank, method, tag, seed, extra_cols=None):
    """pred (E, T, M) -> one row per test episode (slots t >= W-1)."""
    life, cens, up = bank.life.cpu().numpy(), bank.cens.cpu().numpy(), bank.up.cpu().numpy()
    rows = []
    for k, e in enumerate(bank.episodes):
        sl = slice(W - 1, None)
        a = (pred[k, sl], life[k, sl], cens[k, sl], up[k, sl])
        r = bl.evaluate(*a); r.update(ex.evaluate_ext(*a, H))
        r.update(method=method, setting=tag, seed=seed, episode=e, **(extra_cols or {}))
        rows.append(r)
    return rows


def kf_pred(bank, tau, q):
    out = []
    for e in bank.episodes:
        raw = bank.raw[e]
        kf = bl.kf_let(raw["pos_obs"], raw["vel_obs"], bank.iu.cpu().numpy(), bank.ju.cpu().numpy(),
                       sim.comm_range(raw["cfg"]), raw["cfg"], tau, q)
        miss = raw["miss"]                     # held value, aged by one slot
        kf2 = dt.hold_prev(kf, miss); kf2[miss] = np.maximum(kf2[miss] - raw["cfg"]["dt"], 0)
        out.append(kf2)
    return np.stack(out).astype(np.float32)


def tune_kf(va):
    best, rows = None, []
    for tau, q in KF_GRID:
        p = kf_pred(va, tau, q)
        life, cens, up = va.life.cpu().numpy(), va.cens.cpu().numpy(), va.up.cpu().numpy()
        m = up & ~cens
        mae = float(np.abs(p[m] - life[m]).mean())
        rows.append(dict(tau=tau, q=q, val_MAE=mae))
        if best is None or mae < best[2]: best = (tau, q, mae)
    return best[0], best[1], pd.DataFrame(rows)


def snr_trend_pred(bank):
    out = []
    for e in bank.episodes:
        raw = bank.raw[e]
        snr = dt.hold_prev(raw["snr_meas"], raw["miss"])
        up = bank.up[bank.episodes.index(e)].cpu().numpy()
        out.append(ex.snr_trend(snr, up, raw["cfg"]["snr_th_db"], H, raw["cfg"]["dt"], W))
    return np.stack(out).astype(np.float32)


def gbt_xy(bank, k):
    nf = bank.node[k].cpu().numpy(); ef = bank.edge[k].float().cpu().numpy()
    miss = bank.raw[bank.episodes[k]]["miss"]
    return ex.gbt_features(nf, ef, bank.iu.cpu().numpy(), bank.ju.cpu().numpy(), W, miss)


def train_gbt(tr, seed, rows=600_000):
    X, y = [], []
    for k in range(len(tr.episodes)):
        F = gbt_xy(tr, k)[W - 1:]; u = tr.up[k, W - 1:].cpu().numpy()
        X.append(F[u]); y.append(tr.life[k, W - 1:].cpu().numpy()[u])
    X, y = np.concatenate(X), np.concatenate(y)
    i = np.random.default_rng(seed).choice(len(y), min(rows, len(y)), replace=False)
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.1,
                                         max_leaf_nodes=63, random_state=seed).fit(X[i], y[i])


def gbt_pred(g, bank):
    out = np.zeros((len(bank.episodes), bank.T, bank.M), np.float32)
    for k in range(len(bank.episodes)):
        F = gbt_xy(bank, k); u = bank.up[k].cpu().numpy().copy(); u[:W - 1] = False
        out[k][u] = np.clip(g.predict(F[u]), 0, H)
    return out


def done(f):
    return os.path.exists(f)


def run_setting(tag, mob, cfg_over, methods, neural):
    """Train / evaluate all predictors for one (setting, mobility)."""
    print(f"=== {tag} / {mob} ===", flush=True)
    tr = EpisodeBank(mob, TRAIN, device=DEV, **cfg_over)
    va = EpisodeBank(mob, VAL, device=DEV, **cfg_over)
    te = EpisodeBank(mob, TEST, device=DEV, **cfg_over)
    f = f"{OUT}/runs/{tag}_{mob}_analytical.csv"
    if not done(f) and not NEURAL_ONLY:
        tau, q, kt = tune_kf(va); kt.to_csv(f"{OUT}/runs/{tag}_{mob}_kf_tuning.csv", index=False)
        rows = score(te.let.astype(np.float32), te, "LET", tag, 0) + score(kf_pred(te, tau, q), te, "KF-LET", tag, 0,
                     dict(tau=tau, q=q)) + score(snr_trend_pred(te), te, "SNR-trend", tag, 0)
        pd.DataFrame(rows).to_csv(f, index=False); print("  analytical done", flush=True)
    for seed in SEEDS:
        f = f"{OUT}/runs/{tag}_{mob}_GBT_s{seed}.csv"
        if "GBT" in methods and not done(f) and not NEURAL_ONLY:
            g = train_gbt(tr, seed)
            import pickle; pickle.dump(g, open(f"{OUT}/ckpt/{tag}_{mob}_GBT_s{seed}.pkl", "wb"))
            pd.DataFrame(score(gbt_pred(g, te), te, "GBT", tag, seed)).to_csv(f, index=False)
            print(f"  GBT s{seed} done", flush=True)
    for name, seed in itertools.product(neural, SEEDS):
        f = f"{OUT}/runs/{tag}_{mob}_{name}_s{seed}.csv"
        if done(f) or NO_NEURAL: continue
        t0 = time.time()
        model = train(name, seed, tr, va, log=f"{tag}_{mob}_{name}_s{seed}")
        torch.save(model.state_dict(), f"{OUT}/ckpt/{tag}_{mob}_{name}_s{seed}.pt")
        pd.DataFrame(score(predict(model, te), te, name, tag, seed,
                           dict(train_min=(time.time() - t0) / 60))).to_csv(f, index=False)
        print(f"  {name} s{seed} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    del tr, va, te
    if DEV == "cuda": torch.cuda.empty_cache()


NEURAL_MAIN = ["MLP", "LSTM", "StaticGNN", "STGNN", "StaticGNN-noagg"]
NEURAL_ALT = ["LSTM", "StaticGNN"]

if MODE == "main":
    for mob in MOBILITY:
        run_setting("main", mob, {}, ["GBT"], NEURAL_MAIN)
elif MODE == "motion":
    for mob in MOBILITY:
        run_setting("motion", mob, dict(shadow_model="motion"), ["GBT"], NEURAL_ALT)
elif MODE == "disk":
    # analytical predictors under the disk link model (KF-LET setting tuned on the A2A validation episodes)
    for mob in MOBILITY:
        f = f"{OUT}/runs/disk_{mob}_analytical.csv"
        if done(f): continue
        te = EpisodeBank(mob, TEST, channel="disk", device=DEV)
        kt = pd.read_csv(f"{OUT}/runs/main_{mob}_analytical.csv"); kt = kt[kt.method == "KF-LET"].iloc[0]
        rows = score(te.let.astype(np.float32), te, "LET", "disk", 0) + \
               score(kf_pred(te, kt.tau, kt.q), te, "KF-LET", "disk", 0, dict(tau=kt.tau, q=kt.q))
        pd.DataFrame(rows).to_csv(f, index=False); print("  disk", mob, "done", flush=True)
        del te
elif MODE == "general":
    import pickle
    tests = {"N20": dict(n_uav=20), "N50": dict(n_uav=50), "std4": dict(shadow_std=4.0)}
    for mob, (tag, over) in itertools.product(MOBILITY, tests.items()):
        part = "neural" if NEURAL_ONLY else ("classic" if NO_NEURAL else "all")
        f = f"{OUT}/runs/general_{tag}_{mob}_{part}.csv"
        if done(f): continue
        te = EpisodeBank(mob, TEST, device=DEV, **over)
        rows = []
        if not NEURAL_ONLY:
            kt = pd.read_csv(f"{OUT}/runs/main_{mob}_analytical.csv"); kt = kt[kt.method == "KF-LET"].iloc[0]
            rows += score(te.let.astype(np.float32), te, "LET", tag, 0) + score(kf_pred(te, kt.tau, kt.q), te, "KF-LET", tag, 0)
        for seed in SEEDS:
            if not NEURAL_ONLY:
                g = pickle.load(open(f"{OUT}/ckpt/main_{mob}_GBT_s{seed}.pkl", "rb"))
                rows += score(gbt_pred(g, te), te, "GBT", tag, seed)
            for name in ([] if NO_NEURAL else NEURAL_ALT):
                m = build(name, te).to(DEV)
                sd = torch.load(f"{OUT}/ckpt/main_{mob}_{name}_s{seed}.pt", map_location=DEV)
                sd = {k: v for k, v in sd.items() if k not in ("iu", "ju")}
                m.load_state_dict(sd, strict=False)
                rows += score(predict(m, te), te, name, tag, seed)
        pd.DataFrame(rows).to_csv(f, index=False); print("  general", tag, mob, "done", flush=True)
        del te
elif MODE == "timing":
    import pickle, platform
    te = EpisodeBank("gm", TEST[:1], device=DEV)
    idx = te.index(W)[200:201]
    rows = []
    for name in ["MLP", "LSTM", "StaticGNN", "STGNN"]:
        m = build(name, te).to(DEV)
        m.load_state_dict({k: v for k, v in torch.load(f"{OUT}/ckpt/main_gm_{name}_s0.pt", map_location=DEV).items()
                           if k not in ("iu", "ju")}, strict=False); m.eval()
        node, edge, up, hel, *_ = te.batch(idx, W)
        with torch.no_grad():
            for _ in range(10): m(node, edge, up, hel, te.iu, te.ju)
            ts = []
            for _ in range(200):
                t0 = time.perf_counter(); m(node, edge, up, hel, te.iu, te.ju)
                if DEV == "cuda": torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
        rows.append(dict(method=name, params=sum(p.numel() for p in m.parameters()),
                         median_ms=np.median(ts), p95_ms=np.percentile(ts, 95)))
    g = pickle.load(open(f"{OUT}/ckpt/main_gm_GBT_s0.pkl", "rb"))
    F = gbt_xy(te, 0)[200 + W - 1]; u = te.up[0, 200 + W - 1].cpu().numpy(); Xs = F[u]
    for _ in range(10): g.predict(Xs)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter(); g.predict(Xs); ts.append((time.perf_counter() - t0) * 1e3)
    rows.append(dict(method="GBT", params=np.nan, median_ms=np.median(ts), p95_ms=np.percentile(ts, 95)))
    df = pd.DataFrame(rows); df["device"] = (torch.cuda.get_device_name(0) if DEV == "cuda" else platform.processor() or "CPU")
    df["threads"] = torch.get_num_threads(); df["active_links"] = int(u.sum())
    df.to_csv(f"{OUT}/timing_{DEV}.csv", index=False); print(df)
