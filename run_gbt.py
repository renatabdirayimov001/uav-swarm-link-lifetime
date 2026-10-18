import sys, types, numpy as np, pandas as pd, time
import sklearn.metrics, scipy.stats
from sklearn.ensemble import HistGradientBoostingRegressor
sys.modules['torch'] = types.ModuleType('torch')          # data_torch imports torch; only numpy features are used here
import uav_swarm_sim as s, data_torch as d, extra as x, baselines as b
W, H = 8, 60.0
TRAIN, TEST, SEEDS, ROWS = range(0, 14), range(17, 23), [0, 1, 2], 600_000
rows, t0 = [], time.time()
for mob in ["rpgm", "gm", "rwp"]:
    X, y = [], []
    for e in TRAIN:
        ep = s.generate_episode(mobility=mob, seed=10000 + e); nf, ef = d.episode_features(ep)
        F = x.gbt_features(nf, ef, ep["iu"], ep["ju"], W)[W-1:]; u = ep["up"][W-1:]
        X.append(F[u]); y.append(ep["life"][W-1:][u])
    X, y = np.concatenate(X), np.concatenate(y)
    print(mob, "train rows", len(y), f"{time.time()-t0:.0f}s", flush=True)
    models = []
    for seed in SEEDS:
        i = np.random.default_rng(seed).choice(len(y), min(ROWS, len(y)), replace=False)
        models.append(HistGradientBoostingRegressor(loss="absolute_error", max_iter=200, learning_rate=0.1,
                      max_leaf_nodes=63, random_state=seed).fit(X[i], y[i]))
    del X, y
    for e in TEST:
        ep = s.generate_episode(mobility=mob, seed=10000 + e); nf, ef = d.episode_features(ep)
        F = x.gbt_features(nf, ef, ep["iu"], ep["ju"], W); up = ep["up"]; u = up.copy(); u[:W-1] = False
        sl = slice(W-1, None); a = lambda p: (p[sl], ep["life"][sl], ep["censored"][sl], up[sl])
        cand = [("SNR-trend", 0, x.snr_trend(ep["snr_meas"], up, ep["cfg"]["snr_th_db"], H, ep["cfg"]["dt"], W)),
                ("LET", 0, ep["let"])]
        for seed, g in zip(SEEDS, models):
            p = np.zeros(up.shape, np.float32); p[u] = np.clip(g.predict(F[u]), 0, H); cand.append(("GBT", seed, p))
        for name, seed, p in cand:
            r = b.evaluate(*a(p)); r.update(x.evaluate_ext(*a(p), H))
            r.update(method=name, mobility=mob, seed=seed, episode=e); rows.append(r)
    print(mob, "done", f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv("results_gbt.csv", index=False)
print("ALL DONE")
