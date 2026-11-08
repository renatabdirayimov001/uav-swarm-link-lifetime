"""
Experiments of the third revision (code v6).

GPU part (neural predictors), run on the training computer:
  python run_v6.py check     # environment check + a 1-epoch smoke test (about 2-5 min)
  python run_v6.py reeval    # published checkpoints (main + spatial) evaluated on the 30 test episodes, no training
  python run_v6.py fading    # realistic Hello reception (Rician-faded SNR): LSTM, StaticGNN, StaticGNN-noagg x 3 seeds
  python run_v6.py window    # LSTM with W = 4, 8, 16 (seed 0)
  python run_v6.py tune      # equal-budget random search on the validation episodes (main + spatial shadowing)
  python run_v6.py final     # best configuration of each model x 3 seeds, evaluated on the test episodes
  python run_v6.py pack      # zip everything that has to be sent back (results_v6_<computer>.zip)
  python run_v6.py all       # reeval, fading, window, tune, final, pack in this order

CPU part (NO_NEURAL=1), run separately:
  python run_v6.py ref       # LET, KF-LET, SNR trend, calibrated LET, GBT (fixed setting) on the 30 test episodes
  python run_v6.py letsnr    # SNR-corrected LET (and its calibrated version), main / spatial / fading
  python run_v6.py ablation  # GBT feature groups, no absolute position, + neighbourhood summaries
  then fading, window, tune, final with NO_NEURAL=1

Environment variables:
  NEURAL_ONLY=1   neural predictors only (the default; the CPU part is run separately)
  NO_NEURAL=1     analytical predictors and GBT only (CPU part)
  QUICK=1         tiny smoke-test configuration
  OUT=out_v6      output folder
  TUNE_N=10       configurations per model in the random search (configuration 0 = the fixed v1-v5 setting)

Every finished run is written to OUT/runs/*.csv, so any mode can be interrupted and started again;
finished runs are skipped.

Changes with respect to run_v4.py:
  * 30 test episodes (seeds 10017-10046) instead of 6 for every new result;
  * the width, depth, learning rate and weight decay of the models are arguments;
  * the simulator option hello_model="fading" (uav_swarm_sim.py).
The training procedure (Huber loss, AdamW, 30 epochs, early stopping after 6, batch 32, model selection on the
validation MAE of uncensored links) is unchanged.
"""
import sys, os, time, itertools, json, copy, glob, platform, zipfile, pickle
import numpy as np, pandas as pd, torch
from sklearn.ensemble import HistGradientBoostingRegressor
import uav_swarm_sim as sim, baselines as bl, extra as ex, data_torch as dt, extra_v6 as x6
from models import MLPModel, LSTMModel, STGNN, plain_huber
from data_torch import EpisodeBank, FN, FE

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
QUICK = os.environ.get("QUICK", "0") == "1" or MODE == "check"
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
HAS_CUDA = torch.cuda.is_available() and not FORCE_CPU
HAS_MPS = (not HAS_CUDA and not FORCE_CPU and getattr(torch.backends, "mps", None) is not None
           and torch.backends.mps.is_available())                        # Apple Silicon
NO_NEURAL = os.environ.get("NO_NEURAL", "0") == "1"
NEURAL_ONLY = os.environ.get("NEURAL_ONLY", "1") == "1" and not NO_NEURAL
DEV = "cuda" if HAS_CUDA else ("mps" if HAS_MPS else "cpu")
OUT = os.environ.get("OUT", "out_v6" if not QUICK else "out_v6_quick")
TUNE_N = int(os.environ.get("TUNE_N", "10"))
torch.set_num_threads(max(1, os.cpu_count() or 1))

W_DEFAULT, H = 8, 60.0
W = W_DEFAULT
EPOCHS, PATIENCE, BATCH = 30, 6, 32
SEEDS = [0, 1, 2]
TRAIN, VAL, TEST = list(range(0, 14)), list(range(14, 17)), list(range(17, 47))   # 30 test episodes
MOBILITY = ["rpgm", "gm", "rwp"]
NEURAL_TUNED = ["LSTM", "StaticGNN", "StaticGNN-noagg"]
SETTINGS = {"main": {}, "spatial": dict(shadow_model="spatial"), "fading": dict(hello_model="fading")}
KF_GRID = list(itertools.product([0.1, 0.25, 0.5, 1, 2, 3, 5, 10], [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]))
if QUICK:
    EPOCHS, SEEDS, TRAIN, VAL, TEST, MOBILITY = 1, [0], [0, 1], [14], [17], ["gm"]
    TUNE_N, KF_GRID = 2, [(0.1, 0.05)]

# fixed v1-v5 hyperparameters (configuration 0 of the search)
DEFAULT_HP = {"MLP": dict(d=128, layers=3, lr=1e-3, wd=1e-5),
              "LSTM": dict(d=64, layers=2, lr=1e-3, wd=1e-5),
              "StaticGNN": dict(d=64, layers=2, lr=1e-3, wd=1e-5),
              "StaticGNN-noagg": dict(d=64, layers=2, lr=1e-3, wd=1e-5),
              "STGNN": dict(d=64, layers=2, lr=1e-3, wd=1e-5)}
SPACE = dict(d=[32, 64, 128], layers=[1, 2, 3], lr=[3e-4, 1e-3, 3e-3], wd=[1e-5, 1e-4])
GBT_DEFAULT = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=20)
GBT_SPACE = dict(max_iter=[100, 200, 400], learning_rate=[0.05, 0.1, 0.2], max_leaf_nodes=[31, 63, 127],
                 min_samples_leaf=[20, 100])


def configs(model):
    """Configuration 0 = fixed setting; 1..TUNE_N-1 = random draws from SPACE with a FIXED seed, so that
    every neural model is searched over exactly the same list of (d, layers, lr, wd)."""
    rng = np.random.default_rng(2026)
    space, default = (GBT_SPACE, GBT_DEFAULT) if model == "GBT" else (SPACE, DEFAULT_HP[model])
    out, seen = [dict(default)], {json.dumps(default, sort_keys=True)}
    while len(out) < TUNE_N:
        c = {k: v[rng.integers(len(v))] for k, v in space.items()}
        c = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in c.items()}
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key); out.append(c)
    return out


for d_ in ("", "/runs", "/ckpt", "/logs"):
    os.makedirs(OUT + d_, exist_ok=True)


def free_gpu():
    if DEV == "cuda": torch.cuda.empty_cache()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
    with open(f"{OUT}/progress.log", "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


# ------------------------------------------------------------------ models
def build(name, bank, hp=None):
    hp = {**DEFAULT_HP[name], **(hp or {})}
    if name == "MLP":             return MLPModel(FN, FE, H, d=hp["d"], layers=hp["layers"])
    if name == "LSTM":            return LSTMModel(FN, FE, H, d=hp["d"], layers=hp["layers"])
    if name == "StaticGNN":       return STGNN(FN, FE, H, bank.N, bank.iu, bank.ju, d=hp["d"], layers=hp["layers"], temporal=False)
    if name == "StaticGNN-noagg": return STGNN(FN, FE, H, bank.N, bank.iu, bank.ju, d=hp["d"], layers=hp["layers"], temporal=False, agg=False)
    if name == "STGNN":           return STGNN(FN, FE, H, bank.N, bank.iu, bank.ju, d=hp["d"], layers=hp["layers"], temporal=True)
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


def val_mae(pred, bank):
    life, cens, up = bank.life.cpu().numpy(), bank.cens.cpu().numpy(), bank.up.cpu().numpy()
    m = up[:, W - 1:] & ~cens[:, W - 1:]
    return float(np.abs(pred[:, W - 1:][m] - life[:, W - 1:][m]).mean())


def train(name, seed, tr, va, tag, hp=None):
    hp = {**DEFAULT_HP[name], **(hp or {})}
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(name, tr, hp).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    idx = tr.index(W)
    best, best_state, bad, hist = np.inf, None, 0, []
    for epoch in range(EPOCHS):
        model.train(); t0 = time.time(); tot, nb = 0.0, 0
        perm = np.random.permutation(len(idx))
        for k in range(0, len(perm), BATCH):
            node, edge, up, hel, y, c, m = tr.batch(idx[perm[k:k + BATCH]], W)
            loss = plain_huber(model(node, edge, up, hel, tr.iu, tr.ju), y, c, m, tr.H)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        v = val_mae(predict(model, va), va); sched.step(v)
        hist.append(dict(epoch=epoch, train_loss=tot / max(1, nb), val_MAE=v, sec=time.time() - t0, **hp))
        print(f"    {tag} ep{epoch:02d} loss {hist[-1]['train_loss']:.3f} val MAE {v:.3f} ({hist[-1]['sec']:.0f}s)", flush=True)
        if v < best - 1e-3:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    pd.DataFrame(hist).to_csv(f"{OUT}/logs/{tag}.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, len(hist)


def score(pred, bank, method, tag, seed, extra_cols=None, start=None):
    """pred (E, T, M) -> one row per episode, evaluated on the slots t >= start (default W-1)."""
    life, cens, up = bank.life.cpu().numpy(), bank.cens.cpu().numpy(), bank.up.cpu().numpy()
    sl = slice(W - 1 if start is None else start, None)
    rows = []
    for k, e in enumerate(bank.episodes):
        a = (pred[k, sl], life[k, sl], cens[k, sl], up[k, sl])
        r = bl.evaluate(*a); r.update(ex.evaluate_ext(*a, H))
        r.update(method=method, setting=tag, seed=seed, episode=e, W=W, **(extra_cols or {}))
        rows.append(r)
    return rows


# ------------------------------------------------------------------ analytical predictors and GBT (CPU part)
def kf_pred(bank, tau, q):
    out = []
    for e in bank.episodes:
        raw = bank.raw[e]
        kf = bl.kf_let(raw["pos_obs"], raw["vel_obs"], bank.iu.cpu().numpy(), bank.ju.cpu().numpy(),
                       sim.comm_range(raw["cfg"]), raw["cfg"], tau, q)
        miss = raw["miss"]
        out.append(np.maximum(dt.hold_prev(kf, miss) - dt.miss_age(miss) * raw["cfg"]["dt"], 0))
    return np.stack(out).astype(np.float32)


def tune_kf(va):
    best, rows = None, []
    life, cens, up = va.life.cpu().numpy(), va.cens.cpu().numpy(), va.up.cpu().numpy()
    m = up & ~cens
    for tau, q in KF_GRID:
        mae = float(np.abs(kf_pred(va, tau, q)[m] - life[m]).mean())
        rows.append(dict(tau=tau, q=q, val_MAE=mae))
        if best is None or mae < best[2]: best = (tau, q, mae)
    return best[0], best[1], pd.DataFrame(rows)


def snr_trend_pred(bank):
    out = []
    for k, e in enumerate(bank.episodes):
        raw = bank.raw[e]
        snr = dt.hold_prev(raw["snr_meas"], raw["miss"])
        out.append(ex.snr_trend(snr, bank.up[k].cpu().numpy(), raw["cfg"]["snr_th_db"], H, raw["cfg"]["dt"], W))
    return np.stack(out).astype(np.float32)


def gbt_xy(bank, k):
    nf = bank.node[k].cpu().numpy(); ef = bank.edge[k].float().cpu().numpy()
    return ex.gbt_features(nf, ef, bank.iu.cpu().numpy(), bank.ju.cpu().numpy(), W, bank.raw[bank.episodes[k]]["miss"])


def train_gbt(tr, seed, hp=None, rows=600_000):
    hp = {**GBT_DEFAULT, **(hp or {})}
    X, y = [], []
    for k in range(len(tr.episodes)):
        F = gbt_xy(tr, k)[W - 1:]; u = tr.up[k, W - 1:].cpu().numpy()
        X.append(F[u]); y.append(tr.life[k, W - 1:].cpu().numpy()[u])
    X, y = np.concatenate(X), np.concatenate(y)
    i = np.random.default_rng(seed).choice(len(y), min(rows, len(y)), replace=False)
    return HistGradientBoostingRegressor(loss="absolute_error", random_state=seed, **hp).fit(X[i], y[i])


def gbt_pred(g, bank):
    out = np.zeros((len(bank.episodes), bank.T, bank.M), np.float32)
    for k in range(len(bank.episodes)):
        F = gbt_xy(bank, k); u = bank.up[k].cpu().numpy().copy(); u[:W - 1] = False
        out[k][u] = np.clip(g.predict(F[u]), 0, H)
    return out


def let_cal(tr, te):
    sl = slice(W - 1, None)
    up, cens = tr.up[:, sl].cpu().numpy(), tr.cens[:, sl].cpu().numpy()
    m = up & ~cens
    b, a = np.polyfit(tr.let[:, sl][m], tr.life[:, sl].cpu().numpy()[m], 1)
    return np.clip(a + b * te.let, 0, H).astype(np.float32), dict(cal_a=a, cal_b=b)


def done(f):
    return os.path.exists(f)


def banks(setting, mob, need=("tr", "va", "te")):
    over = SETTINGS[setting]
    t0 = time.time()
    b = {k: EpisodeBank(mob, {"tr": TRAIN, "va": VAL, "te": TEST}[k], device=DEV, **over) for k in need}
    log(f"  episodes {setting}/{mob} ({'+'.join(need)}) generated in {time.time() - t0:.0f} s")
    return b


def analytical_and_gbt(tag, b):
    """CPU part of a setting: LET, KF-LET (tuned on validation), SNR trend, calibrated LET, GBT x 3 seeds."""
    tr, va, te = b["tr"], b["va"], b["te"]
    mob = tr.cfg["mobility"]
    f = f"{OUT}/runs/{tag}_{mob}_analytical.csv"
    if not done(f):
        tau, q, kt = tune_kf(va); kt.to_csv(f"{OUT}/runs/{tag}_{mob}_kf_tuning.csv", index=False)
        rows = score(te.let.astype(np.float32), te, "LET", tag, 0) + \
               score(kf_pred(te, tau, q), te, "KF-LET", tag, 0, dict(tau=tau, q=q)) + \
               score(snr_trend_pred(te), te, "SNR-trend", tag, 0)
        pc, cc = let_cal(tr, te); rows += score(pc, te, "LET-cal", tag, 0, cc)
        pd.DataFrame(rows).to_csv(f, index=False); log(f"  {tag}/{mob} analytical done")
    for seed in SEEDS:
        f = f"{OUT}/runs/{tag}_{mob}_GBT_s{seed}.csv"
        if done(f): continue
        g = train_gbt(tr, seed)
        pickle.dump(g, open(f"{OUT}/ckpt/{tag}_{mob}_GBT_s{seed}.pkl", "wb"))
        pd.DataFrame(score(gbt_pred(g, te), te, "GBT", tag, seed)).to_csv(f, index=False)
        log(f"  {tag}/{mob} GBT s{seed} done")


def neural_runs(tag, b, names, seeds, hp_of=None, extra=None, start=None):
    """Train each (model, seed) on b['tr'], select on b['va'], evaluate on b['te']."""
    tr, va, te = b["tr"], b["va"], b["te"]
    mob = tr.cfg["mobility"]
    for name, seed in itertools.product(names, seeds):
        f = f"{OUT}/runs/{tag}_{mob}_{name}_s{seed}.csv"
        if done(f): continue
        hp = (hp_of or {}).get(name)
        t0 = time.time()
        model, best, ep = train(name, seed, tr, va, f"{tag}_{mob}_{name}_s{seed}", hp)
        torch.save(model.state_dict(), f"{OUT}/ckpt/{tag}_{mob}_{name}_s{seed}.pt")
        cols = dict(train_min=(time.time() - t0) / 60, val_MAE_best=best, epochs=ep, **{**DEFAULT_HP[name], **(hp or {})},
                    **(extra or {}))
        pd.DataFrame(score(predict(model, te), te, name, tag, seed, cols, start)).to_csv(f, index=False)
        log(f"  {tag}/{mob} {name} s{seed} done in {(time.time() - t0) / 60:.1f} min (val MAE {best:.3f})")
        del model
        free_gpu()


# ------------------------------------------------------------------ modes
def mode_fading():
    """Realistic Hello reception: a Hello is decoded only if the Rician-faded SNR reaches the threshold."""
    for mob in MOBILITY:
        log(f"=== fading / {mob} ===")
        b = banks("fading", mob)
        if not NEURAL_ONLY: analytical_and_gbt("fading", b)
        if not NO_NEURAL:   neural_runs("fading", b, NEURAL_TUNED, SEEDS)
        del b


def mode_window():
    """LSTM with W = 4, 8, 16 (main configuration, seed 0); all W evaluated on the same slots t >= 15."""
    global W
    for mob in MOBILITY:
        log(f"=== window / {mob} ===")
        b = banks("main", mob)
        for w in (4, 8, 16):
            W = w
            tag = f"window-W{w}"
            if not NO_NEURAL:
                neural_runs(tag, b, ["LSTM"], [0], start=15)
            f = f"{OUT}/runs/{tag}_{mob}_GBT_s0.csv"
            if not NEURAL_ONLY and not done(f):
                g = train_gbt(b["tr"], 0)
                pd.DataFrame(score(gbt_pred(g, b["te"]), b["te"], "GBT", tag, 0, start=15)).to_csv(f, index=False)
                log(f"  {tag}/{mob} GBT done")
        W = W_DEFAULT
        del b


def mode_tune():
    """Random search with the same budget (TUNE_N configurations, seed 0) for every model, scored on the
    validation episodes only. The test episodes are not used here."""
    names = ([] if NO_NEURAL else NEURAL_TUNED) + ([] if NEURAL_ONLY else ["GBT"])
    total = len(names) * TUNE_N * 2 * len(MOBILITY)
    n = 0
    for setting, mob in itertools.product(["main", "spatial"], MOBILITY):
        todo = [(nm, ci) for nm in names for ci in range(TUNE_N)
                if not done(f"{OUT}/runs/tune_{setting}_{mob}_{nm}_c{ci}.csv")]
        n += len(names) * TUNE_N - len(todo)
        if not todo: continue
        log(f"=== tune / {setting} / {mob}: {len(todo)} runs left ===")
        b = banks(setting, mob, ("tr", "va"))
        for nm, ci in todo:
            hp = configs(nm)[ci]
            t0 = time.time()
            if nm == "GBT":
                g = train_gbt(b["tr"], 0, hp)
                pv = gbt_pred(g, b["va"]); best, ep = val_mae(pv, b["va"]), hp["max_iter"]
            else:
                model, best, ep = train(nm, 0, b["tr"], b["va"], f"tune_{setting}_{mob}_{nm}_c{ci}", hp)
                pv = predict(model, b["va"]); del model
                free_gpu()
            r = score(pv, b["va"], nm, f"tune-{setting}", 0)
            vm = pd.DataFrame(r)[["MAE_all", "AUPRC_5", "Cindex"]].mean()
            pd.DataFrame([dict(setting=setting, mobility=mob, method=nm, config=ci, val_MAE=best,
                               val_MAE_all=vm.MAE_all, val_AUPRC_5=vm.AUPRC_5, val_Cindex=vm.Cindex,
                               epochs=ep, minutes=(time.time() - t0) / 60, **hp)]).to_csv(
                f"{OUT}/runs/tune_{setting}_{mob}_{nm}_c{ci}.csv", index=False)
            n += 1
            log(f"  [{n}/{total}] tune {setting}/{mob} {nm} c{ci} val MAE {best:.3f} ({(time.time() - t0) / 60:.1f} min)")
        del b


def best_configs():
    fs = glob.glob(f"{OUT}/runs/tune_*_c*.csv")
    if not fs:
        sys.exit("No tuning results found: run `python run_v6.py tune` first.")
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    best = t.loc[t.groupby(["setting", "mobility", "method"]).val_MAE.idxmin()]
    best.to_csv(f"{OUT}/best_configs.csv", index=False)
    return best


def mode_final():
    """Best configuration of each model (by validation MAE) retrained with 3 seeds and tested."""
    best = best_configs()
    for setting, mob in itertools.product(["main", "spatial"], MOBILITY):
        sel = best[(best.setting == setting) & (best.mobility == mob)]
        if sel.empty: continue
        log(f"=== final / {setting} / {mob} ===")
        b = banks(setting, mob)
        tag = f"final-{setting}"
        for r in sel.itertuples():
            if r.method == "GBT":
                if NEURAL_ONLY: continue
                hp = {k: getattr(r, k) for k in GBT_DEFAULT}
                hp = {k: (float(v) if k == "learning_rate" else int(v)) for k, v in hp.items()}
                for seed in SEEDS:
                    f = f"{OUT}/runs/{tag}_{mob}_GBT_s{seed}.csv"
                    if done(f): continue
                    g = train_gbt(b["tr"], seed, hp)
                    pd.DataFrame(score(gbt_pred(g, b["te"]), b["te"], "GBT", tag, seed, dict(config=r.config, **hp))).to_csv(f, index=False)
                    log(f"  {tag}/{mob} GBT s{seed} done")
            elif not NO_NEURAL:
                hp = dict(d=int(r.d), layers=int(r.layers), lr=float(r.lr), wd=float(r.wd))
                neural_runs(tag, b, [r.method], SEEDS, {r.method: hp}, dict(config=int(r.config)))
        del b


def mode_reeval():
    """Published checkpoints (v2.0 main, v2.1 spatial) evaluated, without retraining, on the 30 test episodes.
    Episodes 17-22 are the six test episodes of the paper, so their rows must reproduce the published values."""
    pub = f"{OUT}/published"
    if not os.path.isdir(pub):
        for z in ("checkpoints_v4.zip", "checkpoints_logs_v5_spatial.zip"):
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if m.endswith(".pt") and (m.split("/")[-1].startswith("main_") or m.split("/")[-1].startswith("spatial_")):
                        os.makedirs(pub, exist_ok=True)
                        open(f"{pub}/{m.split('/')[-1]}", "wb").write(zf.read(m))
    jobs = {"main": ["MLP", "LSTM", "StaticGNN", "STGNN", "StaticGNN-noagg"],
            "spatial": ["LSTM", "StaticGNN", "StaticGNN-noagg"]}
    for setting, mob in itertools.product(jobs, MOBILITY):
        todo = [(nm, s) for nm in jobs[setting] for s in SEEDS
                if not done(f"{OUT}/runs/reeval-{setting}_{mob}_{nm}_s{s}.csv")]
        if not todo: continue
        log(f"=== reeval / {setting} / {mob}: {len(todo)} checkpoints ===")
        te = banks(setting, mob, ("te",))["te"]
        for nm, s in todo:
            m = build(nm, te).to(DEV)
            sd = torch.load(f"{pub}/{setting}_{mob}_{nm}_s{s}.pt", map_location=DEV)
            m.load_state_dict({k: v for k, v in sd.items() if k not in ("iu", "ju")}, strict=False)
            pd.DataFrame(score(predict(m, te), te, nm, f"reeval-{setting}", s)).to_csv(
                f"{OUT}/runs/reeval-{setting}_{mob}_{nm}_s{s}.csv", index=False)
            log(f"  reeval {setting}/{mob} {nm} s{s} done")
        del te
        free_gpu()


def mode_ref():
    """CPU reference on the 30 test episodes: analytical predictors and the fixed-setting GBT (3 seeds)."""
    for setting, mob in itertools.product(["main", "spatial"], MOBILITY):
        if all(done(f"{OUT}/runs/ref-{setting}_{mob}_{x}.csv") for x in ["analytical"] + [f"GBT_s{s}" for s in SEEDS]):
            continue
        log(f"=== ref / {setting} / {mob} ===")
        b = banks(setting, mob)
        analytical_and_gbt(f"ref-{setting}", b)
        del b


def letsnr_bank(bank, decay=False):
    out = []
    iu, ju = bank.iu.cpu().numpy(), bank.ju.cpu().numpy()
    for k, e in enumerate(bank.episodes):
        raw = bank.raw[e]
        p, _ = x6.let_snr(raw["pos_obs"], raw["vel_obs"], raw["snr_meas"], bank.hel[k].cpu().numpy(),
                          bank.up[k].cpu().numpy(), raw["miss"], iu, ju, raw["cfg"], W, H, decay)
        out.append(p)
    return np.stack(out)


def mode_letsnr():
    """SNR-corrected LET (no training) and its linear calibration fitted on the training episodes."""
    for setting, mob in itertools.product(["main", "spatial", "fading"], MOBILITY):
        f = f"{OUT}/runs/letsnr-{setting}_{mob}.csv"
        if done(f): continue
        log(f"=== letsnr / {setting} / {mob} ===")
        b = banks(setting, mob, ("tr", "te"))
        rows = []
        for decay, name in [(False, "LET-SNR"), (True, "LET-SNR-decay")]:
            ptr, pte = letsnr_bank(b["tr"], decay), letsnr_bank(b["te"], decay)
            sl = slice(W - 1, None)
            m = b["tr"].up[:, sl].cpu().numpy() & ~b["tr"].cens[:, sl].cpu().numpy()
            bb, aa = np.polyfit(ptr[:, sl][m], b["tr"].life[:, sl].cpu().numpy()[m], 1)
            pcal = np.clip(aa + bb * pte, 0, H).astype(np.float32)
            pcal[~b["te"].up.cpu().numpy()] = 0
            rows += score(pte, b["te"], name, setting, 0) + \
                    score(pcal, b["te"], name + "-cal", setting, 0, dict(cal_a=aa, cal_b=bb))
        pd.DataFrame(rows).to_csv(f, index=False)
        log(f"  letsnr {setting}/{mob} done")
        del b


def gbt_full_xy(bank, k):
    """Base GBT features (extra.gbt_features) + neighbourhood summaries, (T, M, 41 + 10)."""
    base = gbt_xy(bank, k)
    nb = x6.nbr_features(bank.edge[k].float().cpu().numpy(), bank.up[k].cpu().numpy(),
                         bank.iu.cpu().numpy(), bank.ju.cpu().numpy(), bank.N, bank.cfg["snr_th_db"])
    return np.concatenate([base, nb], -1), base.shape[-1], nb.shape[-1]


def mode_ablation():
    """GBT trained on feature subsets (seed 0) and with neighbourhood summaries (3 seeds); same subsample of
    training rows and same hyperparameters as the reference GBT."""
    for setting, mob in itertools.product(["main", "spatial", "fading"], MOBILITY):
        jobs = [(v, 0) for v in x6.VARIANTS] + [("all+nbr", s) for s in SEEDS if s != 0]
        jobs = [(v, s) for v, s in jobs if not done(f"{OUT}/runs/abl-{setting}_{mob}_{v}_s{s}.csv")]
        if not jobs: continue
        log(f"=== ablation / {setting} / {mob}: {len(jobs)} models ===")
        b = banks(setting, mob, ("tr", "te"))
        tr, te = b["tr"], b["te"]
        X, y = [], []
        for k in range(len(tr.episodes)):
            F, nb_, nn_ = gbt_full_xy(tr, k)
            F = F[W - 1:]; u = tr.up[k, W - 1:].cpu().numpy()
            X.append(F[u]); y.append(tr.life[k, W - 1:].cpu().numpy()[u])
        X, y = np.concatenate(X), np.concatenate(y)
        models = {}
        for v, s in jobs:
            cols = x6.columns(v, nb_, nn_)
            i = np.random.default_rng(s).choice(len(y), min(600_000, len(y)), replace=False)
            t0 = time.time()
            models[(v, s)] = (cols, HistGradientBoostingRegressor(loss="absolute_error", random_state=s,
                                                                  **GBT_DEFAULT).fit(X[i][:, cols], y[i]))
            log(f"  fit {v} s{s} ({len(cols)} features, {time.time() - t0:.0f} s)")
        del X, y
        preds = {key: np.zeros((len(te.episodes), te.T, te.M), np.float32) for key in models}
        for k in range(len(te.episodes)):
            F, _, _ = gbt_full_xy(te, k)
            u = te.up[k].cpu().numpy().copy(); u[:W - 1] = False
            Fu = F[u]
            for key, (cols, g) in models.items():
                preds[key][k][u] = np.clip(g.predict(Fu[:, cols]), 0, H)
        for (v, s), p in preds.items():
            pd.DataFrame(score(p, te, "GBT-" + v, f"abl-{setting}", s, dict(variant=v, n_features=len(models[(v, s)][0])))
                         ).to_csv(f"{OUT}/runs/abl-{setting}_{mob}_{v}_s{s}.csv", index=False)
        log(f"  ablation {setting}/{mob} done")
        del b, preds, models


def mode_check():
    log(f"python {platform.python_version()} | torch {torch.__version__} | numpy {np.__version__}")
    log(f"device: {DEV}" +
        (f" | GPU: {torch.cuda.get_device_name(0)} | memory {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB"
         if DEV == "cuda" else (" (Apple Silicon)" if DEV == "mps" else " -> no GPU: training would be very slow")))
    sim.self_test()
    t0 = time.time()
    b = banks("main", "gm")
    model, best, ep = train("StaticGNN", 0, b["tr"], b["va"], "check_StaticGNN")
    log(f"smoke test: StaticGNN 1 epoch on 2 episodes, val MAE {best:.2f}, {time.time() - t0:.0f} s")
    t0 = time.time()
    model, best, ep = train("LSTM", 0, b["tr"], b["va"], "check_LSTM")
    log(f"smoke test: LSTM 1 epoch on 2 episodes, val MAE {best:.2f}, {time.time() - t0:.0f} s")
    log("CHECK PASSED")


def mode_pack():
    name = f"results_v6_{platform.node() or 'pc'}.zip".replace(" ", "_")
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as z:
        for root in ("runs", "logs", "ckpt"):
            for f in glob.glob(f"{OUT}/{root}/*"):
                z.write(f)
        for f in [f"{OUT}/progress.log", f"{OUT}/best_configs.csv"]:
            if os.path.exists(f): z.write(f)
    log(f"packed {name} ({os.path.getsize(name) / 2**20:.1f} MB) -> send this file back")


if __name__ == "__main__":
    log(f"run_v6 {MODE} | device {DEV} | out {OUT} | "
        f"{'neural only' if NEURAL_ONLY else ('CPU part only' if NO_NEURAL else 'all predictors')}"
        f"{' | QUICK' if QUICK else ''}")
    if not NO_NEURAL and DEV == "cpu":
        log("WARNING: no GPU found; neural training on the CPU will take many hours")
    modes = {"check": mode_check, "reeval": mode_reeval, "fading": mode_fading, "window": mode_window,
             "tune": mode_tune, "final": mode_final, "pack": mode_pack,
             "ref": mode_ref, "letsnr": mode_letsnr, "ablation": mode_ablation}
    if MODE == "all":
        for m in ("reeval", "fading", "window", "tune", "final", "pack"):
            modes[m]()
    elif MODE in modes:
        modes[MODE]()
    else:
        sys.exit(f"unknown mode {MODE}; use one of: {', '.join(modes)} or all")
    log(f"run_v6 {MODE} finished")
