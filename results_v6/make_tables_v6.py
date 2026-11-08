"""Numbers of the third revision computed from results_v6_all_runs.csv (run from the repository root):
Table 7 (paired differences, 30 test episodes), Table 9 (sources of the learned advantage), the fading rows of
Table 10, the window study and the hyperparameter search. Learned predictors are averaged over seeds per episode."""
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("results_v6/results_v6_all_runs.csv")
d["setting"] = d.setting.str.replace("^ref-|^abl-", "", regex=True)
d["nAUPRC"] = (d.AUPRC_5 - d.pos_5) / (1 - d.pos_5)
MOB = ["rpgm", "gm", "rwp"]
COLS = ["MAE", "MAE_all", "AUPRC_5", "nAUPRC", "Cindex", "pos_5"]


def ep(setting, method):
    x = d[(d.setting == setting) & (d.method == method)]
    return x.groupby(["mobility", "episode"])[COLS].mean()


def ci(a, b, col, mob):
    z = (a.loc[mob][col] - b.loc[mob][col]).dropna()
    m, h = z.mean(), stats.t.ppf(.975, len(z) - 1) * z.std(ddof=1) / np.sqrt(len(z))
    return f"{m:+.3f} [{m - h:+.3f}, {m + h:+.3f}] ({int((z > 0).sum())}/{len(z)} > 0)"


print("Table 7: paired differences over 30 test episodes")
for st in ["main", "spatial"]:
    L, LS = ep(st, "LET"), ep(f"reeval-{st}", "LSTM")
    S, N = ep(f"reeval-{st}", "StaticGNN"), ep(f"reeval-{st}", "StaticGNN-noagg")
    for mob in MOB:
        print(f"  {st:8s}{mob:5s} LSTM-LET AUPRC {ci(LS, L, 'AUPRC_5', mob)} MAE {ci(LS, L, 'MAE', mob)} "
              f"MAE_all {ci(LS, L, 'MAE_all', mob)} | agg-noagg MAE {ci(S, N, 'MAE', mob)} AUPRC {ci(S, N, 'AUPRC_5', mob)}")

print("\nTable 9: AUPRC for breaks within 5 s, 30 test episodes")
rows = [("LET", "LET"), ("KF-LET", "KF-LET"), ("SNR-corrected LET", "LET-SNR-decay"), ("GBT, K", "GBT-K"),
        ("GBT, K+M", "GBT-K+M"), ("GBT, K+S", "GBT-K+S"), ("GBT, S", "GBT-S"), ("GBT, all features", "GBT"),
        ("GBT, no absolute positions", "GBT-noAbsPos"), ("GBT, + neighbourhood summaries", "GBT-all+nbr"),
        ("LSTM", "LSTM"), ("StaticGNN w/o agg.", "StaticGNN-noagg"), ("StaticGNN", "StaticGNN")]
neural = {"main": "reeval-main", "spatial": "reeval-spatial", "fading": "fading"}
print("  " + " " * 32 + "  ".join(f"{s}/{m}" for s in neural for m in MOB))
for lab, m in rows:
    vals = []
    for st in neural:
        x = ep(neural[st] if m in ("LSTM", "StaticGNN", "StaticGNN-noagg") else st, m).groupby(level=0).mean()
        vals += [x.loc[mob].AUPRC_5 for mob in MOB]
    print(f"  {lab:32s}" + "  ".join(f"{v:.3f}" for v in vals))

print("\nTable 10, fading Hello reception (LET vs learned range over GBT, LSTM, StaticGNN)")
for mob in MOB:
    L = ep("fading", "LET").loc[mob].mean()
    Ls = [ep("fading", m).loc[mob].mean() for m in ["GBT", "LSTM", "StaticGNN"]]
    rng = lambda c: f"{min(x[c] for x in Ls):.3f}-{max(x[c] for x in Ls):.3f}"
    print(f"  {mob}: Pos {L.pos_5:.2f} | MAE {L.MAE:.2f} / {rng('MAE')} | MAE_all {L.MAE_all:.2f} / {rng('MAE_all')} | "
          f"AUPRC {L.AUPRC_5:.3f} / {rng('AUPRC_5')} | nAUPRC {L.nAUPRC:.2f} / {rng('nAUPRC')}")

print("\nWindow length (seed 0, slots t >= 15)")
for m in ["LSTM", "GBT"]:
    for mob in MOB:
        r = [d[(d.setting == f"window-W{w}") & (d.method == m) & (d.mobility == mob)][["MAE", "MAE_all", "AUPRC_5"]].mean()
             for w in (4, 8, 16)]
        print(f"  {m:5s}{mob:5s}" + " | ".join(f"W={w}: MAE {x.MAE:.2f} MAE_all {x.MAE_all:.2f} AUPRC {x.AUPRC_5:.4f}"
                                            for w, x in zip((4, 8, 16), r)))

print("\nHyperparameter search: tuned minus fixed setting, 30 test episodes")
for st in ["main", "spatial"]:
    for m, fixed in [("GBT", st), ("LSTM", f"reeval-{st}"), ("StaticGNN", f"reeval-{st}"), ("StaticGNN-noagg", f"reeval-{st}")]:
        tu, fx = ep(f"final-{st}", m), ep(fixed, m)
        for mob in MOB:
            z = (tu.loc[mob] - fx.loc[mob]).mean()
            print(f"  {st:8s}{m:16s}{mob:5s} AUPRC {z.AUPRC_5:+.4f} MAE {z.MAE:+.3f} MAE_all {z.MAE_all:+.3f}")
    S, N = ep(f"final-{st}", "StaticGNN"), ep(f"final-{st}", "StaticGNN-noagg")
    for mob in MOB:
        print(f"  tuned {st} {mob}: agg - no agg AUPRC {ci(S, N, 'AUPRC_5', mob)}")
