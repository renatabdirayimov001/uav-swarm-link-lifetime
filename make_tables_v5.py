"""LaTeX table bodies for the v5 revision (written to tables_v5.json)."""
import json, numpy as np, pandas as pd
import aggregate as a
a.DIRS = ["./out_v4", "./colab_out", "./colab_v5"]
df = a.load(); s, seed = a.summary(df)
M = s.xs("mean", axis=1, level=1); S = s.xs("std", axis=1, level=1)
P = pd.concat([pd.read_csv("paired.csv"), pd.read_csv("v5_analysis/paired_v5.csv")])
P = P.drop_duplicates(subset=["setting", "mobility", "a", "b", "metric"], keep="last")
NM = {"rpgm": "RPGM", "gm": "GM", "rwp": "RWP"}
DN = {"LET": "LET", "KF-LET": "KF-LET", "LET-cal": "Calibrated LET", "SNR-trend": "SNR trend", "GBT": "GBT",
      "MLP": "MLP", "LSTM": "LSTM", "StaticGNN": "StaticGNN", "StaticGNN-noagg": "StaticGNN w/o agg.",
      "ST-GNN": "ST-GNN", "Oracle": "Oracle (true lifetime)", "MinHop": "Min-hop (no lifetime)"}
MOBS = ["rpgm", "gm", "rwp"]


def f(x, p=2):
    t = f"{x:.{p}f}"
    return t.replace("-", "$-$") if x < 0 else t


def nap(auprc, pos):
    return (auprc - pos) / (1 - pos)


out = {}

# ---- Table 4: analytical predictors, disk vs A2A (+ calibrated LET on A2A)
L = []
for mob in MOBS:
    first = True
    for st, lab, meths in [("disk", "Disk", ["LET", "KF-LET"]), ("main", "A2A", ["LET", "KF-LET", "LET-cal"])]:
        for j, m in enumerate(meths):
            r = M.loc[(st, mob, m)]; sd = S.loc[(st, mob, m)]
            pos = f"{r.pos_rate:.3f}" if j == 0 else ""
            L.append(f"{NM[mob] if first else ''} & {lab if j == 0 else ''} & {DN[m]} & {r.MAE:.2f} $\\pm$ {sd.MAE:.2f}"
                     f" & {r.RMSE:.2f} & {f(r.R2, 3)} & {r.AUC:.3f} & {r.AUPRC:.3f} $\\pm$ {sd.AUPRC:.3f} & {pos}\\\\")
            first = False
    L.append("\\midrule")
out["analytical"] = "\n".join(L[:-1])

# ---- Table 5: all predictors, main A2A, with Pos. and normalised AUPRC
order = ["LET", "KF-LET", "LET-cal", "SNR-trend", "GBT", "MLP", "LSTM", "StaticGNN", "StaticGNN-noagg", "ST-GNN"]
L = []
for mob in MOBS:
    g = M.loc["main", mob].loc[order]
    best = {"MAE": g.MAE.idxmin(), "R2": g.R2.idxmax(), "AUC": g.AUC.idxmax(), "AUPRC": g.AUPRC.idxmax()}
    B = lambda m, c, t: f"\\textbf{{{t}}}" if best[c] == m else t
    for j, m in enumerate(order):
        r = g.loc[m]; sd = S.loc[("main", mob, m)]; ss = seed.loc[("main", mob, m)]
        sseed = "--" if np.isnan(ss.MAE) else f"{ss.MAE:.2f}"
        pos = f"{r.pos_rate:.3f}" if j == 0 else ""
        L.append(f"{NM[mob] if j == 0 else ''} & {pos} & {DN[m]} & {B(m, 'MAE', f'{r.MAE:.2f} $\\pm$ {sd.MAE:.2f}')} & {sseed}"
                 f" & {B(m, 'R2', f(r.R2, 3))} & {B(m, 'AUC', f'{r.AUC:.3f}')}"
                 f" & {B(m, 'AUPRC', f'{r.AUPRC:.3f} $\\pm$ {sd.AUPRC:.3f}')} & {nap(r.AUPRC, r.pos_rate):.3f}\\\\")
    L.append("\\midrule")
out["learned"] = "\n".join(L[:-1])

# ---- Table 6: censoring-aware metrics
L = []
for mob in MOBS:
    g = M.loc["main", mob].loc[order]
    best = {"MAE_all": g.MAE_all.idxmin(), "Cindex": g.Cindex.idxmax(), "AUPRC_2": g.AUPRC_2.idxmax(),
            "AUPRC_10": g.AUPRC_10.idxmax()}
    B = lambda m, c, t: f"\\textbf{{{t}}}" if best[c] == m else t
    for j, m in enumerate(order):
        r = g.loc[m]
        L.append(f"{NM[mob] if j == 0 else ''} & {DN[m]} & {B(m, 'MAE_all', f'{r.MAE_all:.2f}')} & {B(m, 'Cindex', f'{r.Cindex:.3f}')}"
                 f" & {B(m, 'AUPRC_2', f'{r.AUPRC_2:.3f}')} & {B(m, 'AUPRC_10', f'{r.AUPRC_10:.3f}')}\\\\")
    L.append("\\midrule")
out["censored"] = "\n".join(L[:-1])


# ---- Table 7: paired differences (main and spatial)
def pr(st, mob, x, y, met, p=2):
    r = P[(P.setting == st) & (P.mobility == mob) & (P.a == x) & (P.b == y) & (P.metric == met)].iloc[0]
    return f"{f(r['diff'], p)} [{f(r.lo, p)}, {f(r.hi, p)}]"


L = []
for st, lab in [("main", "Main"), ("spatial", "Spatial")]:
    for j, mob in enumerate(MOBS):
        L.append(f"{lab if j == 0 else ''} & {NM[mob]} & {pr(st, mob, 'LSTM', 'LET', 'AUPRC', 3)} & {pr(st, mob, 'LSTM', 'LET', 'MAE')}"
                 f" & {pr(st, mob, 'LSTM', 'LET', 'MAE_all')} & {pr(st, mob, 'StaticGNN', 'StaticGNN-noagg', 'MAE')}"
                 f" & {pr(st, mob, 'StaticGNN', 'StaticGNN-noagg', 'AUPRC', 3)}\\\\")
    L.append("\\midrule")
out["paired"] = "\n".join(L[:-1])

# ---- Table 8: robustness (other shadowing models, generalisation, retraining)
L = []
SN = {"motion": "Motion-driven shadowing$^{a}$", "spatial": "Spatial shadowing field$^{a}$",
      "N20": "$N=20$$^{b}$", "N50": "$N=50$$^{b}$", "std4": "$\\sigma_X=4$~dB$^{b}$",
      "N50r": "$N=50$, retrained$^{c}$", "std4r": "$\\sigma_X=4$~dB, retrained$^{c}$"}
LRN = {"motion": ["GBT", "LSTM", "StaticGNN"], "spatial": ["GBT", "LSTM", "StaticGNN", "StaticGNN-noagg"],
       "N20": ["GBT", "LSTM", "StaticGNN"], "N50": ["GBT", "LSTM", "StaticGNN"], "std4": ["GBT", "LSTM", "StaticGNN"],
       "N50r": ["GBT"], "std4r": ["GBT"]}
for st in ["motion", "spatial", "N20", "N50", "std4", "N50r", "std4r"]:
    mobs = ["rpgm"] if st.endswith("r") else MOBS
    for j, mob in enumerate(mobs):
        g = M.loc[st, mob]; lr = g.loc[LRN[st]]; le = g.loc["LET"]; pos = le.pos_rate

        def rg(c, p=2, fn=lambda x: x):
            v = fn(lr[c]); return f"{v.min():.{p}f}" if len(v) == 1 else f"{v.min():.{p}f}--{v.max():.{p}f}"
        L.append(f"{SN[st] if j == 0 else ''} & {NM[mob]} & {pos:.2f} & {le.MAE:.2f} & {rg('MAE')} & {le.MAE_all:.2f} & {rg('MAE_all')}"
                 f" & {le.AUPRC:.3f} & {rg('AUPRC', 3)} & {nap(le.AUPRC, pos):.2f} & {rg('AUPRC', 2, lambda x: nap(x, pos))}\\\\")
    L.append("\\midrule")
out["robust"] = "\n".join(L[:-1])

# ---- Table 9: spatial shadowing, all predictors
L = []
sp_order = ["LET", "KF-LET", "LET-cal", "GBT", "LSTM", "StaticGNN", "StaticGNN-noagg"]
for mob in MOBS:
    g = M.loc["spatial", mob].loc[sp_order]
    best = {"MAE": g.MAE.idxmin(), "MAE_all": g.MAE_all.idxmin(), "Cindex": g.Cindex.idxmax(), "AUPRC": g.AUPRC.idxmax()}
    B = lambda m, c, t: f"\\textbf{{{t}}}" if best[c] == m else t
    for j, m in enumerate(sp_order):
        r = g.loc[m]; sd = S.loc[("spatial", mob, m)]
        L.append(f"{NM[mob] if j == 0 else ''} & {f'{r.pos_rate:.3f}' if j == 0 else ''} & {DN[m]} & {B(m, 'MAE', f'{r.MAE:.2f}')}"
                 f" & {f(r.R2, 3)} & {B(m, 'MAE_all', f'{r.MAE_all:.2f}')} & {B(m, 'Cindex', f'{r.Cindex:.3f}')}"
                 f" & {B(m, 'AUPRC', f'{r.AUPRC:.3f} $\\pm$ {sd.AUPRC:.3f}')} & {nap(r.AUPRC, r.pos_rate):.3f}\\\\")
    L.append("\\midrule")
out["spatial"] = "\n".join(L[:-1])

# ---- Table 10: sensitivity (seed-0 GBT, LET, calibrated LET)
d0 = df[df.seed.fillna(0) == 0]
ep0 = d0.groupby(["setting", "mobility", "method", "episode"])[a.METRICS].mean().reset_index()
M0 = ep0.groupby(["setting", "mobility", "method"])[a.METRICS].mean()
CASES = [("main", "Reference"),
         ("sens-W4", "$W=4$"), ("sens-W16", "$W=16$"), ("sens-dt025", "$\\Delta t=0.25$~s"),
         ("sens-dt1", "$\\Delta t=1$~s"), ("sens-k1", "$k=1$"), ("sens-k3", "$k=3$"),
         ("sens-v6", "$\\bar v=6$~m/s"), ("sens-v20", "$\\bar v=20$~m/s")]
L = []
for st, lab in CASES:
    cells = []
    for mob in MOBS:
        g = M0.loc[(st, mob)]
        le, gb = g.loc['LET'], g.loc['GBT']
        cells += [f"{le.MAE:.1f} / {gb.MAE:.1f}", f"{le.MAE_all:.1f} / {gb.MAE_all:.1f}", f"{le.AUPRC:.2f} / {gb.AUPRC:.2f}"]
    L.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    if st == "main": L.append("\\midrule")
out["sens"] = "\n".join(L)

# ---- Table 11: event-level and route-level metrics
ev = pd.read_csv("v5_analysis/event_metrics.csv").set_index(["mobility", "method"])
rt = pd.read_csv("v5_analysis/route_metrics.csv").set_index(["mobility", "method"])
L = []
for m in ["LET", "KF-LET", "LET-cal", "GBT", "MLP", "LSTM", "StaticGNN", "StaticGNN-noagg", "ST-GNN", "Oracle", "MinHop"]:
    cells = []
    for mob in MOBS:
        if (mob, m) in ev.index and m != "Oracle":
            e = ev.loc[(mob, m)]; cells += [f"{e.MAE_interval:.2f}", f"{e.recall_lead5:.2f}"]
        else:
            cells += ["--", "--"]
        r = rt.loc[(mob, m)]; cells += [f"{r.route_breaks_per_min:.2f}", f"{r.mean_route_life:.1f}"]
    L.append(f"{DN[m]} & " + " & ".join(cells) + "\\\\")
    if m in ("LET-cal", "ST-GNN"): L.append("\\midrule")
out["event_route"] = "\n".join(L)

json.dump(out, open("tables_v5.json", "w"), indent=1)
for k, v in out.items():
    print("=== ", k); print(v)
