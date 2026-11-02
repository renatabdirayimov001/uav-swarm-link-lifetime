"""Figures 3-5 of the revised paper from the v4 results."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import aggregate as a
from uav_swarm_sim import generate_episode
from data_torch import held_let
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12, "xtick.labelsize": 11.5,
                     "ytick.labelsize": 11.5, "legend.fontsize": 11.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
INK, MUTED = "#2b2b29", "#6b6a64"
CAT = ["#7a7a72", "#eb6834", "#1baf7a", "#eda100", "#2a78d6", "#8fbbea", "#6250d6"]
OUT = "figs/"; import os; os.makedirs(OUT, exist_ok=True)
a.DIRS = ["out_v4", "colab_out", "colab_v5"]; df = a.load(); ep = a.per_episode(df)
names = {"rpgm": "RPGM", "gm": "Gauss–Markov", "rwp": "RWP"}

# Fig 3: LET vs true lifetime (episode 18 = seed 10017), LET as known to the receiver
W = 8
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, mob in zip(axs, ["rpgm", "gm"]):
    e = generate_episode(mobility=mob, channel="a2a", seed=10017)
    let = held_let(e["let"], e)
    m = e["up"] & ~e["censored"]; m[:W - 1] = False
    t, l = e["life"][m], let[m]
    hb = ax.hexbin(t, l, gridsize=30, cmap="Blues", mincnt=1, bins="log", extent=(0, 60, 0, 60), linewidths=0.2)
    ax.plot([0, 60], [0, 60], color="#e34948", lw=1.5, ls="--")
    ax.text(41, 56, "perfect prediction", color="#e34948", fontsize=11, ha="center")
    ax.set_title(f"{names[mob]}: MAE = {np.abs(l - t).mean():.1f} s (episode 18)", color=INK)
    ax.set_xlabel("true residual lifetime [s]"); ax.set_xlim(0, 60); ax.set_ylim(0, 60)
axs[0].set_ylabel("LET prediction [s]")
cb = fig.colorbar(hb, ax=axs, pad=0.02); cb.set_label("samples (log scale)")
fig.savefig(OUT + "fig_let_baseline_error.pdf"); plt.close(fig)

# Fig 4: channel effect (disk vs A2A), mean +- std over test episodes
fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.0), sharey="row")
for c, mob in enumerate(["rpgm", "gm", "rwp"]):
    for row, (k, lab) in enumerate([("MAE", "MAE [s]"), ("AUPRC", "AUPRC (break within 5 s)")]):
        ax = axs[row, c]; w = 0.36
        for j, meth in enumerate(["LET", "KF-LET"]):
            g = [ep[(ep.setting == st) & (ep.mobility == mob) & (ep.method == meth)][k] for st in ["disk", "main"]]
            ax.bar(np.arange(2) + (j - .5) * w, [x.mean() for x in g], w * .9, yerr=[x.std() for x in g],
                   color=CAT[j], label=meth, capsize=3, error_kw=dict(lw=1, ecolor=MUTED), edgecolor="white", linewidth=1)
        ax.set_xticks([0, 1], ["disk model", "A2A channel"])
        if c == 0: ax.set_ylabel(lab)
        if row == 0: ax.set_title(names[mob], color=INK)
axs[1, 0].set_ylim(0, 1)
h, l = axs[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.savefig(OUT + "fig_channel_effect.pdf"); plt.close(fig)

# Fig 5: relative change vs LET, mean over test episodes, error bars = 95% CI of the paired difference
meths = ["KF-LET", "GBT", "MLP", "LSTM", "StaticGNN", "StaticGNN-noagg", "ST-GNN"]
LAB = {"StaticGNN-noagg": "StaticGNN w/o agg."}
from scipy import stats
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
mobs = ["rpgm", "gm", "rwp"]; y = np.arange(3); h = 0.115
for k, (ax, met, title) in enumerate(zip(axs, ["MAE", "AUPRC"], ["MAE reduction vs. LET [%]", "AUPRC gain vs. LET [%]"])):
    for j, m in enumerate(meths):
        vals, errs = [], []
        for mob in mobs:
            g = ep[(ep.setting == "main") & (ep.mobility == mob)]
            x = g[g.method == m].set_index("episode")[met]; b = g[g.method == "LET"].set_index("episode")[met]
            r = (b - x) / b * 100 if met == "MAE" else (x - b) / b * 100
            vals.append(r.mean()); errs.append(stats.t.ppf(0.975, len(r) - 1) * r.std() / np.sqrt(len(r)))
        pos = y + (j - 3) * h
        ax.barh(pos, vals, h * 0.9, xerr=errs, color=CAT[j], label=LAB.get(m, m), edgecolor="white", linewidth=0.8,
                error_kw=dict(lw=0.8, ecolor=INK, capsize=1.5))
        for p, v, e in zip(pos, vals, errs):
            ax.text(v + (e + 0.6 if v >= 0 else -e - 0.6), p, ("0" if abs(v) < 0.5 else f"{v:+.0f}"), va="center",
                    ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=1); ax.set_title(title + "  (higher is better)", color=INK); ax.set_xlim(-22, 40)
axs[0].set_yticks(y, [names[m] for m in mobs]); axs[0].invert_yaxis()
h_, l_ = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.87))
fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.savefig(OUT + "fig_relative_gain.pdf"); plt.close(fig)
print("ok")
