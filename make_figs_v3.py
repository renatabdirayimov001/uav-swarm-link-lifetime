"""Figures 1-3 of the paper from simulator v3 and Experiment 1 (v3) results."""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uav_swarm_sim import generate_episode, comm_range
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
INK, MUTED = "#2b2b29", "#6b6a64"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "figs_v3/"
E1 = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/exp1_run/results_exp1.csv"
import os; os.makedirs(OUT, exist_ok=True)

# Fig 1: link graph snapshots at t = 60 s
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
for ax, mob, name in [(axs[0], "rpgm", "RPGM"), (axs[1], "gm", "Gauss–Markov")]:
    ep = generate_episode(mobility=mob, channel="a2a", seed=10017)
    t = 120; P = ep["pos"][t]; up = ep["up"][t]
    for k in np.where(up)[0]:
        i, j = ep["iu"][k], ep["ju"][k]
        ax.plot(P[[i, j], 0], P[[i, j], 1], color=MUTED, lw=0.7, alpha=0.6, zorder=1)
    if mob == "rpgm":
        g = np.arange(len(P)) // ep["cfg"]["group_size"]
        ax.scatter(P[:, 0], P[:, 1], c=[CAT[x % 6] for x in g], s=30, zorder=2, edgecolor="white", lw=0.6)
    else:
        ax.scatter(P[:, 0], P[:, 1], c=CAT[0], s=30, zorder=2, edgecolor="white", lw=0.6)
    ax.set_title(f"{name}: {int(up.sum())} active links at t = 60 s", color=INK)
    ax.set_xlim(0, 1500); ax.set_ylim(0, 1500); ax.set_aspect("equal"); ax.set_xlabel("x [m]")
axs[0].set_ylabel("y [m]")
fig.tight_layout(); fig.savefig(OUT + "fig_graph_snapshot.pdf"); plt.close(fig)

# Fig 2: LET vs true lifetime, one test episode
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, mob, name in [(axs[0], "rpgm", "RPGM"), (axs[1], "gm", "Gauss–Markov")]:
    ep = generate_episode(mobility=mob, channel="a2a", seed=10017)
    m = ep["up"] & ~ep["censored"]; t, l = ep["life"][m], ep["let"][m]
    hb = ax.hexbin(t, l, gridsize=30, cmap="Blues", mincnt=1, bins="log", extent=(0, 60, 0, 60), linewidths=0.2)
    ax.plot([0, 60], [0, 60], color="#e34948", lw=1.5, ls="--")
    ax.text(41, 56, "perfect prediction", color="#e34948", fontsize=9, ha="center")
    ax.set_title(f"{name}: MAE = {np.abs(l - t).mean():.1f} s (episode 18)", color=INK)
    ax.set_xlabel("true residual lifetime [s]"); ax.set_xlim(0, 60); ax.set_ylim(0, 60)
axs[0].set_ylabel("LET prediction [s]")
cb = fig.colorbar(hb, ax=axs, pad=0.02); cb.set_label("samples (log scale)")
fig.savefig(OUT + "fig_let_baseline_error.pdf"); plt.close(fig)

# Fig 3: channel effect from Experiment 1 CSV
r = pd.read_csv(E1)
names = {"rpgm": "RPGM", "gm": "Gauss–Markov", "rwp": "RWP"}
fig, axs = plt.subplots(2, 3, figsize=(10, 5.6), sharey="row")
for c, mob in enumerate(["rpgm", "gm", "rwp"]):
    for row, (k, lab) in enumerate([("MAE", "MAE [s]"), ("AUPRC", "AUPRC (break within 5 s)")]):
        ax = axs[row, c]; w = 0.36
        for j, meth in enumerate(["LET", "KF-LET"]):
            g = [r[(r.mobility == mob) & (r.channel == ch) & (r.method == meth)][k] for ch in ["disk", "a2a"]]
            ax.bar(np.arange(2) + (j - .5) * w, [x.mean() for x in g], w * .9, yerr=[x.std() for x in g],
                   color=CAT[j], label=meth, capsize=3, error_kw=dict(lw=1, ecolor=MUTED), edgecolor="white", linewidth=1)
        ax.set_xticks([0, 1], ["disk model", "A2A channel"])
        if c == 0: ax.set_ylabel(lab)
        if row == 0: ax.set_title(names[mob], color=INK)
axs[1, 0].set_ylim(0, 1)
h, l = axs[0, 0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.savefig(OUT + "fig_channel_effect.pdf"); plt.close(fig)
print("figures written to", OUT)
