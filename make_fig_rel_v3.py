import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight"})
INK = "#2b2b29"
meths = ["KF-LET", "GBT", "MLP", "LSTM", "StaticGNN", "ST-GNN"]
CAT = ["#eb6834", "#7a7a72", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
# (MAE, AUPRC): LET first, then meths -- Experiment 2 v3 (Table 5)
e2 = {"RPGM": [(11.33,.618),(11.41,.604),(12.43,.750),(12.78,.758),(12.17,.755),(11.91,.756),(12.38,.755)],
      "Gauss–Markov": [(8.50,.659),(8.97,.650),(6.34,.800),(6.48,.802),(6.35,.803),(6.44,.800),(6.40,.801)],
      "RWP": [(7.40,.624),(7.52,.624),(6.11,.790),(6.20,.789),(6.06,.797),(6.27,.778),(6.22,.787)]}
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
mobs = list(e2); y = np.arange(len(mobs)); h = 0.13
for k, (ax, title) in enumerate(zip(axs, ["MAE reduction vs. LET [%]  (higher is better)", "AUPRC gain vs. LET [%]  (higher is better)"])):
    for j, m in enumerate(meths):
        vals = []
        for mob in mobs:
            let, x = e2[mob][0], e2[mob][j + 1]
            vals.append(100 * (let[0] - x[0]) / let[0] if k == 0 else 100 * (x[1] - let[1]) / let[1])
        pos = y + (j - 2.5) * h
        ax.barh(pos, vals, h * 0.9, color=CAT[j], label=m, edgecolor="white", linewidth=0.8)
        for p, v in zip(pos, vals):
            ax.text(v + (0.6 if v >= 0 else -0.6), p, ("0" if abs(v) < 0.5 else f"{v:+.0f}"), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=INK, lw=1); ax.set_title(title, fontsize=10, color=INK); ax.set_xlim(-20, 37)
axs[0].set_yticks(y, mobs); axs[0].invert_yaxis()
h_, l_ = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.legend(h_, l_, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.savefig(sys.argv[1] + "fig_relative_gain.pdf")
