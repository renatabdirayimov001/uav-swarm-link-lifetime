"""Aggregate the v4 runs: per-episode means over seeds, mean +- std over test episodes,
seed std, paired differences with 95% CI (t, 5 dof) and Wilcoxon p."""
import glob, os, re, sys, numpy as np, pandas as pd
from scipy import stats

DIRS = sys.argv[1:] or ["./out_v4"]
METRICS = ["MAE", "RMSE", "R2", "AUC", "AUPRC", "MAE_all", "Cindex", "AUPRC_2", "AUPRC_10", "pos_rate", "pos_2", "pos_10"]


def load():
    fs = []
    for d in DIRS:
        fs += glob.glob(f"{d}/runs/*.csv") + glob.glob(f"{d}/**/runs/*.csv", recursive=True)
    out = []
    for f in sorted(set(fs)):
        b = os.path.basename(f)
        if "kf_tuning" in b: continue
        df = pd.read_csv(f)
        m = (re.match(r"(main|motion|disk|spatial|N50r|std4r|sens-[A-Za-z0-9]+)_(rpgm|gm|rwp)_", b)
             or re.match(r"general_(N20|N50|std4)_(rpgm|gm|rwp)_", b))
        if not m: continue
        df["setting"], df["mobility"] = m.group(1), m.group(2)
        out.append(df)
    df = pd.concat(out, ignore_index=True)
    df["method"] = df["method"].replace({"STGNN": "ST-GNN"})
    return df.drop_duplicates(subset=["setting", "mobility", "method", "seed", "episode"])


def per_episode(df):
    return df.groupby(["setting", "mobility", "method", "episode"])[METRICS].mean().reset_index()


def summary(df):
    ep = per_episode(df)
    s = ep.groupby(["setting", "mobility", "method"])[METRICS].agg(["mean", "std"])
    seed = (df.groupby(["setting", "mobility", "method", "seed"])[METRICS].mean()
              .groupby(["setting", "mobility", "method"]).std())
    return s, seed


def paired(df, setting, mob, a, b, metric):
    """mean of (a - b) over episodes, 95% CI, wins of a (lower is better for errors), Wilcoxon p."""
    ep = per_episode(df)
    x = ep[(ep.setting == setting) & (ep.mobility == mob) & (ep.method == a)].set_index("episode")[metric]
    y = ep[(ep.setting == setting) & (ep.mobility == mob) & (ep.method == b)].set_index("episode")[metric]
    d = (x - y).dropna()
    n = len(d)
    if n < 2: return None
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    try:
        p = stats.wilcoxon(d).pvalue
    except ValueError:
        p = np.nan
    return dict(setting=setting, mobility=mob, a=a, b=b, metric=metric, diff=d.mean(), lo=d.mean() - h,
                hi=d.mean() + h, n=n, a_lower=int((d < 0).sum()), p=p)


if __name__ == "__main__":
    df = load()
    print(df.groupby(["setting", "mobility", "method"]).size().unstack(0).fillna(0).astype(int))
    s, seed = summary(df)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(s.xs("mean", axis=1, level=1).round(3))
