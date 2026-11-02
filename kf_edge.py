import os, sys, itertools, pandas as pd
os.environ["OUT"] = os.environ.get("OUT", "out_v4"); sys.argv = ["x", "none"]
import run_v4 as R
from data_torch import EpisodeBank
rows = []
for mob in ["rpgm", "gm", "rwp"]:
    va = EpisodeBank(mob, R.VAL, device="cpu")
    R.KF_GRID = list(itertools.product([0.1, 0.25], [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]))
    _, _, kt = R.tune_kf(va); kt["mobility"] = mob; rows.append(kt); print(mob, kt.val_MAE.min(), flush=True)
pd.concat(rows).to_csv("out_v4/kf_edge_tau_small.csv", index=False)
