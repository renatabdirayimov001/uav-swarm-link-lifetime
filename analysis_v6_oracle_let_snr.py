"""SNR-corrected LET with the exact large-scale SNR (oracle) on the 30 test episodes (Section 4.6)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import uav_swarm_sim as sim, extra_v6 as x6, data_torch as dt, extra as ex
W, H = 8, 60.0
fade = x6.rician_db_mean(7.6)
rows = []
for name, over in [("main", {}), ("spatial", dict(shadow_model="spatial"))]:
    for mob in ["rpgm", "gm", "rwp"]:
        for s in range(17, 47):
            ep = sim.generate_episode(mobility=mob, seed=10000 + s, **over)
            up = ep["up"]; miss = dt.held_mask(ep); rx = up & ~miss
            p, _ = x6.let_snr(ep["pos_obs"], ep["vel_obs"], ep["snr_ls"] + fade, rx, up, miss, ep["iu"], ep["ju"], ep["cfg"], W, H, True)
            sl = slice(W - 1, None)
            r = ex.evaluate_ext(p[sl], ep["life"][sl], ep["censored"][sl], up[sl], H)
            rows.append(dict(setting=name, mob=mob, episode=s, AUPRC_5=r["AUPRC_5"], MAE_all=r["MAE_all"]))
        d = pd.DataFrame(rows); print(name, mob, d[(d.setting == name) & (d.mob == mob)].AUPRC_5.mean().round(3), flush=True)
pd.DataFrame(rows).to_csv("results_v6/results_v6_oracle_let_snr.csv", index=False)
