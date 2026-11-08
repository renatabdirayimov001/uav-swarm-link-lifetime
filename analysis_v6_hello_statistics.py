"""Hello reception statistics of the 30 test episodes (seeds 10017-10046), ideal vs fading Hello model (Section 3.4)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import uav_swarm_sim as sim
rows = []
for mob in ["rpgm", "gm", "rwp"]:
    for model in ["ideal", "fading"]:
        acc = dict(miss=0, brk=0, active=0, intervals=0, cens=0, pos5=0, slots=0, lifes=[])
        for s in range(17, 47):
            ep = sim.generate_episode(mobility=mob, seed=10000 + s, hello_model=model)
            up, hel = ep["up"], ep["hello"]
            T = up.shape[0]; Ts = int(round(ep["cfg"]["episode_s"] / ep["cfg"]["dt"])) if "episode_s" in ep["cfg"] else T
            miss = up & ~hel
            acc["miss"] += miss[:-1].sum(); acc["brk"] += (miss[:-1] & ~up[1:]).sum()
            acc["active"] += up.sum(); acc["slots"] += up.shape[0]
            acc["intervals"] += (up[1:] & ~up[:-1]).sum() + up[0].sum()
            acc["cens"] += ep["censored"][up].sum()
            acc["pos5"] += ((~ep["censored"]) & (ep["life"] <= 5.0))[up].sum()
            acc["lifes"].append(ep["life"][up & ~ep["censored"]])
        rows.append(dict(mobility=mob, hello=model, links_per_slot=acc["active"] / acc["slots"],
                         intervals_per_ep=acc["intervals"] / 30, P_break_given_miss=acc["brk"] / acc["miss"],
                         miss_share=acc["miss"] / acc["active"], censored=acc["cens"] / acc["active"],
                         pos5=acc["pos5"] / acc["active"], median_tau_uncens=float(np.median(np.concatenate(acc["lifes"])))))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results_v6/results_v6_hello_statistics.csv", index=False)
