"""Regenerate every canonical episode and compare its SHA-256 fingerprint with episode_sha256.csv."""
import hashlib, numpy as np, pandas as pd
import uav_swarm_sim as sim
ref = pd.read_csv("episode_sha256.csv"); bad = 0
over = {"link": {}, "motion": dict(shadow_model="motion"), "spatial": dict(shadow_model="spatial")}
for r in ref.itertuples():
    ep = sim.generate_episode(mobility=r.mobility, channel=r.channel, seed=int(r.seed), **over[r.shadowing])
    h = hashlib.sha256()
    for k in ["pos_obs", "vel_obs", "snr_meas", "up", "hello", "life", "censored", "let"]:
        a = np.ascontiguousarray(ep[k]); h.update(k.encode()); h.update(a.astype(np.float32 if a.dtype.kind == "f" else a.dtype).tobytes())
    if h.hexdigest() != r.sha256:
        bad += 1; print("MISMATCH", r.shadowing, r.mobility, r.channel, r.seed)
print(f"{len(ref) - bad} of {len(ref)} episodes identical")
