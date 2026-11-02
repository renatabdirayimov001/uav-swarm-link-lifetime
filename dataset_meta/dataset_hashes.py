"""SHA-256 fingerprints of the canonical episodes (verify that a regenerated dataset is identical)."""
import hashlib, itertools, sys, numpy as np, pandas as pd, platform
import uav_swarm_sim as sim
rows = []
settings = [("link", {}), ("motion", dict(shadow_model="motion")), ("spatial", dict(shadow_model="spatial"))]
for (sh, over), mob, ch in itertools.product(settings, ["rpgm", "gm", "rwp"], ["a2a", "disk"]):
    if sh != "link" and ch == "disk": continue
    for s in range(23):
        ep = sim.generate_episode(mobility=mob, channel=ch, seed=10000 + s, **over)
        h = hashlib.sha256()
        for k in ["pos_obs", "vel_obs", "snr_meas", "up", "hello", "life", "censored", "let"]:
            a = np.ascontiguousarray(ep[k]); h.update(k.encode()); h.update(a.astype(np.float32 if a.dtype.kind == "f" else a.dtype).tobytes())
        rows.append(dict(shadowing=sh, mobility=mob, channel=ch, seed=10000 + s, split="train" if s < 14 else ("val" if s < 17 else "test"),
                         samples=int(ep["up"].sum()), sha256=h.hexdigest()))
    print(sh, mob, ch, flush=True)
df = pd.DataFrame(rows); df.to_csv("dataset_meta/episode_sha256.csv", index=False)
open("dataset_meta/hash_environment.txt", "w").write(f"python {platform.python_version()}\nnumpy {np.__version__}\n")
print("done", len(df))
