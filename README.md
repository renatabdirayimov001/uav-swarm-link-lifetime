# Residual Link Lifetime Prediction for UAV Swarms

Code, results and trained models for the paper:

> R. Abdirayimov et al., "Learning-Based Residual Link Lifetime Prediction for UAV Swarms under Shadowed Air-to-Air Channels: Simulator, Dataset and Benchmark," submitted to *Drones* (MDPI), 2026.

**Version 1.0 (v3 of the code).** This version fixes two issues found in an internal code audit of the
first upload: the link state is now *causal* (it depends only on the current and past slots), and every
label has a full 60-s follow-up (episodes are simulated for 660 s, samples are taken only in the first
600 s). Both properties are checked by the unit tests in `uav_swarm_sim.self_test()`. All results in
this repository and in the paper were produced with this version.

## Repository files

| File | Content |
|---|---|
| `uav_swarm_sim.py` | Simulator: RPGM, Gauss–Markov and 3-D random waypoint mobility; air-to-air channel (log-distance path loss, correlated shadowing, Rician fading); causal link state; residual-lifetime labels; LET; unit tests (`self_test()`) |
| `baselines.py` | LET and Kalman-filter (Singer model) LET; evaluation metrics |
| `models.py` | MLP, LSTM, StaticGNN and ST-GNN (PyTorch) |
| `data_torch.py` | Feature construction and batching |
| `extra.py` | SNR-trend heuristic, gradient-boosted-tree features, censoring-aware metrics (C-index, MAE over all active links) |
| `run_gbt.py` | Gradient-boosted-tree and SNR-trend baselines (CPU, ~5 min) |
| `make_figs_v3.py`, `make_fig_rel_v3.py` | Figures of the paper |
| `Experiment1_v3_Dataset_and_Baselines.ipynb` | Dataset generation, KF-LET tuning (12 settings on validation), LET / KF-LET evaluation (CPU, ~30 min) |
| `Experiment2_v3_Learned_Models.ipynb` | Training and evaluation of the neural predictors (T4 GPU, ~1.5 h) |
| `results_exp1.csv`, `dataset_stats.csv`, `kf_tuning.csv` | Per-episode results of Experiment 1, dataset statistics, KF-LET tuning |
| `results_exp2.csv`, `ablation_exp2.csv`, `cost_exp2.csv`, `config_exp2.json` | Per-episode and per-seed results of Experiment 2, SNR-feature ablation, model cost, configuration |
| `runs_exp2.zip`, `logs_exp2.zip`, `checkpoints_exp2.zip` | Per-run result files, training logs (per epoch) and the 36 trained checkpoints (`*.pt`) |
| `results_gbt.csv` | Per-episode results of the GBT and SNR-trend baselines (incl. censoring-aware metrics) |
| `fig_*.pdf` | Figures used in the paper |

## Reproducing the results

1. Open a notebook in Google Colab (*File → Upload notebook*).
2. Experiment 1 runs on CPU. Experiment 2 needs a GPU (*Runtime → Change runtime type → T4 GPU*).
3. `Run all`. Each notebook writes the source files it needs with `%%writefile`; these copies are identical to the `.py` files in this repository.
4. The GBT baseline: `python run_gbt.py` (CPU, ~5 min).

The dataset (11 million labelled samples) is not stored; it is regenerated deterministically from the episode
seeds `10000 + s` (s = 0…22; episodes 0–13 training, 14–16 validation, 17–22 test).

Note: `uav_swarm_sim.py` also contains an optional spatially correlated shadowing model
(`shadow_model="spatial"`). It is **not** used for the results in the paper; the default
(`shadow_model="link"`) reproduces them exactly.

## Requirements

Python ≥ 3.10, see `requirements.txt` (experiments were run with PyTorch 2.x on Google Colab).

## License and citation

MIT license (see `LICENSE`). Please cite the paper; citation metadata is in `CITATION.cff`.
