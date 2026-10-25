# Residual Link Lifetime Prediction for UAV Swarms

Code, results and trained models for the paper:

> R. Abdirayimov et al., "Learning-Based Residual Link Lifetime Prediction for UAV Swarms under Shadowed Air-to-Air Channels: Simulator, Dataset and Benchmark," submitted to *Drones* (MDPI), 2026.

## Version 2.0 (code v4) — used for all results in the revised paper

Changes with respect to version 1.0 (code v3):

1. **Hello information model.** A Hello message is received in a slot exactly when the link is available.
   In a slot in which a link is still active but its Hello is missed (the first k-1 unavailable slots), the
   pair features of the previous slot are held, the LET is aged by one slot and a missed-Hello flag is set
   (`data_torch.py`). GNN messages are exchanged only over links with a received Hello (`models.py`).
2. **New ablation:** StaticGNN without neighbour aggregation (`StaticGNN-noagg`).
3. **Censoring-aware metrics for all predictors** (MAE over all active links, C-index, AUPRC for 2/5/10 s).
4. **Alternative shadowing model** `shadow_model="motion"` (correlation driven by the displacement of both UAVs).
5. **Generalisation tests without retraining:** N = 20, N = 50, shadowing std 4 dB.
6. **Single-slot inference time** of all predictors, including the GBT, on one CPU.
7. **Statistics:** mean ± std over test episodes for all methods, seed std separately, paired 95% confidence intervals.

The simulator with the default settings (`shadow_model="link"`) generates exactly the same episodes as version 1.0.

| File (v2.0) | Content |
|---|---|
| `uav_swarm_sim.py` | Simulator (v4): mobility, A2A channel, causal link state, labels, LET, Hello flag, `shadow_model="motion"`, unit tests |
| `data_torch.py`, `models.py`, `extra.py`, `baselines.py` | Features with the Hello information model, predictors, metrics |
| `run_v4.py` | All v2.0 experiments: `main`, `disk`, `motion`, `general`, `timing` (resumable) |
| `Experiment_v4_Neural.ipynb` | Colab notebook for the neural predictors (T4 GPU, about 3 h) |
| `results_v4_all_runs.csv` | Per-run, per-test-episode metrics of every predictor and setting |
| `paired.csv`, `timing_cpu_v4.csv` | Paired differences with 95% CIs; CPU latency of all predictors |
| `aggregate.py`, `make_tables.py`, `make_figs_v4.py` | Aggregation, LaTeX tables and figures of the revised paper |
| `checkpoints_v4.zip`, `logs_v4.zip` | Trained models (neural `.pt`, GBT `.pkl`) and training logs |
| `fig_*.pdf` | Figures of the revised paper |

Reproduction:
- CPU part (LET, KF-LET, SNR trend, GBT): `NO_NEURAL=1 python run_v4.py main`, then `disk`, `general`, `motion`
- GPU part (neural predictors): open `Experiment_v4_Neural.ipynb` in Google Colab with a T4 GPU and run all cells
- Timing: `FORCE_CPU=1 python run_v4.py timing`
- Tables and figures: `python make_tables.py`, `python make_figs_v4.py`

## Version 1.0 (code v3) — kept for reference

Version 1.0 fixed two issues found in an internal code audit of the first upload: the link state is
*causal* and every label has a full 60-s follow-up. The v1.0 notebooks, results and checkpoints are kept below.

### Version 1.0 files

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
