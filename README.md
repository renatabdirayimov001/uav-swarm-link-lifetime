# Residual Link Lifetime Prediction for UAV Swarms

Code, results and trained models for the paper:

> R. Abdirayimov et al., "Learning-Based Residual Link Lifetime Prediction for UAV Swarms under Shadowed Air-to-Air Channels: Simulator, Dataset and Benchmark," submitted to *Drones* (MDPI), 2026.

## Version 2.2 (code v6) — used for the analyses added in the third revision

Backward compatible: with the default settings the simulator generates exactly the same episodes as v2.0/v2.1
(SHA-256 fingerprints unchanged), and the published checkpoints load unchanged; re-evaluated on the six test
episodes of the paper, they reproduce the published per-episode results exactly.

1. **Realistic Hello reception** `hello_model="fading"` in `uav_swarm_sim.py`: a Hello is decoded only when the
   instantaneous (Rician-faded) SNR reaches the threshold (optional extra random loss `hello_ploss`); the link
   state follows the received Hellos, as in a protocol.
2. **SNR-corrected LET and feature studies** (`extra_v6.py`): LET with the effective range estimated from the SNR
   of the received Hellos (with and without decay of the shadowing estimate), GBT feature groups for the ablation,
   and neighbourhood summaries for a per-link GBT.
3. **Configurable model width and depth** (`models.py`) for an equal-budget hyperparameter search.
4. **`run_v6.py`**: 30 test episodes (seeds 10017–10046) for all new results. Neural part (GPU): `check`,
   `reeval` (published checkpoints on 30 episodes), `fading`, `window` (LSTM, W = 4/8/16), `tune` (random search,
   validation only), `final` (best configuration × 3 seeds), `pack`. CPU part (`NO_NEURAL=1`): `ref`, `letsnr`,
   `ablation`, then `fading`, `window`, `tune`, `final`. Training instructions in Russian: `TRAINING_v6_RU.md`.
5. **`analysis_v6_hello_statistics.py`, `analysis_v6_oracle_let_snr.py`**: Hello-loss statistics and the
   SNR-corrected LET with the exact large-scale SNR.

### Results of v2.2 (`results_v6/`)

| File | Content |
|---|---|
| `results_v6_all_runs.csv` | Per-episode metrics of every run (column `run_on`: `cpu` = analytical predictors and GBT, `gpu` = neural predictors) |
| `results_v6_tuning.csv`, `results_v6_best_configs.csv` | Equal-budget hyperparameter search (validation episodes only) and the selected configurations |
| `results_v6_kf_tuning.csv` | KF-LET grid search on the validation episodes |
| `results_v6_hello_statistics.csv`, `results_v6_oracle_let_snr.csv` | Outputs of the two analysis scripts |
| `make_tables_v6.py` | Recomputes Table 7, Table 9, the fading rows of Table 10, the window study and the search summary |
| `checkpoints_v6.zip`, `training_logs_v6.zip`, `progress_*.log` | Neural checkpoints and GBT models, per-epoch training logs, run logs |

```
python results_v6/make_tables_v6.py
```

## Version 2.1 (code v5) — used for all results in the second revision

Changes with respect to version 2.0 (all backward compatible: with the default settings the simulator
generates exactly the same episodes, and the features are identical for the default k = 2):

1. **Spatially correlated shadowing field** `shadow_model="spatial"`: a static 2-D Gaussian field
   (correlation length 100 m) averaged along each link; dataset regenerated and all predictors retrained
   (`python run_v4.py spatial` for the CPU part, `Experiment_v5_Spatial.ipynb` for the neural part).
2. **Extended KF-LET grid** τm ∈ {0.1, …, 10} s × q ∈ {0.05, …, 2} and **calibrated LET** (a + b·LET fitted
   on the training episodes, row `LET-cal`).
3. **Sensitivity study** (`python run_v4.py sens`): window W = 4/16, Hello interval 0.25/1 s, break rule
   k = 1/3, mean speed 6/20 m/s. The Hello information model now supports any k (sequential hold, missed-Hello age).
4. **Retraining diagnostic** for N = 50 and σ = 4 dB under RPGM (`python run_v4.py retrain`).
5. **Event-level and route-level metrics** (`analysis_v5.py`): dataset counts (link intervals, break events),
   interval-weighted MAE, break recall at a matched 5% false-alarm rate, widest-path routing (route breaks per minute).
6. **Dataset documentation** in `dataset_meta/`: `DATASHEET.md`, SHA-256 fingerprints of all 276 canonical
   episodes (`episode_sha256.csv`) with `verify_dataset.py`, pinned environment (`requirements-lock.txt`,
   `Dockerfile`), data license `LICENSE-DATA` (CC BY 4.0).

| File (v2.1) | Content |
|---|---|
| `results_v5/results_v5_all_runs.csv` | Per-run, per-test-episode metrics of all settings (main, disk, motion, spatial, general, retrain, sensitivity) |
| `results_v5/summary_all.csv`, `results_v5/paired_v5.csv` | Means per setting and predictor; paired differences with 95% CIs |
| `results_v5/event_metrics.csv`, `results_v5/route_metrics.csv`, `results_v5/dataset_counts.csv` | Event-level, route-level and dataset statistics |
| `results_v5/kf_tuning_v5.zip`, `results_v5/kf_edge_tau_small.csv` | KF-LET validation grids |
| `checkpoints_logs_v5_spatial.zip` | Neural checkpoints and training logs for the spatial shadowing field |
| `make_tables_v5.py`, `make_figs_v4.py` | Tables and figures of the second revision |

Reproduction: `NO_NEURAL=1 python run_v4.py main` (then `disk`, `general`, `motion`, `spatial`, `retrain`, `sens`),
neural predictors with the two Colab notebooks, then `python analysis_v5.py`, `python make_tables_v5.py`,
`python make_figs_v4.py`. Verify the dataset with `python dataset_meta/verify_dataset.py`.

## Version 2.0 (code v4) — used for the first revision

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

The default `shadow_model="link"` reproduces the main configuration exactly; `"motion"` and `"spatial"`
are the alternative shadowing models used in the robustness sections of the paper.

## Requirements

Python ≥ 3.10, see `requirements.txt` (experiments were run with PyTorch 2.x on Google Colab).

## License and citation

Code: MIT license (see `LICENSE`). Generated data and fingerprints: CC BY 4.0 (`dataset_meta/LICENSE-DATA`). Please cite the paper; citation metadata is in `CITATION.cff`.
