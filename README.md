# Double-Diffusion

Research code accompanying the manuscript *Double-Diffusion: Balancing Speed, Accuracy, and
Uncertainty in Probabilistic Forecasting for Urban Sensor Networks*. The paper is currently **under
review** (submitted to ACM SIGSPATIAL 2026, Applications track); details may change, and this
repository will be updated accordingly.

Double-Diffusion integrates a **closed-form graph-heat prior** into a denoising diffusion
forecaster. The prior propagates the last observation over the sensor graph with no learned
parameters, and it plays two roles: it is the *residual target* the model generates, and it
*conditions* the denoiser. Following the Resfusion warm start, the reverse chain begins from the
noised prior partway along the schedule (about step `S'=37` of `S=100`) and only refines a small
residual, instead of synthesizing from pure noise. A compact three-axis denoiser (`DD-Net`) does the
denoising, and a graph-spectral read-out of the prior residual switches its Chebyshev spatial gate on
or off per domain, before any training.

## Repository layout

```
double_diffusion/      The method (ours)
  models.py            DD-Net denoiser + ResidualDiffusionCalibrator + graph_heat_forecast (closed-form prior)
  run_validation.py    Train / evaluate harness -- the protocol lives here (splits, K=32 full test, metrics)
  fit_prior_k.py       Offline least-squares fit of the prior coefficient k*
  diagnose_prior_k.py  Helper used by fit_prior_k (graph-heat residual energy)
  seasonal.py          Seasonal-prior index helper
  calibration.py       Optional split-conformal interval widening
prior/
  physics_ode.py       PhysicsODESolver = exact U diag(e^{-k lambda t}) U^T x_t (the graph-heat prior)
preprocessing/
  dataset.py           Chronological 60/20/20 splits, sliding windows, loaders, graph tensors
  graph_utils.py       Distance graph + normalized Laplacian eigendecomposition
  build_beijing.py     Build data/beijing/processed   (air quality)
  build_athens.py      Build data/athens/processed    (air quality)
  build_traffic.py     Build data/pems04|pems08/processed (traffic)
diagnostics/
  run_spatial_gate_diagnostic.py   The graph-spectral gate rule (rho_G -> gate on/off)
  spectral_gate_all_summaries.json Reference output of the diagnostic
configs/
  kfit_l2_hpc.json     Fitted prior coefficients k* per dataset (beijing 0.1, athens 0.05, pems08/pems04 0.02)
  *.yaml               Per-dataset reference configurations
```

Everything here is our own code. The baseline models compared in the paper are **not** included; see
[Baselines](#baselines).

## Installation

```bash
# 1. PyTorch for your CUDA build (example: CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2. the rest
pip install -r requirements.txt
```

Python 3.10+ and PyTorch >= 2.4 are recommended. A CUDA GPU is recommended for training; inference and
the diagnostics run on CPU.

## Data preparation

The raw datasets are public and are **not** redistributed here. The four networks are two air-quality
deployments (Beijing, Athens) and two traffic networks (PEMS08, PEMS04). Place the raw files where the
build scripts expect them and run:

```bash
python preprocessing/build_beijing.py
python preprocessing/build_athens.py
python preprocessing/build_traffic.py        # builds pems04 and pems08
```

Each script writes `data/<name>/processed/` containing the windowed tensors plus the shared graph
tensors (`adj.npy`, `laplacian.npy`, `eigenvalues.npy`, `eigenvectors.npy`) and the per-dataset
normalization (`mean.npy`, `std.npy`). All models use `H=12` input steps and `H'=24` forecast steps
with identical windows and normalization.

## Offline configuration (once per dataset, on the train split)

**Prior coefficient `k*`.** The fitted values are already provided in `configs/kfit_l2_hpc.json` and
are loaded automatically by `--auto-k-file`. To regenerate them, run `double_diffusion/fit_prior_k.py`.

**Spatial gate.** The gate is set from a single graph-spectral statistic of the prior residual, with no
forecast evaluation:

```bash
python diagnostics/run_spatial_gate_diagnostic.py
```

This prints `rho_G` (low graph-frequency residual energy fraction) per dataset and writes
`diagnostics/spectral_gate_output.json`. The rule is `gate ON` iff `rho_G < 1/2`. On the four networks
the gate is **off** for air quality (residual already smooth) and **on** for traffic (sharp spatial
structure remains): pass `--no-spatial` for Beijing/Athens, and omit it for PEMS08/PEMS04.

## Training and evaluation

`run_validation.py` trains the model and then scores the full test split at `K=32` samples, reporting
MAE/RMSE (raw units) and CRPS (normalized), under both the single-shot (`initial`) and `rolling` views.

Air quality (gate **off**):

```bash
python double_diffusion/run_validation.py \
  --dataset beijing --history-len 12 --future-len 24 \
  --model fastdd --prior-mode graph_ode \
  --auto-k-file configs/kfit_l2_hpc.json --auto-k-kind scalar \
  --d-model 64 --n-blocks 4 --cheb-order 3 \
  --diffusion-steps 100 --beta-end 0.2 --schedule linear \
  --no-spatial \
  --batch-size 32 --epochs 200 --patience 20 \
  --rolling-train --reveal-prefixes 6 \
  --n-samples 32 --eval-policies initial,rolling --seed 0
```

Traffic (gate **on**) is the same command with `--dataset pems08` (or `pems04`) and **without**
`--no-spatial`. Results are written to `double_diffusion/outputs/<run_name>/results.json`. Add `--amp`
for mixed precision, and `--device cpu` to force CPU.

To evaluate an existing checkpoint without retraining, pass `--eval-only <path/to/best_model.pt>` with
the same architecture flags the checkpoint was trained with.

## Reproducing the paper numbers

Table 2 reports seed-0, full-test results with `K=32` for the configuration above: one architecture
(`d_model=64`, `n_blocks=4`, `cheb_order=3`, `S=100`, `beta_end=0.2`), a per-dataset prior coefficient
from `configs/kfit_l2_hpc.json`, and the per-domain gate from the diagnostic (off for air quality, on
for traffic). The same configuration is applied to all four datasets; only the gate and `k*` vary, and
both are set offline from the training split.

## Baselines

The baseline comparators (STGCN, DLinear, GMSDR, GMAN, DeepAR, MC Dropout, DiffSTG, CSDI) are run from
their **official** implementations and are not redistributed in this repository. See the corresponding
references in the paper for each method's source. Our evaluation uses identical splits, windows,
normalization, and the same `K=32` full-test protocol for every probabilistic model.

## License

The code in this repository is released for research use. Third-party baseline implementations referred
to above remain under their own licenses.
