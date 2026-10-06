# MultiAttLLM — Electricity Demand & Generation Forecasting

A PyTorch reimplementation and extension of **MultiAttLLM**, the
multi-attention LLM-based forecasting architecture from:

> Z. Hu, Y. Gao, L. Sun, M. Mae, *"A novel attention-enhanced LLM approach
> for accurate power demand and generation forecasting,"* Renewable Energy
> 252 (2025), 123465.

This codebase forecasts hourly **electricity demand**, **renewable energy
generation**, and **coal generation** for four Japanese regions (Tokyo,
Hokkaido, Tohoku, Kyushu), using a frozen GPT-2 backbone reprogrammed for
time-series input via cross-attention, combined with a separate covariate
branch and a fusion decoder.

In addition to reproducing the paper's architecture, this project adds a
set of **improved-architecture modifications (M1–M4)** — RevIN, a learned
text-prototype bank with gating, a channel-independent fusion decoder, and
an adaptive gated fusion mechanism — selectable alongside the paper's
original design for direct, like-for-like comparison.

---

## Contents

- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
  - [Google Colab](#google-colab)
  - [Local / other environments](#local--other-environments)
- [Running the project](#running-the-project)
  - [CLI flags](#cli-flags)
  - [Examples](#examples)
- [Configuration](#configuration)
- [Model architectures](#model-architectures)
- [Data](#data)
- [Outputs](#outputs)
- [Console output](#console-output)
- [CPU support](#cpu-support)
- [Known limitations](#known-limitations)

---

## Features

- **Two selectable model architectures** sharing one training pipeline:
  the paper's original design, and an improved architecture with four
  additional modifications (M1–M4), toggleable individually or together.
- **Frozen, cached GPT-2 backbone** — downloaded once, reused on every
  subsequent run.
- **CLI flags** for the common experiment knobs (save model weights, switch
  architecture, override prototype count) without editing config files.
- **Readable, grouped configuration** — every setting lives in one of
  three plain-Python config files, with no cryptic abbreviations.
- **Clean console output** — a grouped summary of every config value at
  startup, one line per training epoch, and a formatted metrics table at
  test time.
- **Automatic, publication-quality plots** — actual-vs-predicted forecasts
  for every target, generated automatically after each test run, with real
  calendar dates on the x-axis.
- **Runs on GPU or CPU** (CPU is supported but significantly slower — see
  [CPU support](#cpu-support)).

---

## Repository structure

```
.
├── configs/
│   ├── data_config.py       # dataset paths, feature/target columns, split sizes, window lengths
│   ├── env_config.py        # device, training loop, optimizer/scheduler, checkpoint paths
│   ├── model_config.py      # architecture hyperparameters and the M1–M4 / base-model switches
│   └── __init__.py
├── data/
│   ├── tokyo.csv
│   ├── hokkaido.csv
│   ├── tohoku.csv
│   └── kyushu.csv
├── engine/
│   ├── environment.py       # seeding, device resolution, weights-folder setup
│   └── trainer.py           # training loop, evaluation, checkpointing, run naming
├── architecture/
│   ├── multiattllm.py       # improved architecture (M1–M4 on)
│   └── base_multiattllm.py  # paper's original architecture (M1–M4 off by default)
├── modules/
│   ├── attention.py
│   ├── cross_attention.py
│   ├── decoder.py
│   ├── adaptive_gated_fusion.py   # M4
│   ├── embed.py
│   ├── flatten_head.py
│   ├── llm_block.py
│   └── revin.py                   # M1
├── utils/
│   ├── data_loader.py
│   ├── logging_utils.py
│   ├── masking.py
│   ├── metrics.py
│   ├── time_features.py
│   └── tools.py              # checkpointing helpers, LR schedules, plotting
├── weights/                   # GPT-2 download cache (auto-created)
├── models/                    # saved model checkpoints (auto-created, only with --save-model)
├── results/                   # per-run outputs (auto-created)
├── main.py                    # entry point
└── requirements.txt
```

---

## Requirements

- Python 3.10+ (developed and tested on 3.13)
- A CUDA-capable GPU is strongly recommended (see [CPU support](#cpu-support))
- See `requirements.txt` for exact package versions. Notably:
  - `torch >= 2.3.0` — required for the mixed-precision API used in
    `engine/trainer.py`; older versions will raise an `AttributeError` if
    mixed precision is enabled.
  - `transformers` — for the frozen GPT-2 backbone.

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Setup

### Google Colab

The project is developed primarily against Google Colab. A typical setup
cell looks like this — store a GitHub personal access token as a Colab
secret named `GH_TOKEN` first (Colab's key icon in the left sidebar):

```python
import os
from google.colab import userdata

username = "toqlune"
repo_name = "fnyp"

# Clone repository without authentication
!git clone https://github.com/{username}/{repo_name}.git

%cd {repo_name}

!pip install -r requirements.txt
!python main.py
```

A few Colab-specific notes:

- Make sure the Colab runtime has a GPU attached (**Runtime → Change
  runtime type → GPU**) before running `main.py`, or the run will
  automatically fall back to CPU (see [CPU support](#cpu-support)).
- The first run downloads GPT-2 (~500 MB) into `weights/`. If this fails
  with an SSL certificate error (sometimes seen in Colab/corporate
  networks), uncomment the `CURL_CA_BUNDLE` line near the top of
  `main.py`.
- You may see `Warning: You are sending unauthenticated requests to the HF
  Hub.` on first download — this is informational only (Hugging Face's
  generic anonymous-download notice), not an error.
- Files written under `weights/`, `results/`, and `models/` persist only
  for the lifetime of the Colab runtime unless you commit/push them or
  mount Google Drive — plan accordingly for long training runs.

### Local / other environments

```bash
git clone https://github.com/toqlune/fnyp.git
cd fnyp
pip install -r requirements.txt
python main.py
```

If installing PyTorch separately for a specific CUDA version, follow
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
rather than relying on the default PyPI wheel — see the comment at the top
of `requirements.txt`.

---

## Running the project

```bash
python main.py
```

With no flags, this trains and tests the **improved architecture** (all of
M1–M4 enabled) on the dataset and target region configured in
`configs/data_config.py`, and saves nothing beyond the normal working
checkpoint and results files (no persisted model weights unless asked).

### CLI flags

| Flag | Short | Takes a value | Effect |
|---|---|---|---|
| `--save-model` | `-sm` | no | Persists the trained model's weights to `configs.saved_models_dir` (`./models/`) after training and testing. Without it, no standalone model file is written. |
| `--base-model` | `-bm` | no | Runs the paper's original base architecture instead of the improved one (no M1–M4). Equivalent to setting `use_base_model = True` in `configs/model_config.py`; the flag only ever turns this *on*, it never forces it off, so the config-file setting still works independently. |
| `--num-text-proto N` | `-ntp N` | yes (positive integer) | Overrides `configs.num_text_prototypes` for this run. Improved-model only — has no effect (and prints a note) when `--base-model` is also set, since base mode sizes its reprogramming bank with `word_projection_size` instead. |

Flags combine freely, and both short and long forms work (`--num-text-proto=128` also works).

### Examples

```bash
# Improved architecture, default settings (num_text_prototypes=512)
python main.py

# Improved architecture, override the prototype count
python main.py -ntp 64

# Paper's original base architecture
python main.py -bm

# Base architecture, also persist the trained weights
python main.py -bm -sm

# Improved architecture, save the trained weights
python main.py -sm
```

---

## Configuration

All settings live in `configs/`, grouped by concern:

- **`data_config.py`** — dataset file paths, which columns are features vs.
  targets, train/validation/test split sizes, lookback window and
  forecast-horizon lengths, scaling options.
- **`model_config.py`** — architecture hyperparameters (model dimension,
  attention heads, decoder layers, GPT-2 layers kept, dropout), plus the
  switches that select and configure the model variant:
  - `use_base_model` — paper's original architecture vs. the improved one
    (also settable via `-bm`).
  - `use_revin` — base-mode only; isolates the RevIN (M1) modification on
    its own, independent of the other three.
  - `num_text_prototypes` — improved-model only (also settable via `-ntp`).
  - `word_projection_size` — base-model only, the paper's vocabulary
    reprogramming size (`d_wproj`).
- **`env_config.py`** — device selection, training loop (epochs, batch
  size, early stopping), optimizer and learning-rate schedule, and where
  checkpoints/weights/saved models are written.

There's no central place that merges these — `main.py` reads all three
directly and merges them into one configuration object at startup, which
is also what's printed in the CONFIGS table at the start of every run (see
[Console output](#console-output)).

---

## Model architectures

Both architectures share the same overall pipeline — a frozen GPT-2
encoder processing the target series via cross-attention, a separate
linear covariate branch, and a fusion decoder combining both — but differ
in four specific respects, referred to throughout the codebase as **M1–M4**:

| | Improved (default) | Base (`-bm` / `use_base_model=True`) |
|---|---|---|
| **M1 — Normalization** | Learnable-affine RevIN, scoped to target channels | Plain (non-learnable) per-instance normalization by default; set `use_revin=True` to use the same learnable RevIN as the improved model, isolated from M2–M4 |
| **M2 — LLM reprogramming** | A freshly learned text-prototype bank + GLU gate | GPT-2's own vocabulary embedding table, linearly projected down to `word_projection_size` entries, no gate |
| **M3 — Channel handling** | Channel-independent: each target channel processed as its own sequence | Channel-dependent: all target channels embedded and decoded together |
| **M4 — Branch fusion** | A learned adaptive gate blends the self-attention and cross-attention branches | A plain additive residual |

Both produce the same output shape and are driven by the same training
loop in `engine/trainer.py` — switching between them changes nothing about
how a run is invoked beyond the flags/config above.

---

## Data

Expected CSV format (see `data/*.csv` for the shipped regions): one row per
hour, a `date` column, and columns for electricity demand/generation by
source plus weather variables. Which columns count as model inputs
(`feature_columns`) and which are forecast targets (`target_columns`) is
set in `configs/data_config.py` — targets must also be included in
`feature_columns`.

Zero-shot cross-region evaluation is supported: set
`source_data_file_name` to the region a model was trained on and
`data_file_name` to a different region to evaluate on, without retraining.

---

## Outputs

Every run creates a uniquely named identifier (`run_id`), e.g.
`2026-01-01T01-01-00_num-text-prototypes-512`, built from a timestamp plus
whichever config value(s) distinguish that run (shown via
`run_id_config_keys` in `engine/trainer.py`). This identifier is used
consistently for:

- **`results/<run_id>/`** — the working directory for that run, containing:
  - `checkpoints/checkpoint` — the best-validation-loss model weights,
    used internally to reload before testing (not the same as
    `--save-model`'s output).
  - `loss_records.csv` — per-epoch train/validation loss.
  - `predictions_<region>.csv` — true and predicted values per target,
    with an aligned `date` column, over the non-overlapping forecast
    windows in the test set.
  - `metrics_<region>.csv` — MSE/RMSE/NRMSE/MAE/MAPE/RAE/R²/correlation
    per target.
  - `<target>.png` — an actual-vs-predicted plot per target (see below).
  - `pred_<region>.npy` / `true_<region>.npy` — raw prediction/ground-truth
    arrays.
- **`results/results_<run_id>.csv`** — a copy of that run's metrics, one
  level up, for easy comparison across runs.
- **`models/model_<run_id>.pth`** — only created with `--save-model`/`-sm`.
- **`weights/`** — the GPT-2 download cache, shared across every run and
  every architecture variant.

Plots are generated automatically for every target at the end of testing,
showing the last 7 forecast windows (a readable snapshot, not the full
test year) with real calendar dates on the x-axis, consistent styling
(600 DPI, dashed grid, unit-labeled y-axis) across every run.

---

## Console output

Every run prints:

1. A **CONFIGS** table at startup — every active configuration value,
   grouped by Data / Model / Environment, two columns, so the exact
   settings behind any given run are always visible and loggable.
2. A **TRAINING MODEL [...]** / **TESTING MODEL [...]** header before each
   phase, tagged with whichever config value(s) distinguish this run
   (e.g. `[num_text_prototypes=512]` or `[word_projection_size=2000,
   use_revin=True]`).
3. One line per training epoch (train/validation loss, learning rate,
   epoch time, estimated time remaining), plus short early-stopping
   notices.
4. A formatted per-target metrics table at the end of testing.

---

## CPU support

The pipeline runs on CPU if no GPU is available or `use_gpu=False` in
`env_config.py` — this is automatic, no configuration needed. However:

- **Performance**: GPT-2's forward/backward passes dominate runtime
  regardless of device. Expect CPU training to take substantially longer
  than GPU — plausibly hours rather than minutes for a full run. For a
  quick smoke test, temporarily lower `num_train_epochs` in
  `env_config.py`.
- **Do not enable `use_mixed_precision`** on a CPU-only machine — it
  relies on a CUDA-specific autocast/gradient-scaler path and will raise
  an error if there's no GPU.

---

## Known limitations

- `model_config.py`'s `model_dimension` (32) and `feedforward_dimension`
  (64) don't match the values reported in the paper's Table 3 (`d_model`
  = 64, `d_ff` = 128) — flagged in the config file itself rather than
  silently changed, since the paper's own sensitivity analysis suggests
  the architecture is fairly robust to this choice.
- `word_projection_size` defaults to 2000 (the paper's stated value); an
  earlier standalone reference implementation of the base architecture
  used 3000 — this is noted in `model_config.py` if you need to match that
  variant instead.
- A small number of parameters inherited from the original codebase
  (`attention_scaling_factor`, the `tau`/`delta` arguments threaded
  through the attention modules) are accepted but have no effect on model
  behavior — documented in-place in `modules/attention.py` rather than
  silently removed.
