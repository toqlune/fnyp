"""
Data configuration — electricity demand/generation forecasting (Tokyo).

`source_data_path` defaults to the same file as `data_path`, giving a
standard single-region train/val/test split. For zero-shot cross-region
evaluation (paper Section 4.6 / Table 8), set `data_path` to the target
region's CSV while leaving `source_data_path` pointing at the region the
model was trained on — the loader fits its scalers on `source_data_path`'s
train split and evaluates on `data_path`.
"""

# ── File locations ─────────────
root_path = './data'
data_path = 'tokyo.csv'
source_data_path = data_path

# ── Columns ──────────
# 22 feature columns total (confirmed against the actual CSV — one more
# than the paper's 21, since this dataset also includes Snowfall).
# NOTE: verify 'Wind_ccurtailment' matches your CSV header exactly — it
# looks like a typo carried over from the original codebase.
feature_cols = [
    'Electricity', 'Renewable_energy', 'Nuclear', 'Coal', 'Hydro',
    'Geothermal', 'Biomass', 'Solar', 'Solar_curtailment', 'Wind',
    'Wind_ccurtailment', 'Water_pumping', 'Interconnection',
    'Temperature', 'Relative_humidity', 'Precipitation', 'Dew_point',
    'Vapor_pressure', 'Wind_speed', 'Sunshine_duration', 'Snowfall',
    'Global_horizontal_irradiance',
]
target = ['Electricity', 'Renewable_energy', 'Coal']

# ── Task type ──────────
features = 'M'   # multivariate in, multivariate out

# ── Time ────────────
freq = 'h'

# ── Split sizes (rows) ───────────────
# 2018-2021 train (paper Section 3.3), 2022 val, 2023 test.
num_train = 8760 * 2 + 24   # 17,544 hourly rows
num_test = 8760              # one year

# ── Window lengths ─────────────
seq_len = 72     # 3-day lookback
label_len = seq_len
pred_len = 168   # 1-week forecast horizon

# ── Scaling ──────────────
scale = True

# ── Auxiliary forecast input width ────────────────
# Pending confirmation from exp_forecasting.py — carried over from the
# original config, not yet verified as consumed downstream.
forecast_dim = 1