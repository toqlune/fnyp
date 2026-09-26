"""
Data configuration — electricity demand/generation forecasting (Tokyo).

`source_data_file_name` defaults to the same file as `data_file_name`,
giving a standard single-region train/val/test split. For zero-shot
cross-region evaluation (paper Section 4.6 / Table 8), set
`data_file_name` to the target region's CSV while leaving
`source_data_file_name` pointing at the region the model was trained on —
the loader fits its scalers on `source_data_file_name`'s train split and
evaluates on `data_file_name`.
"""

# ── File locations ─────────────
data_root_path = './data'
data_file_name = 'tokyo.csv'
source_data_file_name = data_file_name

# ── Columns ──────────
# 22 feature columns total (confirmed against the actual CSV — one more
# than the paper's 21, since this dataset also includes Snowfall).
# NOTE: verify 'Wind_ccurtailment' matches your CSV header exactly — it
# looks like a typo carried over from the original codebase.
feature_columns = [
    'Electricity', 'Renewable_energy', 'Nuclear', 'Coal', 'Hydro',
    'Geothermal', 'Biomass', 'Solar', 'Solar_curtailment', 'Wind',
    'Wind_ccurtailment', 'Water_pumping', 'Interconnection',
    'Temperature', 'Relative_humidity', 'Precipitation', 'Dew_point',
    'Vapor_pressure', 'Wind_speed', 'Sunshine_duration', 'Snowfall',
    'Global_horizontal_irradiance',
]
target_columns = ['Electricity', 'Renewable_energy', 'Coal']

# ── Task type ──────────
forecasting_task_type = 'M'   # multivariate in, multivariate out

# ── Time ────────────
time_frequency = 'h'

# ── Split sizes (rows) ───────────────
# 2018-2021 train (paper Section 3.3), 2022 val, 2023 test.
num_train_rows = 8760 * 2 + 24   # 17,544 hourly rows
num_test_rows = 8760              # one year

# ── Window lengths ─────────────
lookback_window_length = 72       # 3-day lookback (renamed from `seq_len`)
label_sequence_length = lookback_window_length  # (renamed from `label_len`)
prediction_length = 168           # 1-week forecast horizon (renamed from `pred_len`)

# ── Scaling ──────────────
apply_scaling = True

# ── Auxiliary forecast input width ────────────────
# Pending confirmation from exp_forecasting.py — carried over from the
# original config, not yet verified as consumed downstream.
auxiliary_forecast_dim = 1