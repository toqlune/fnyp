"""
Loads the electricity dataset and builds train/val/test DataLoaders.

Uses a fixed-window, interval-output split (paper Section 3.3): each
prediction point sees its full lookback window in one forward pass, and
inputs never overlap with the future horizon they're predicting.

Zero-shot cross-region evaluation is supported natively: set
`source_data_file_name` to the region the model was trained on and
`data_file_name` to the region being evaluated. Feature/target scalers are
always fit on `source_data_file_name`'s train split; when both point to the
same file (the default), this is just a standard single-region split.
"""
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.time_features import time_features


class ElectricityDataset(Dataset):
    """One split ('train' / 'val' / 'test') of the electricity dataset."""

    SPLITS = {'train': 0, 'val': 1, 'test': 2}

    def __init__(self, configs, flag):
        assert flag in self.SPLITS, f"flag must be one of {list(self.SPLITS)}"
        self.split = self.SPLITS[flag]

        self.lookback_window_length = configs.lookback_window_length
        self.label_sequence_length = configs.label_sequence_length
        self.prediction_length = configs.prediction_length
        self.forecasting_task_type = configs.forecasting_task_type
        self.target_columns = configs.target_columns
        self.feature_columns = list(configs.feature_columns)
        self.apply_scaling = configs.apply_scaling
        self.time_frequency = configs.time_frequency
        # NOTE: not currently consumed anywhere downstream — trainer.py's
        # model call never passes this tensor in. Kept for now; candidate
        # for removal once architecture/multiattllm.py is reviewed.
        self.auxiliary_forecast_dim = configs.auxiliary_forecast_dim
        self.use_time_features_encoding = (configs.time_embedding_type == 'timeF')
        self.num_train_rows = configs.num_train_rows
        self.num_test_rows = configs.num_test_rows

        self._load(configs.data_root_path, configs.data_file_name, configs.source_data_file_name)

    def __len__(self):
        return len(self.data_x) - self.lookback_window_length - self.prediction_length + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.lookback_window_length
        r_begin = s_end - self.label_sequence_length
        r_end = r_begin + self.label_sequence_length + self.prediction_length

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_forecast = self.data_forecast[r_begin:r_end, :self.auxiliary_forecast_dim]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_forecast

    def inverse_transform(self, data):
        """De-standardizes target-column predictions back to real units."""
        return self.target_scaler.inverse_transform(data)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_csv(data_root_path, file_name):
        full_path = os.path.join(data_root_path, file_name)
        try:
            return pd.read_csv(full_path)
        except UnicodeDecodeError:
            return pd.read_csv(full_path, encoding='SHIFT-JIS')

    def _load(self, data_root_path, data_file_name, source_data_file_name):
        df_eval = self._read_csv(data_root_path, data_file_name)
        df_source = self._read_csv(data_root_path, source_data_file_name)

        # Reorder columns so covariates come first, targets last — the model
        # relies on this ordering to split target vs. covariate features.
        covariate_cols = [c for c in self.feature_columns if c not in self.target_columns]
        ordered_cols = ['date'] + covariate_cols + self.target_columns
        df_eval = df_eval[ordered_cols]
        df_source = df_source[ordered_cols]

        n_val = len(df_eval) - self.num_train_rows - self.num_test_rows
        starts = [0, self.num_train_rows - self.lookback_window_length,
                  len(df_eval) - self.num_test_rows - self.lookback_window_length]
        ends = [self.num_train_rows, self.num_train_rows + n_val, len(df_eval)]
        start, end = starts[self.split], ends[self.split]

        features = df_eval.iloc[:, 1:].values  # everything except 'date'

        if self.apply_scaling:
            train_slice = slice(starts[0], ends[0])
            self.scaler = StandardScaler().fit(df_source.iloc[train_slice, 1:].values)
            self.target_scaler = StandardScaler().fit(
                df_source[self.target_columns].iloc[train_slice].values)
            self.std_ = self.target_scaler.scale_
            features = self.scaler.transform(features)

        n_targets = len(self.target_columns)
        if self.forecasting_task_type == 'S':
            self.data_x = features[start:end, -1:]
            self.data_y = features[start:end, -1:]
        else:
            self.data_x = features[start:end, :len(self.feature_columns)]
            self.data_y = features[start:end, -n_targets:]

        self.data_forecast = features[start:end, :self.auxiliary_forecast_dim]
        self.data_stamp = self._encode_time(df_eval['date'][start:end])

        # Measured off the real data — always correct even if feature_columns
        # in config drifts from what's actually in the CSV. Total channel
        # count fed into the model (target + covariate columns combined).
        self.num_input_channels = self.data_x.shape[-1]

    def _encode_time(self, date_series):
        dates = pd.to_datetime(date_series)
        if not self.use_time_features_encoding:
            stamp = pd.DataFrame({
                'month': dates.dt.month,
                'day': dates.dt.day,
                'weekday': dates.dt.weekday,
                'hour': dates.dt.hour,
            })
            return stamp.values
        return time_features(dates.values, freq=self.time_frequency).transpose(1, 0)


def get_dataloader(configs, flag):
    """Builds the dataset and DataLoader for one split ('train'/'val'/'test')."""
    dataset = ElectricityDataset(configs, flag)
    is_test = flag == 'test'
    loader = DataLoader(
        dataset,
        batch_size=1 if is_test else configs.batch_size,
        shuffle=not is_test,
        num_workers=configs.num_data_loader_workers,
        drop_last=True,
    )
    return dataset, loader