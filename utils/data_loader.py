"""
Loads the electricity dataset and builds train/val/test DataLoaders.

Uses a fixed-window, interval-output split (paper Section 3.3): each split's
windows never overlap, and each prediction point sees its full lookback
window in one forward pass.

Zero-shot cross-region evaluation is supported natively: set
`source_data_path` to the region the model was trained on and `data_path`
to the region being evaluated. Feature/target scalers are always fit on
`source_data_path`'s train split; when both paths point to the same file
(the default), this is just a standard single-region split.
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

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.features = configs.features
        self.target = configs.target
        self.feature_cols = list(configs.feature_cols)
        self.scale = configs.scale
        self.freq = configs.freq
        self.forecast_dim = configs.forecast_dim
        self.timeenc = 1 if configs.embed == 'timeF' else 0
        self.num_train = configs.num_train
        self.num_test = configs.num_test

        self._load(configs.root_path, configs.data_path, configs.source_data_path)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_forecast = self.data_forecast[r_begin:r_end, :self.forecast_dim]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_forecast

    def inverse_transform(self, data):
        """De-standardizes target-column predictions back to real units."""
        return self.target_scaler.inverse_transform(data)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_csv(root_path, path):
        full_path = os.path.join(root_path, path)
        try:
            return pd.read_csv(full_path)
        except UnicodeDecodeError:
            return pd.read_csv(full_path, encoding='SHIFT-JIS')

    def _load(self, root_path, data_path, source_data_path):
        df_eval = self._read_csv(root_path, data_path)
        df_source = self._read_csv(root_path, source_data_path)

        # Reorder columns so covariates come first, targets last — the model
        # relies on this ordering to split target vs. covariate features.
        covariate_cols = [c for c in self.feature_cols if c not in self.target]
        ordered_cols = ['date'] + covariate_cols + self.target
        df_eval = df_eval[ordered_cols]
        df_source = df_source[ordered_cols]

        n_val = len(df_eval) - self.num_train - self.num_test
        starts = [0, self.num_train - self.seq_len, len(df_eval) - self.num_test - self.seq_len]
        ends = [self.num_train, self.num_train + n_val, len(df_eval)]
        start, end = starts[self.split], ends[self.split]

        features = df_eval.iloc[:, 1:].values  # everything except 'date'

        if self.scale:
            train_slice = slice(starts[0], ends[0])
            self.scaler = StandardScaler().fit(df_source.iloc[train_slice, 1:].values)
            self.target_scaler = StandardScaler().fit(
                df_source[self.target].iloc[train_slice].values)
            self.std_ = self.target_scaler.scale_
            features = self.scaler.transform(features)

        n_targets = len(self.target)
        if self.features == 'S':
            self.data_x = features[start:end, -1:]
            self.data_y = features[start:end, -1:]
        else:
            self.data_x = features[start:end, :len(self.feature_cols)]
            self.data_y = features[start:end, -n_targets:]

        self.data_forecast = features[start:end, :self.forecast_dim]
        self.data_stamp = self._encode_time(df_eval['date'][start:end])

        # Measured off the real data — always correct even if feature_cols
        # in config drifts from what's actually in the CSV.
        self.enc_in = self.data_x.shape[-1]

    def _encode_time(self, date_series):
        dates = pd.to_datetime(date_series)
        if self.timeenc == 0:
            stamp = pd.DataFrame({
                'month': dates.dt.month,
                'day': dates.dt.day,
                'weekday': dates.dt.weekday,
                'hour': dates.dt.hour,
            })
            return stamp.values
        return time_features(dates.values, freq=self.freq).transpose(1, 0)


def get_dataloader(configs, flag):
    """Builds the dataset and DataLoader for one split ('train'/'val'/'test')."""
    dataset = ElectricityDataset(configs, flag)
    is_test = flag == 'test'
    loader = DataLoader(
        dataset,
        batch_size=1 if is_test else configs.batch_size,
        shuffle=not is_test,
        num_workers=configs.num_workers,
        drop_last=True,
    )
    return dataset, loader