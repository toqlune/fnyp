"""
Continuous calendar-position features for hourly data.

The original implementation dispatched across many pandas frequency codes
(yearly, quarterly, weekly, minute, second, ...); this project's data is
always hourly (configs.freq = 'h'), so only the hourly feature set is kept
here. Add the other TimeFeature classes and the frequency-dispatch table
back if the pipeline ever needs to support another frequency.
"""
import numpy as np
import pandas as pd


def time_features(dates, freq='h'):
    """
    Encodes each timestamp's hour-of-day, day-of-week, day-of-month, and
    day-of-year as continuous values in [-0.5, 0.5], stacked as shape
    (4, len(dates)) — matching what TimeFeatureEmbedding expects for
    freq='h' (see modules/embed.py: FREQ_TO_DIM['h'] = 4).
    """
    if freq != 'h':
        raise NotImplementedError(
            f"time_features only supports freq='h' in this trimmed pipeline, got '{freq}'")

    index = pd.DatetimeIndex(dates)
    return np.vstack([
        index.hour / 23.0 - 0.5,
        index.dayofweek / 6.0 - 0.5,
        (index.day - 1) / 30.0 - 0.5,
        (index.dayofyear - 1) / 365.0 - 0.5,
    ])