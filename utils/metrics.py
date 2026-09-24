"""
Forecast accuracy metrics for the MultiAttLLM pipeline.

results_evaluation() is the single entry point exp_forecasting.py calls;
everything else here is a helper it composes.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def results_evaluation(y_true, y_pred):
    """
    Computes the full set of metrics reported in the paper for one
    forecast target: MSE, RMSE, NRMSE, MAE, MAPE (outlier-robust), RAE,
    R^2, and the correlation coefficient.

    Returns them as a list, in this fixed order:
    [mse, rmse, nrmse, mae, mape, rae, r2, corr]
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    nrmse = normalized_rmse(y_true, rmse)
    mape = mean_absolute_percentage_error_robust(y_true, y_pred)
    rae = relative_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    corr = correlation_coefficient(y_true, y_pred)

    return [mse, rmse, nrmse, mae, mape, rae, r2, corr]


def correlation_coefficient(y_true, y_pred):
    """Pearson correlation coefficient between true and predicted values."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    true_centered = y_true - y_true.mean()
    pred_centered = y_pred - y_pred.mean()
    numerator = np.sum(true_centered * pred_centered)
    denominator = np.sqrt(np.sum(true_centered ** 2) * np.sum(pred_centered ** 2))
    return numerator / denominator


def normalized_rmse(y_true, rmse):
    """RMSE normalized by the range of the true values, so it's comparable
    across targets with different scales (e.g. Electricity vs. Coal)."""
    y_true = np.asarray(y_true)
    return rmse / (np.max(y_true) - np.min(y_true))


def relative_absolute_error(y_true, y_pred):
    """MAE relative to a naive 'always predict the mean' baseline —
    values below 1.0 mean the model beats that baseline."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    baseline_mae = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / baseline_mae


def mean_absolute_percentage_error_robust(y_true, y_pred, outlier_threshold=2.0, eps=1e-8):
    """
    MAPE with extreme percentage errors excluded, so a handful of points
    near zero (where percentage error explodes) don't dominate the metric.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    percentage_error = np.abs((y_pred - y_true) / (np.abs(y_true) + eps))
    valid = percentage_error < outlier_threshold

    n_outliers = (~valid).sum()
    if n_outliers > 0:
        print(f"MAPE: excluded {n_outliers}/{len(y_true)} outlier points "
              f"(error > {outlier_threshold * 100:.0f}%)")

    if not valid.any():
        return np.nan
    return np.mean(percentage_error[valid])