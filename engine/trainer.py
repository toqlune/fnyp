"""
Trains and evaluates MultiAttLLM. Replaces the old Exp_Basic/Exp_Forecast
split — with a single model and a single task, no base-class abstraction
is needed.
"""
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from architecture.multiattllm import Model
from utils.data_loader import get_dataloader
from utils.metrics import results_evaluation
from utils.tools import EarlyStopping, adjust_learning_rate, save_config, visual

warnings.filterwarnings('ignore')

# Config fields embedded in every generated run identifier (saved-model
# filenames, results filenames, working checkpoint folder). Keep this short
# and intentional — include only whatever actually varies between your
# experiment runs, not every hyperparameter.
RUN_ID_CONFIG_KEYS = ['num_text_prototypes']


def _config_slug(configs):
    parts = [f"{key.replace('_', '-')}-{getattr(configs, key)}" for key in RUN_ID_CONFIG_KEYS]
    return '_'.join(parts)


def build_run_id(configs, run_index):
    """Builds a short, readable identifier for one run:
    <timestamp>_<config-slug>[_run<run_index>]
    e.g. 2026-01-01T01-01-00_num-text-prototypes-64

    Used to name the saved-model file, the results file, and the working
    checkpoint folder for this run.
    """
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    run_id = f"{timestamp}_{_config_slug(configs)}"
    if configs.num_runs > 1:
        run_id += f"_run{run_index}"
    return run_id


class Trainer:

    def __init__(self, configs):
        self.args = configs
        self.device = configs.device

        # enc_in / c_out are measured off the real training data rather
        # than hand-typed, so they can never drift from the actual CSV.
        # The loaded train set is cached and reused when train() runs, so
        # the data is never read from disk twice.
        self._train_data, self._train_loader = get_dataloader(configs, 'train')
        configs.enc_in = self._train_data.enc_in
        configs.c_out = len(configs.target_columns)

        self.model = Model(configs).float().to(self.device)

    def _get_data(self, flag):
        if flag == 'train':
            return self._train_data, self._train_loader
        return get_dataloader(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _select_scheduler(self, optimizer, train_loader):
        if self.args.lr_schedule_type == 'COS':
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
        return lr_scheduler.OneCycleLR(
            optimizer, steps_per_epoch=len(train_loader), pct_start=self.args.onecycle_warmup_fraction,
            epochs=self.args.num_train_epochs, max_lr=self.args.learning_rate)

    # ------------------------------------------------------------------ #

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y, _, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_y_mark)
                target = batch_y[:, -self.args.prediction_length:, :]
                total_loss.append(criterion(outputs, target).item())
        self.model.train()
        return float(np.mean(total_loss))

    def train(self, run_id):
        _, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val' if self.args.use_validation_split else 'test')

        run_checkpoint_dir = os.path.join(self.args.checkpoint_dir, run_id, 'checkpoints')
        os.makedirs(run_checkpoint_dir, exist_ok=True)
        save_config(self.args, os.path.join(run_checkpoint_dir, 'configs.pkl'))

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.early_stopping_patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(optimizer, train_loader)
        scaler = torch.amp.GradScaler('cuda') if self.args.use_mixed_precision else None

        loss_records = {"epoch": [], "time": [], "train_loss": [], "vali_loss": []}
        run_start = time.time()
        time_now = time.time()

        for epoch in range(self.args.num_train_epochs):
            self.model.train()
            epoch_start = time.time()
            iter_count = 0
            train_loss = []

            for i, (batch_x, batch_y, _, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                target = batch_y[:, -self.args.prediction_length:, :]

                if self.args.use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_x, batch_y_mark)
                        loss = criterion(outputs, target)
                else:
                    outputs = self.model(batch_x, batch_y_mark)
                    loss = criterion(outputs, target)
                train_loss.append(loss.item())

                verbose_interval = max(len(train_loader) // 5, 1)
                if (i + 1) % verbose_interval == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.num_train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time / 60:.2f}min")
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if self.args.lr_schedule_type == 'TST':
                    adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            train_loss = float(np.mean(train_loss))
            vali_loss = self.vali(vali_loader, criterion)

            loss_records["epoch"].append(epoch + 1)
            loss_records["time"].append(round((time.time() - run_start) / 60, 4))
            loss_records["train_loss"].append(train_loss)
            loss_records["vali_loss"].append(vali_loss)

            print(f" Epoch: {epoch + 1} cost time: {round((time.time() - epoch_start) / 60, 2)} min")
            print(f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, run_checkpoint_dir)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            epoch_minutes = (time.time() - epoch_start) / 60
            left_time = 1 + (self.args.early_stopping_patience - early_stopping.counter) * epoch_minutes
            print(f"  Left time: {round(left_time, 2)} min")

            if self.args.lr_schedule_type == 'COS':
                scheduler.step()
                print(f"lr = {optimizer.param_groups[0]['lr']:.10f}")
            elif self.args.lr_schedule_type != 'TST':
                adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args, printout=True)
            else:
                print(f"Updating learning rate to {scheduler.get_last_lr()[0]}")

        self.model.load_state_dict(torch.load(os.path.join(run_checkpoint_dir, 'checkpoint')))

        run_dir = os.path.join(self.args.checkpoint_dir, run_id)
        loss_path = os.path.join(run_dir, "loss_records.csv")
        pd.DataFrame(loss_records).to_csv(loss_path, index=False)
        print("Loss records saved to:", loss_path)
        return self.model

    def test(self, run_id, load_checkpoint=False):
        test_data, test_loader = self._get_data('test')
        run_dir = os.path.join(self.args.checkpoint_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        if load_checkpoint:
            print('Loading model checkpoint...')
            checkpoint_path = os.path.join(run_dir, 'checkpoints', 'checkpoint')
            self.model.load_state_dict(torch.load(checkpoint_path))

        preds, trues, iter_times = [], [], []
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, _, batch_y_mark in test_loader:
                start = time.time()
                batch_x = batch_x.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_y_mark)
                iter_times.append(time.time() - start)

                outputs = outputs.detach().cpu().numpy()
                target = batch_y[:, -self.args.prediction_length:, :].numpy()

                if test_data.scale and self.args.apply_inverse_transform:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, shape[-1])).reshape(shape)
                    target = test_data.inverse_transform(target.reshape(-1, shape[-1])).reshape(shape)

                preds.append(outputs)
                trues.append(target)

        print(f"Cost time: {np.mean(iter_times):.4f} s/iter")
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        mse, rmse, nrmse, mae, mape, rae, r2, corr = results_evaluation(trues.flatten(), preds.flatten())
        print(f'mae: {mae}, r2: {r2}')

        with open(os.path.join(self.args.checkpoint_dir, "result_long_term_forecast.txt"), 'a') as f:
            f.write(f"{run_id}\nmae:{mae}, r2:{r2}\n\n")

        tag = self.args.data_file_name[:-4]
        np.save(os.path.join(run_dir, f'pred_{tag}.npy'), preds)
        np.save(os.path.join(run_dir, f'true_{tag}.npy'), trues)

        return self._evaluate_per_target(trues, preds, trainable_params, run_dir)

    def _evaluate_per_target(self, trues, preds, trainable_params, run_dir):
        forecast_stride = self.args.prediction_length
        n_targets = len(self.args.target_columns)
        trues = trues[::forecast_stride].reshape(-1, n_targets)
        preds = preds[::forecast_stride].reshape(-1, n_targets)

        columns, rows = {}, []
        for name in self.args.target_columns:
            idx = self.args.target_columns.index(name)
            y_true, y_pred = trues[:, idx], preds[:, idx]
            columns[f'{name}_true'] = y_true
            columns[f'{name}_pred'] = y_pred

            # last 7 forecast windows, for a readable snapshot
            window = self.args.prediction_length * 7
            visual(y_true[-window:], y_pred[-window:], target_name=name,
                   save_path=os.path.join(run_dir, f'{name}.png'))

            mse, rmse, nrmse, mae, mape, rae, r2, corr = results_evaluation(y_true, y_pred)
            print(f'{name} — mse:{mse}, rmse:{rmse}, mae:{mae}, r2:{r2}, corr:{corr}')
            rows.append([mse, rmse, nrmse, mae, mape, rae, r2, corr])

        metrics_df = pd.DataFrame(
            rows, columns=['mse', 'rmse', 'nrmse', 'mae', 'mape', 'rae', 'r2', 'corr'],
            index=self.args.target_columns)
        metrics_df.insert(0, 'trainable_params', trainable_params)
        metrics_df.loc['mean'] = metrics_df.mean()
        print(metrics_df.loc['mean'])

        pred_df = pd.DataFrame(columns)
        tag = self.args.data_file_name[:-4]
        pred_df.to_csv(os.path.join(run_dir, f'predictions_{tag}.csv'))
        metrics_df.to_csv(os.path.join(run_dir, f'metrics_{tag}.csv'))
        return pred_df, metrics_df

    def save_model(self, run_id):
        """Persists the trained model's weights permanently to
        configs.saved_models_dir. Only called when the --save-model CLI
        flag is set — separate from the working checkpoint written under
        checkpoint_dir during training, which exists purely so the
        best-validation epoch can be reloaded before testing."""
        os.makedirs(self.args.saved_models_dir, exist_ok=True)
        save_path = os.path.join(self.args.saved_models_dir, f'model_{run_id}.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to: {save_path}')
        return save_path