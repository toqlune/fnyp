"""
Training utilities for the MultiAttLLM pipeline: config persistence,
learning-rate scheduling, early stopping, and result plotting.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend('agg')


# ── Config persistence ────────────────────────────────────────────────────

def save_config(config, filepath):
    """Pickles the run's config next to its checkpoints, for reproducibility."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filepath, 'wb') as f:
        pickle.dump(config, f)
    print(f"Saved config to: {filepath}")


def load_config(filepath):
    """Loads a config previously written by save_config."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ── Learning rate schedule ───────────────────────────────────────────────

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True, accelerator=None):
    """Applies one of several hand-tuned learning-rate decay schedules,
    selected by args.lradj."""
    schedules = {
        'type1': lambda: {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))},
        'type2': lambda: {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8},
        'type3': lambda: {epoch: args.learning_rate if epoch < 3
                           else args.learning_rate * (0.9 ** ((epoch - 3) // 1))},
        'PEMS': lambda: {epoch: args.learning_rate * (0.95 ** (epoch // 1))},
        'TST': lambda: {epoch: scheduler.get_last_lr()[0]},
        'constant': lambda: {epoch: args.learning_rate},
    }
    if args.lradj not in schedules:
        raise ValueError(f"Unknown lradj schedule: '{args.lradj}'")

    lr_adjust = schedules[args.lradj]()
    if epoch not in lr_adjust:
        return

    lr = lr_adjust[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if printout:
        message = f'Updating learning rate to {lr}'
        accelerator.print(message) if accelerator is not None else print(message)


# ── Early stopping ───────────────────────────────────────────────────────

class EarlyStopping:
    """Stops training when validation loss stops improving, and saves the
    best model checkpoint seen so far."""

    def __init__(self, patience=7, verbose=False, delta=0, save_best=True, accelerator=None):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_best = save_best
        self.accelerator = accelerator

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            if self.save_best:
                self._save_checkpoint(val_loss, model, path)
        else:
            self.counter += 1
            self._log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, val_loss, model, path):
        # Saving is unconditional — only the log message is gated by
        # `verbose`, so the best model is never silently skipped.
        if self.verbose:
            self._log(f'Validation loss decreased '
                       f'({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint'))
        self.val_loss_min = val_loss

    def _log(self, message):
        self.accelerator.print(message) if self.accelerator is not None else print(message)


# ── Plotting ──────────────────────────────────────────────────────────────
# One shared style for every plot the pipeline produces.
PLOT_DPI = 450
PREDICTION_COLOR = '#e6550d'         # consistent accent color for predictions
DEFAULT_TRUE_COLOR = '#333333'
TARGET_COLORS = {                     # ground-truth color per forecast target
    'Electricity': '#1f77b4',         # blue  — grid demand
    'Renewable_energy': '#2ca02c',    # green — renewable generation
    'Coal': '#6b4226',                # brown — fossil-fuel generation
}

plt.rcParams.update({
    'figure.dpi': PLOT_DPI,
    'savefig.dpi': PLOT_DPI,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def visual(y_true, y_pred=None, target_name=None, save_path='./pic/test.pdf'):
    """
    Plots a forecast target's ground truth against its prediction.

    `target_name` (e.g. 'Electricity', 'Renewable_energy', 'Coal') selects
    the ground-truth color from TARGET_COLORS, so a given variable is drawn
    in the same color across every plot the pipeline produces.
    """
    true_color = TARGET_COLORS.get(target_name, DEFAULT_TRUE_COLOR)
    label = target_name.replace('_', ' ') if target_name else 'Value'

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Ground truth', color=true_color, linewidth=2)
    if y_pred is not None:
        plt.plot(y_pred, label='Prediction', color=PREDICTION_COLOR,
                  linewidth=2, linestyle='--')

    plt.title(f'{label} — forecast vs. actual' if target_name else 'Forecast vs. actual')
    plt.xlabel('Time step (hours)')
    plt.ylabel(label)
    plt.legend(loc='upper left')
    plt.tight_layout()

    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()