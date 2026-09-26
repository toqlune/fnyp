"""
One-time environment setup: reproducible seeds, device selection, and
making sure the folders the run depends on exist.

Note: PYTORCH_CUDA_ALLOC_CONF must be set before torch is imported anywhere
to take effect, so it lives at the top of main.py, not here.
"""
import os
import random

import numpy as np
import torch


def setup_environment(configs):
    """Fixes random seeds, ensures required folders exist, and returns the
    torch device to train on."""
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)

    # Created once; the GPT-2 backbone downloads here on the first run and
    # is loaded from this folder on every run after that.
    os.makedirs(configs.gpt2_weights_dir, exist_ok=True)

    if configs.use_gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(configs.gpu_device_id)
        device = torch.device(f'cuda:{configs.gpu_device_id}')
        print(f'Using GPU: cuda:{configs.gpu_device_id}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device