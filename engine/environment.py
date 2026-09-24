"""
One-time environment setup: reproducible seeds and device selection.

Note: PYTORCH_CUDA_ALLOC_CONF must be set before torch is imported anywhere
to take effect, so it lives at the top of main.py, not here.
"""
import os
import random

import numpy as np
import torch


def setup_environment(configs):
    """Fixes random seeds and returns the torch device to train on."""
    random.seed(configs.fix_seed)
    np.random.seed(configs.fix_seed)
    torch.manual_seed(configs.fix_seed)

    if configs.use_gpu and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(configs.gpu)
        device = torch.device(f'cuda:{configs.gpu}')
        print(f'Using GPU: cuda:{configs.gpu}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device