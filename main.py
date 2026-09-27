"""
Entry point: trains and evaluates MultiAttLLM on the electricity dataset.

Usage:
    python main.py                 # train + test, nothing persisted to disk
    python main.py --save-model    # train + test, and save the trained
                                    # weights under configs.saved_models_dir
"""
import argparse
import os

# Must be set before torch is imported anywhere, so the CUDA allocator
# picks it up from process start.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
# Uncomment if GPT-2 downloads fail on SSL certificate verification
# (seen in some Colab/corporate-network setups) — this disables cert
# checking for those downloads, so only enable it if you hit that error.
# os.environ['CURL_CA_BUNDLE'] = ''

from types import SimpleNamespace

import torch

from configs import data_config, env_config, model_config
from engine.environment import setup_environment
from engine.trainer import RUN_ID_CONFIG_KEYS, Trainer, build_run_id
from utils.logging_utils import print_config_table, print_run_header


def build_config():
    """Merges the three config modules into one namespace, remembering
    which module each attribute came from so it can be displayed grouped
    by section (see utils/logging_utils.py:print_config_table)."""
    configs = SimpleNamespace()
    config_groups = {}
    for group_name, module in (('Data', data_config), ('Model', model_config), ('Environment', env_config)):
        for key, value in vars(module).items():
            if not key.startswith('_'):
                setattr(configs, key, value)
                config_groups[key] = group_name
    configs._config_groups = config_groups
    return configs


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate MultiAttLLM.')
    parser.add_argument(
        '--save-model', action='store_true', dest='save_model',
        help='Persist the trained model to configs.saved_models_dir after '
             'training and testing. Without this flag, no model file is written.')
    return parser.parse_args()


def main():
    configs = build_config()
    configs.save_model = parse_args().save_model
    configs.device = setup_environment(configs)

    print_config_table(configs, groups=configs._config_groups)

    os.makedirs('results', exist_ok=True)

    for run in range(configs.num_runs):
        trainer = Trainer(configs)
        run_id = build_run_id(configs, run)

        if configs.is_training_mode:
            print_run_header('TRAINING MODEL', configs, RUN_ID_CONFIG_KEYS)
            trainer.train(run_id)

        print_run_header('TESTING MODEL', configs, RUN_ID_CONFIG_KEYS)
        _, metrics = trainer.test(run_id, load_checkpoint=not configs.is_training_mode)
        metrics.to_csv(os.path.join('results', f'results_{run_id}.csv'))

        if configs.save_model:
            trainer.save_model(run_id)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()