"""
Entry point: trains and evaluates MultiAttLLM on the electricity dataset.

Usage:
    python main.py                       # improved model, train + test, nothing saved to disk
    python main.py -sm                   # also save the trained weights (configs.saved_models_dir)
    python main.py -bm                   # run the paper's base architecture (no M1-M4) instead
    python main.py -ntp 128              # override num_text_prototypes (improved model only)
    python main.py -bm -sm               # flags combine freely

Long forms also work: --save-model, --base-model, --num-text-proto[=N].
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
from engine.trainer import Trainer, build_run_id, run_id_config_keys
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
        '-sm', '--save-model', action='store_true', dest='save_model',
        help='Persist the trained model to configs.saved_models_dir after '
             'training and testing. Without this flag, no model file is written.')
    parser.add_argument(
        '-bm', '--base-model', action='store_true', dest='base_model',
        help="Run the paper's original base architecture instead of the "
             "improved model — no M1 (learnable RevIN affine), M2 (text-"
             "prototype bank + GLU gate), M3 (channel-independent decoder), "
             "or M4 (adaptive gated fusion). Equivalent to setting "
             "use_base_model = True in configs/model_config.py.")
    parser.add_argument(
        '-ntp', '--num-text-proto', type=int, default=None, dest='num_text_proto',
        metavar='N',
        help='Override configs.num_text_prototypes with N. Improved model '
             'only — has no effect when --base-model is set, since base '
             'mode uses word_projection_size instead (a note is printed if '
             'both are given together).')

    args = parser.parse_args()
    if args.num_text_proto is not None and args.num_text_proto <= 0:
        parser.error('--num-text-proto must be a positive integer')
    return args


def main():
    configs = build_config()
    args = parse_args()

    # -bm only turns base mode on; it never turns it off, so setting
    # use_base_model = True directly in model_config.py still works without
    # needing the flag on every invocation.
    if args.base_model:
        configs.use_base_model = True
    configs.save_model = args.save_model
    if args.num_text_proto is not None:
        if configs.use_base_model:
            print('Note: --num-text-proto has no effect in base-model mode '
                  '(it uses word_projection_size instead) — ignoring.')
        else:
            configs.num_text_prototypes = args.num_text_proto

    configs.device = setup_environment(configs)

    # Only one of these applies depending on use_base_model — showing both
    # would be confusing, so hide whichever one this run isn't using.
    irrelevant_key = 'num_text_prototypes' if configs.use_base_model else 'word_projection_size'
    print_config_table(configs, groups=configs._config_groups, exclude=(irrelevant_key,))

    os.makedirs('results', exist_ok=True)

    for run in range(configs.num_runs):
        trainer = Trainer(configs)
        run_id = build_run_id(configs, run)
        highlight_keys = run_id_config_keys(configs)

        if configs.is_training_mode:
            print_run_header('TRAINING MODEL', configs, highlight_keys)
            trainer.train(run_id)

        print_run_header('TESTING MODEL', configs, highlight_keys)
        _, metrics = trainer.test(run_id, load_checkpoint=not configs.is_training_mode)
        metrics.to_csv(os.path.join('results', f'results_{run_id}.csv'))

        if configs.save_model:
            trainer.save_model(run_id)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()