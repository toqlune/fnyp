"""
Entry point: trains and evaluates MultiAttLLM on the electricity dataset.
"""
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
from engine.trainer import Trainer, build_run_name


def build_config():
    """Merges the three config modules into one namespace."""
    configs = SimpleNamespace()
    for module in (data_config, model_config, env_config):
        for key, value in vars(module).items():
            if not key.startswith('_'):
                setattr(configs, key, value)
    return configs


def main():
    configs = build_config()
    configs.device = setup_environment(configs)

    for run in range(configs.num_runs):
        trainer = Trainer(configs)
        setting = build_run_name(configs, run)

        if configs.is_training:
            print(f'>>>>>>> training: {setting} >>>>>>>')
            trainer.train(setting)

        print(f'>>>>>>> testing: {setting} >>>>>>>')
        _, metrics = trainer.test(setting, load_checkpoint=not configs.is_training)

        os.makedirs('results', exist_ok=True)
        metrics.to_csv(os.path.join('results', f'metrics_run{run}.csv'))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()