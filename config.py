import json
import yaml

def get_config_universal(dataset_name):
    with open(f'./configs/{dataset_name}_config.json') as f:
        config = json.load(f)
    return config


def get_sweep_config_universal(dataset_name):
    with open(f'./configs/sweep_{dataset_name}_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_model_config(model_config):
    with open(f'./configs/{model_config}.json') as f:
        config = json.load(f)
    return config
