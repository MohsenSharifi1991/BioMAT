import wandb
from config import get_sweep_config
from sweep_main import run_main
import torch
import os

offline = False
if offline:
    os.environ["WANDB_MODE"] = "dryrun"
torch.cuda.empty_cache()
sweep_config = get_sweep_config()
sweep_id = wandb.sweep(sweep_config, project="opensim_kinematic_sweep_v2")
wandb.agent(sweep_id, function=run_main)

# opensim_kinematic_sweep_v2
'''
https://open.gitcode.host/wandb-docs/library/init.html
'''