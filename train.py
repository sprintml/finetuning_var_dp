import warnings

warnings.filterwarnings("ignore")

import os, sys, random

sys.path.append("./submodules/VAR")
sys.path.append("./submodules/VAR/models")

slurm_job_id: int = os.getenv("SLURM_JOB_ID")
if slurm_job_id is None:
    slurm_job_id: int = 0


import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
import numpy as np

import torch

from src.utils import dist
from src.trainers import trainers

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if hasattr(cfg.wandb, "disable") and cfg.wandb.disable:
        wandb.init(mode="disabled")
    else:
        if dist.is_local_master():
            wandb.config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            wandb.init(
                project=cfg.wandb.project,
                name=(
                    cfg.wandb.run_name
                    if hasattr(cfg.wandb, "run_name")
                    else f"{cfg.model.name}_{cfg.finetuning.name}_{cfg.trainer.name}_{cfg.dataset.name}_ep:{cfg.trainer.epochs}_lr:{cfg.finetuning.training_params.lr}"
                ),
                job_type="train",
                config={
                    "dataset": cfg.dataset.name,
                    "image_size": cfg.model.image_size,
                    "learning_rate": cfg.finetuning.training_params.lr,
                    "epochs": cfg.trainer.epochs,
                    "seed": cfg.cfg.seed,
                    "SLURM_JOB_ID": slurm_job_id,
                },
            )
    if dist.is_local_master():
        print(OmegaConf.to_yaml(cfg))

    model_cfg = cfg.model
    finetuning_cfg = cfg.finetuning
    dataset_cfg = cfg.dataset
    trainer_cfg = cfg.trainer
    config = cfg.cfg
    with open_dict(model_cfg):
        model_cfg.device = trainer_cfg.device
        model_cfg.seed = config.seed

    action_input = [config, model_cfg, finetuning_cfg, trainer_cfg, dataset_cfg]

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train(*action_input)

    print("fin")


def train(
    config: DictConfig,
    model_cfg: DictConfig,
    finetuning_cfg: DictConfig,
    trainer_cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> None:
    trainer = trainers[trainer_cfg.name][finetuning_cfg.name](
        config, model_cfg, finetuning_cfg, trainer_cfg, dataset_cfg
    )
    trainer.train()


if __name__ == "__main__":
    try:
        dist.initialize()
        if dist.initialized():
            dist.print_dist_info()
        dist.barrier()
        main()
    finally:
        dist.barrier()
        dist.finalize()
