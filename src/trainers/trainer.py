from omegaconf import DictConfig, OmegaConf
import torch
from typing import Tuple, List
from torch import Tensor as T
from torch.utils.data import DataLoader

from src.models import gen_models
from submodules.VAR.models.var import VAR
import os

import logging

logging.disable(logging.INFO)


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        model_cfg: DictConfig,
        finetuning_cfg: DictConfig,
        trainer_cfg: DictConfig,
        dataset_cfg: DictConfig,
    ):
        self.config = config
        self.model_cfg = model_cfg
        self.finetuning_cfg = finetuning_cfg
        self.training_params = finetuning_cfg.training_params
        self.trainer_cfg = trainer_cfg
        self.dataset_cfg = dataset_cfg

        self.model: torch.nn.Module = gen_models[self.model_cfg.name][
            self.finetuning_cfg.name
        ](
            self.model_cfg,
            self.dataset_cfg,
        )

    @property
    def output_dir(self):
        return os.path.join(
            self.config.path_to_models,
            self.model_cfg.name,
            self.finetuning_cfg.name,
            self.trainer_cfg.name,
            self.dataset_cfg.name,
            "" if not hasattr(self.model_cfg, "subdir") else str(self.model_cfg.subdir),
        )

    def step(self, images: T) -> Tuple[T, T]:
        raise NotImplementedError

    def setup_dataset(self) -> None:
        raise NotImplementedError

    def setup_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @property
    def transform(self):
        return None

    def _update(self, *args, **kwargs) -> float:
        raise NotImplementedError
