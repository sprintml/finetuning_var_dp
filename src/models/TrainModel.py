from torch import nn
from omegaconf import DictConfig


class TrainVARWrapper(nn.Module):
    r"""
    Exists only to make sure VAR is not initialized twice.
    The models are initialized in their respective trainer classes.
    """

    def __init__(self, model_cfg: DictConfig, dataset_cfg: DictConfig):

        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
