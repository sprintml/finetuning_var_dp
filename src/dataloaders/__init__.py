from src.dataloaders.VAR_loader import get_var_dataloader

from torch.utils.data import DataLoader
from typing import Dict

loaders: Dict[str, DataLoader] = {
    "var": get_var_dataloader,
}
