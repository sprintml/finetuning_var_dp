from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
import os


class ImageFolderDataset:
    def __init__(self, dataset_cfg, transform: Compose) -> ImageFolder:
        self.dataset = ImageFolder(
            os.path.join(dataset_cfg.dataset_path, dataset_cfg.split),
            transform=transform,
        )
        self.collate_fn = None
