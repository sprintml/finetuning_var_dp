from src.local_datasets.local_dataset import ImageFolderDataset

from typing import Dict

datasets: Dict[str, type] = {
    "flowers102": ImageFolderDataset,
    "cars196": ImageFolderDataset,
    "pet": ImageFolderDataset,
    "food101": ImageFolderDataset,
    "cub200": ImageFolderDataset,
}
