import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, transforms


def var_default_augmentation(
    x: torch.Tensor,
    final_reso: int,
    mid_reso_factor: float = 1.125,
    hflip: bool = True,
) -> torch.Tensor:
    """
    Default VAR training augmentation.
    Works for a single image (3-D) or a batch (4-D).
    """

    squeeze = False
    if x.ndim == 3:  # single image: C×H×W
        x = x.unsqueeze(0)  # → 1×C×H×W
        squeeze = True

    B, C, H, W = x.shape
    device = x.device

    # optional horizontal flip
    if hflip and random.random() < 0.5:
        x = torch.flip(x, dims=(-1,))

    # resize so shorter edge = mid_reso
    mid_reso = round(mid_reso_factor * final_reso)
    if H < W:
        new_h, new_w = mid_reso, int(W * mid_reso / H)
    else:
        new_h, new_w = int(H * mid_reso / W), mid_reso

    x_resized = torch.empty(B, C, new_h, new_w, device=device)
    for b in range(B):
        x_resized[b] = TF.resize(
            x[b],
            [new_h, new_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    # random crop to final_reso × final_reso
    top, left, h, w = transforms.RandomCrop.get_params(
        x_resized, output_size=(final_reso, final_reso)
    )
    x_cropped = TF.crop(x_resized, top, left, h, w)  # B×C×256×256

    return x_cropped.squeeze(0) if squeeze else x_cropped


def force_resize_256(img: torch.Tensor):
    """Strict 256 x 256 tensor for the model input - aspect ratio is sacrificed.
    The original raw image is still kept separately for augmentation views.
    """
    return TF.resize(img, (256, 256), interpolation=InterpolationMode.BICUBIC)


def twin_collate(batch):
    imgs, labels = zip(*batch)
    img_256 = torch.stack([force_resize_256(i) for i in imgs], dim=0)
    labels = torch.tensor(labels)
    return img_256, imgs, labels  # imgs is the raw list
