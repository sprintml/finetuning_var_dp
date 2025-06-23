import os, torch, random, numpy as np


def set_randomness(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


get_path = (
    lambda x, config, model_cfg, trainer_cfg, finetuning_cfg, dataset_cfg: os.path.join(
        x,
        config.run_id,
        model_cfg.name,
        trainer_cfg.name,
        finetuning_cfg.name,
        dataset_cfg.name,
        dataset_cfg.split,
        str(model_cfg.subdir) if hasattr(model_cfg, "subdir") else "",
        f"ckpt_ep{trainer_cfg.checkpoint}",
    )
)
