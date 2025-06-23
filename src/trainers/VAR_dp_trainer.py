from omegaconf import DictConfig, open_dict

import wandb
import numpy as np
import os, time, gc
import os.path as osp

from tqdm.auto import tqdm
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms

from peft import LoraConfig, LNTuningConfig, get_peft_model

from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from src.utils import dist
from src.models import gen_models
from src.trainers.utils import NullDDP
from src.local_datasets import datasets
from src.trainers.VAR_finetuning import FFTVARTrainer
from src.utils.dp_basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from src.utils.dp_var import VAR, VQVAE, VectorQuantizer2, build_vae_var
from src.utils.dp_var_train_utils import twin_collate, var_default_augmentation
from src.local_datasets.utils import (
    pil_loader,
    print_aug,
    normalize_01_into_pm1,
)

from submodules.VAR.models.vqvae import VQVAE, VectorQuantizer2

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


def forward(
    model: VAR, concat_tensor: torch.Tensor
) -> torch.Tensor:  # returns logits_BLV
    """
    :param label_B: label_B
    :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
    :return: logits BLV, V is vocab_size
    """

    # unpack tensor
    label_B_exp = concat_tensor[:, :, :1]
    label_B_exp_temp = label_B_exp[:, :1]
    label_B_exp1 = label_B_exp_temp.squeeze(1).squeeze(1)

    label_B = label_B_exp1.long()
    x_BLCv_wo_first_l = concat_tensor[:, :, 1:]
    bg, ed = 0, model.L
    B = x_BLCv_wo_first_l.shape[0]
    with torch.amp.autocast("cuda", enabled=False):
        sos = cond_BD = model.class_emb(label_B)
        sos = sos.unsqueeze(1).expand(B, model.first_l, -1) + model.pos_start.expand(
            B, model.first_l, -1
        )

        x_BLC = torch.cat((sos, model.word_embed(x_BLCv_wo_first_l.float())), dim=1)
        x_BLC += (
            model.lvl_embed(model.lvl_1L[:, :ed].expand(B, -1)) + model.pos_1LC[:, :ed]
        )  # lvl: BLC;  pos: 1LC

    cond_BD_or_gss = model.shared_ada_lin(cond_BD)

    # hack: get the dtype if mixed precision is used
    temp = x_BLC.new_ones(8, 8)
    main_type = torch.matmul(temp, temp).dtype

    x_BLC = x_BLC.to(dtype=main_type)
    cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)

    attn_bias = model.attn_bias_for_masking[:, :, :ed, :ed]

    AdaLNSelfAttn.forward

    for i, b in enumerate(model.blocks):
        x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
    x_BLC = model.get_logits(x_BLC.float(), cond_BD)

    if model.prog_si == 0:
        if isinstance(model.word_embed, nn.Linear):
            x_BLC[0, 0, 0] += (
                model.word_embed.weight[0, 0] * 0 + model.word_embed.bias[0] * 0
            )
        else:
            s = 0
            for p in model.word_embed.parameters():
                if p.requires_grad:
                    s += p.view(-1)[0] * 0
            x_BLC[0, 0, 0] += s
    return x_BLC  # logits BLV, V is vocab_size


class DPVARTrainer:
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

        self.model: VAR = gen_models[self.model_cfg.name][self.finetuning_cfg.name](
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

    def load_models(self):
        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt, var_ckpt = (
            "vae_ch160v4096z32.pth",
            f"var_d{self.model_cfg.model_depth}.pth",
        )
        if dist.is_local_master():
            if not os.path.exists(f"out/model_checkpoints/var/{vae_ckpt}"):
                os.system(
                    f"wget {hf_home}/{vae_ckpt} -O model_checkpoints/var/{vae_ckpt}"
                )
            if not os.path.exists(f"out/model_checkpoints/var/{var_ckpt}"):
                os.system(
                    f"wget {hf_home}/{var_ckpt} -O model_checkpoints/var/{var_ckpt}"
                )
        dist.barrier()

        self.vae_local.load_state_dict(
            torch.load(f"out/model_checkpoints/var/{vae_ckpt}", map_location="cpu")
        )

        var_temp = torch.load(
            f"out/model_checkpoints/var/{var_ckpt}", map_location="cpu"
        )

        # Remove the classification weights from the var_ckpt so, PyTorch won't try to load them
        if "class_emb.weight" in var_temp:
            print("Model will be reinitialize new weights for the classifier.")
            del var_temp["class_emb.weight"]
        if "class_emb.bias" in var_temp:
            del var_temp["class_emb.bias"]

        missing_keys, unexpected_keys = self.var_wo_ddp.load_state_dict(
            var_temp, strict=False
        )

        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print("Following keys are missing or unexpected in the loaded model:")
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

        for p in self.vae_local.parameters():
            p.requires_grad = False

        return self.vae_local, self.var_wo_ddp

    def setup_params(self):
        raise NotImplementedError

    def debug_print_all_params(self):

        if dist.is_local_master():
            print("=" * 50)
            print("Printing all parameters.")
            print("=" * 50)
            for name, param in self.var_wo_ddp.named_parameters():
                if param.requires_grad:
                    print(name, param.size())
            print("=" * 50)
            print("=" * 50)

    def setup_dataset_params(
        self,
    ) -> None:
        self.dataset_size = len(
            datasets[self.dataset_cfg.name](self.dataset_cfg, lambda x: x).dataset
        )
        self.update_batch_size = int(
            self.dataset_size * self.trainer_cfg.q // dist.get_world_size()
        )  # logical batch size
        with open_dict(self.trainer_cfg):
            self.model_cfg.batch_size = (
                self.update_batch_size
            )  # override for the opacus dataloader

    def setup_opacus(self):

        self.target_delta = 1 / self.dataset_size
        print(f"[*] Target delta: {self.target_delta:.2e}")

        self.noise_multiplier = get_noise_multiplier(
            target_epsilon=self.trainer_cfg.target_epsilon,
            target_delta=self.target_delta,  # 1 data point in the dataset
            sample_rate=self.trainer_cfg.q,
            epochs=self.trainer_cfg.epochs,
            accountant=self.trainer_cfg.accountant,
        )

        self.privacy_engine = PrivacyEngine(accountant=self.trainer_cfg.accountant)

        self.var, self.optimizer, self.data_loader = self.privacy_engine.make_private(
            module=self.var,
            optimizer=self.optimizer,
            data_loader=self.ld_train,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.trainer_cfg.max_grad_norm,
            poisson_sampling=self.trainer_cfg.poisson_sampling,
        )

        k = self.k
        if k > 1 and self.trainer_cfg.poisson_sampling:
            for m in self.var.modules():
                if hasattr(m, "grad_accumulation_allowed"):
                    m.grad_accumulation_allowed = True

    def setup_model(self, *args, **kwargs) -> Tuple[VQVAE, VAR]:

        vae_local, var_wo_ddp = build_vae_var(
            V=self.model_cfg.V,
            Cvae=self.model_cfg.Cvae,
            ch=self.model_cfg.ch,
            num_classes=self.dataset_cfg.num_classes,
            share_quant_resi=self.model_cfg.share_quant_resi,
            device=dist.get_device(),
            patch_nums=self.model_cfg.patch_nums,
            depth=self.model_cfg.model_depth,
            shared_aln=self.model_cfg.shared_aln,
        )

        self.vae_local: VQVAE = vae_local
        self.var_wo_ddp: VAR = var_wo_ddp

        self.load_models()
        dist.barrier()

        return self.vae_local, self.var_wo_ddp

    def build_optimizer(self) -> None:

        opt_clz = {
            "adam": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=True),
            "adamw": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=True),
        }[self.training_params.opt.lower().strip()]

        opt_kw = dict(
            lr=self.training_params.lr,
            weight_decay=self.training_params.weight_decay,
        )

        trainable_params = [p for p in self.var_wo_ddp.parameters() if p.requires_grad]

        print(f"[*] Number of trainable parameters: {len(trainable_params)}")

        self.optimizer = opt_clz(
            params=trainable_params,
            **opt_kw,
        )
        return self.optimizer

    def build_var_dataset(
        self,
        data_path: str,
        final_reso: int,
        hflip=False,
        mid_reso=1.125,
    ):
        mid_reso = round(
            mid_reso * final_reso
        )  # Example: mid_reso = round(1.125 * 256) = 288
        train_aug, val_aug = (
            [
                transforms.Resize(
                    mid_reso, interpolation=InterpolationMode.LANCZOS
                ),  # transforms.Resize: resize the shorter edge to mid_reso
                transforms.RandomCrop((final_reso, final_reso)),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ],
            [
                transforms.Resize(
                    mid_reso, interpolation=InterpolationMode.LANCZOS
                ),  # transforms.Resize: resize the shorter edge to mid_reso
                transforms.CenterCrop((final_reso, final_reso)),
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ],
        )

        base_aug = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize_01_into_pm1,
            ]
        )

        if hflip:
            train_aug.insert(0, transforms.RandomHorizontalFlip())
        train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)

        train_set = DatasetFolder(
            root=osp.join(data_path, "train"),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=(
                base_aug if self.trainer_cfg.default_var_augmentation else train_aug
            ),
        )
        try:
            val_set = DatasetFolder(
                root=osp.join(data_path, "val"),
                loader=pil_loader,
                extensions=IMG_EXTENSIONS,
                transform=val_aug,
            )
        except Exception as e:
            print(
                f"Looking for 'val' folder failed, using 'test' folder instead. Dataset {e=}, {data_path=}"
            )
            val_set = DatasetFolder(
                root=osp.join(data_path, "test"),
                loader=pil_loader,
                extensions=IMG_EXTENSIONS,
                transform=val_aug,
            )

        num_classes = self.dataset_cfg.num_classes
        print(f"[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}")
        print_aug(
            base_aug if self.trainer_cfg.default_var_augmentation else train_aug,
            "[train]",
        )
        print_aug(val_aug, "[val]")

        return num_classes, train_set, val_set

    def build_dataset(self) -> None:

        self.dataset = datasets[self.dataset_cfg.name](
            self.dataset_cfg, lambda x: x
        ).dataset

        self.num_classes = self.dataset_cfg.num_classes
        self.dataset_path = self.dataset_cfg.dataset_path
        self.final_reso = self.model_cfg.image_size
        self.hflip = self.model_cfg.hflip
        self.mid_reso = self.model_cfg.mid_reso

        self.num_classes, self.dataset_train, self.dataset_val = self.build_var_dataset(
            data_path=self.dataset_cfg.dataset_path,
            final_reso=self.final_reso,
            hflip=self.hflip,
            mid_reso=self.mid_reso,
        )

    def setup(self) -> None:

        dist.barrier()
        self.setup_model()

        self.setup_params()
        self.vae_local = self.vae_local.to(dist.get_device())
        self.var_wo_ddp = self.var_wo_ddp.to(dist.get_device())

        self.var_wo_ddp.forward = partial(
            forward, self.var_wo_ddp
        )  # Override the VAR default forward function because Opacus doesn't support multi-value inputs
        self.prog_si = -1  # NO PROGRESSIVE TRAINING
        self.max_physical_batch_size = (
            4  # Avoid OOM by setting this to a small value and then handle BS with `q`.
        )
        self.k = self.trainer_cfg.augmentation_multiplicity_k
        if self.k > 1:
            with open_dict(self.trainer_cfg):
                self.trainer_cfg.default_var_augmentation = True
        else:
            with open_dict(self.trainer_cfg):
                self.trainer_cfg.default_var_augmentation = False

        dist.barrier()

        self.var: DPDDP = (DPDDP if dist.initialized() else NullDDP)(
            self.var_wo_ddp,
        )

        dist.barrier()

        self.build_optimizer()
        self.setup_dataset_params()
        self.build_dataset()
        self.setup_dataloader()
        self.label_smooth = self.training_params.label_smooth
        self.quantize_local = self.vae_local.quantize
        self.quantize_local: VectorQuantizer2

        if type(self.var_wo_ddp).__name__ != "PeftModel":
            del self.var_wo_ddp.rng
            self.var_wo_ddp.rng = torch.Generator(device=dist.get_device())
        else:
            try:
                del self.var_wo_ddp.base_model.rng
            except:
                pass
            finally:
                self.var_wo_ddp.base_model.rng = torch.Generator(
                    device=dist.get_device()
                )

        self.train_loss = nn.CrossEntropyLoss(
            label_smoothing=self.label_smooth, reduction="none"
        )
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction="mean")

        self.patch_nums = self.model_cfg.patch_nums
        self.L = sum(pn * pn for pn in self.patch_nums)
        self.last_l = self.patch_nums[-1] * self.patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=dist.get_device()) / self.L
        self.begin_ends = []

        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.setup_opacus()

    def setup_dataloader(self) -> None:

        self.num_workers = self.config.dataloader_num_workers
        self.global_batch_size = (
            self.model_cfg.batch_size * dist.get_world_size()
            if dist.initialized()
            else self.model_cfg.batch_size
        )

        def get_different_generator_for_each_rank() -> Optional[torch.Generator]:
            if self.config.seed is None:
                return None
            g = torch.Generator()
            g.manual_seed(self.config.seed * dist.get_world_size() + dist.get_rank())
            return g

        self.ld_val = DataLoader(
            self.dataset_val,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_size=min(max(1, (self.model_cfg.batch_size // 4)), 32),
            shuffle=False,
            drop_last=False,
        )

        self.ld_train = DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            generator=get_different_generator_for_each_rank(),
            batch_size=self.global_batch_size,
            shuffle=True,
            collate_fn=(
                twin_collate if self.trainer_cfg.default_var_augmentation else None
            ),
            drop_last=False,
        )

        if dist.is_local_master():
            print("=" * 50)
            print(f"[*] Global Batch Size: {self.global_batch_size}")
            print(
                f"[*] Batch Size Per GPU: {(self.global_batch_size) // dist.get_world_size()}"
            )
            print(
                f"[*] Max Physical Batch Size: {self.max_physical_batch_size * dist.get_world_size()}"
            )
            print(
                f"[*] Validation Batch Size: {min((self.model_cfg.batch_size // 4), 32)}"
            )
            logical_batch_size = self.update_batch_size * dist.get_world_size()
            self.steps_per_epoch = int(
                np.ceil(self.dataset_size / logical_batch_size)
                * np.ceil(
                    logical_batch_size
                    / (self.max_physical_batch_size * dist.get_world_size())
                )
            )
            self.total_steps = self.steps_per_epoch * self.trainer_cfg.epochs

            print(f"[*] Approx. Steps per epoch: {self.steps_per_epoch}")
            print(f"[*] Approx. Total steps: {self.total_steps}")
            print("=" * 50)

        del self.dataset_train
        del self.dataset_val

        return self.ld_train, self.ld_val

    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0

        for _, param in model.named_parameters():
            num_params = param.numel()
            # Handle DeepSpeed Zero 3: ds_numel if param.numel() == 0
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # For 4bit layers (bitsandbytes), double reported param count
            if param.__class__.__name__ == "Params4bit":
                num_params *= 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        percent_trainable = (trainable_params / all_param * 100) if all_param else 0.0

        if dist.is_local_master():
            print(
                f"trainable params: {trainable_params:,d} || "
                f"all params: {all_param:,d} || "
                f"trainable%: {percent_trainable:.4f}"
            )

    @torch.no_grad()
    def eval_ep(self, ld_val):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training

        if dist.is_local_master():
            print("\n[*] Moving to evaluation...", flush=True)

        self.var_wo_ddp.eval()

        eval_pbar = tqdm(
            ld_val,
            desc="Evaluating",
            disable=not dist.is_local_master(),
            dynamic_ncols=True,
            leave=False,
        )

        for inp_B3HW, label_B in eval_pbar:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            self.var_wo_ddp.forward
            req_size_at_1 = x_BLCv_wo_first_l.size(1)
            req_size_at_2 = x_BLCv_wo_first_l.size(2)
            # print("Actual shape of b : ", label_B)
            label_B_expanded = (
                label_B.unsqueeze(1).unsqueeze(2).expand(-1, req_size_at_1, -1)
            )

            """
            Concatenate the label_B_expanded and x_BLCv_wo_first_l
            along the last dimension to pass a everything in a single
            tensor to the forward function of the model. Later, they
            are unpacked into label_B and x_BLCv_wo_first_l.
            """
            concat_tensor = torch.cat((label_B_expanded, x_BLCv_wo_first_l), dim=2)
            logits_BLV = self.var_wo_ddp(concat_tensor)

            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += (
                self.val_loss(
                    logits_BLV.data[:, -self.last_l :].reshape(-1, V),
                    gt_BL[:, -self.last_l :].reshape(-1),
                )
                * B
            )
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (
                100 / gt_BL.shape[1]
            )
            acc_tail += (
                logits_BLV.data[:, -self.last_l :].argmax(dim=-1)
                == gt_BL[:, -self.last_l :]
            ).sum() * (100 / self.last_l)
            tot += B

            if dist.is_local_master():
                eval_pbar.set_postfix(
                    {
                        "Loss": f"{(L_mean/tot):.4f}",
                        "Acc": f"{(acc_mean/tot):.2f}%",
                    }
                )

        self.var_wo_ddp.train(training)

        stats = L_mean.new_tensor(
            [L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot]
        )
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()

        if dist.is_local_master():
            wandb.log(
                {
                    "val/loss_mean": L_mean,
                    "val/loss_tail": L_tail,
                    "val/acc_mean": acc_mean,
                    "val/acc_tail": acc_tail,
                }
            )

        if dist.is_local_master():
            print(f"[*] Evaluation completed in {time.time() - stt:.2f}s\n", flush=True)

        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - stt

    def train_one_ep(
        self,
        ep: int,
        is_first_ep: bool,
        data_loader,
    ):
        """Train for one epoch"""

        step_cnt = 0
        stats = {
            "Lm": 0.0,  # Mean loss
            "Lt": 0.0,  # Tail loss
            "Accm": 0.0,  # Mean accuracy
            "Acct": 0.0,  # Tail accuracy
            "tnm": 0.0,  # Gradient norm
        }

        start_time = time.time()

        import warnings

        if is_first_ep:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

        running_loss = 0.0
        running_acc = 0.0
        running_loss_tail = 0.0
        running_acc_tail = 0.0

        steps_this_epoch = 0

        with BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=self.max_physical_batch_size,
            optimizer=self.optimizer,
        ) as memory_safe_data_loader:

            if self.trainer_cfg.default_var_augmentation and self.k > 1:

                for it, (inp256, raw_imgs, label) in enumerate(memory_safe_data_loader):

                    # DEBUG ───────────────
                    # current_epsilon = self.privacy_engine.get_epsilon(self.target_delta)
                    # if dist.is_local_master() and it == 0:
                    #     print(f"[DBG-eps] Start ep{ep+1}: ε={current_epsilon:.2f}")
                    # ─────────────────────

                    steps_this_epoch += 1
                    current_epsilon = self.privacy_engine.get_epsilon(self.target_delta)
                    if current_epsilon >= self.trainer_cfg.target_epsilon:
                        break
                    inp256 = inp256.to(dist.get_device(), non_blocking=True)
                    label = label.to(dist.get_device(), non_blocking=True)

                    self.cur_it = f"{it+1}"

                    stepping = True
                    step_cnt += int(stepping)

                    (
                        clipped_grad_norm,
                        loss_val,
                        acc_val,
                        loss_tail_val,
                        acc_tail_val,
                    ) = self.train_step_with_k(
                        stepping=stepping,
                        imgs_256=inp256,
                        imgs_raw=raw_imgs,
                        label_B=label,
                    )

                    running_loss += loss_val
                    running_acc += acc_val
                    running_loss_tail += loss_tail_val
                    running_acc_tail += acc_tail_val

                    # Print progress every 100 iterations
                    if (it + 1) % 100 == 0 and dist.is_local_master():
                        elapsed = time.time() - start_time
                        steps_per_sec = (it + 1) / elapsed

                        print(
                            f"\n[Epoch {ep} Progress] "
                            f"{it+1} steps | "
                            f"Steps/sec: {steps_per_sec:.2f} | "
                        )

            elif not self.trainer_cfg.default_var_augmentation and self.k == 1:

                for it, (inp, label) in enumerate(memory_safe_data_loader):

                    # DEBUG ───────────────
                    # current_epsilon = self.privacy_engine.get_epsilon(self.target_delta)
                    # if dist.is_local_master() and it == 0:
                    #     print(f"[DBG-eps] Start ep{ep+1}: ε={current_epsilon:.2f}")
                    # ─────────────────────

                    steps_this_epoch += 1
                    current_epsilon = self.privacy_engine.get_epsilon(self.target_delta)
                    if current_epsilon >= self.trainer_cfg.target_epsilon:
                        break
                    inp = inp.to(dist.get_device(), non_blocking=True)
                    label = label.to(dist.get_device(), non_blocking=True)

                    self.cur_it = f"{it+1}"

                    stepping = True
                    step_cnt += int(stepping)

                    (
                        clipped_grad_norm,
                        loss_val,
                        acc_val,
                        loss_tail_val,
                        acc_tail_val,
                    ) = self.train_step(
                        stepping=stepping,
                        inp_B3HW=inp,
                        label_B=label,
                    )

                    running_loss += loss_val
                    running_acc += acc_val
                    running_loss_tail += loss_tail_val
                    running_acc_tail += acc_tail_val

                    # Print progress every 100 iterations
                    if (it + 1) % 100 == 0 and dist.is_local_master():
                        elapsed = time.time() - start_time
                        steps_per_sec = (it + 1) / elapsed

                        print(
                            f"\n[Epoch {ep} Progress] "
                            f"{it+1} steps | "
                            f"Steps/sec: {steps_per_sec:.2f} | "
                        )

            else:
                raise ValueError(
                    "Invalid configuration: "
                    "Either default_var_augmentation should be True and k should be greater than 1."
                    "Or default_var_augmentation should be False and k should be equal to 1."
                    f"Current Flags: aug={self.trainer_cfg.default_var_augmentation=}, k={self.k=}"
                )

        # Calculate epoch statistics
        stats = {
            "Lm": running_loss / steps_this_epoch,
            "Lt": running_loss_tail / steps_this_epoch,
            "Accm": running_acc / steps_this_epoch,
            "Acct": running_acc_tail / steps_this_epoch,
            "tnm": clipped_grad_norm,
        }

        elapsed = time.time() - start_time
        remain_time = f"{(elapsed * (self.trainer_cfg.epochs - ep - 1)) / 60:.1f}min"
        finish_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(time.time() + elapsed * (self.trainer_cfg.epochs - ep - 1)),
        )

        if dist.is_local_master():
            wandb.log(
                {
                    "train_epoch/epoch_loss_mean": stats["Lm"],
                    "train_epoch/epoch_loss_tail": stats["Lt"],
                    "train_epoch/epoch_acc_mean": stats["Accm"],
                    "train_epoch/epoch_acc_tail": stats["Acct"],
                    "train_epoch/epoch_grad_norm": stats["tnm"],
                    "epoch": (ep + 1),
                }
            )

        return stats, (elapsed, remain_time, finish_time)

    def train_step(
        self,
        stepping: bool,
        inp_B3HW: FTen,
        label_B: Union[ITen, FTen],
    ) -> Tuple[
        Optional[Union[Ten, float]], Optional[float], float, float, float, float
    ]:

        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

        # Moved randomness out of the forward step
        label_B = torch.where(
            torch.rand(B, device=label_B.device) < self.var_wo_ddp.cond_drop_rate,
            self.var_wo_ddp.num_classes,
            label_B,
        )

        # Forward pass
        req_size_at_1, req_size_at_2 = x_BLCv_wo_first_l.size(
            1
        ), x_BLCv_wo_first_l.size(2)

        label_B_expanded = (
            label_B.unsqueeze(1).unsqueeze(2).expand(-1, req_size_at_1, -1)
        )

        # Concatenate the label_B_expanded and x_BLCv_wo_first_l
        concat_tensor = torch.cat((label_B_expanded, x_BLCv_wo_first_l), dim=2)
        logits_BLV = self.var(concat_tensor)

        loss = (
            self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1))
            .view(B, -1)
            .mul(self.loss_weight)
            .sum(dim=-1)
            .mean()
        )

        # Backward pass
        loss.backward()

        # Calculate accuracy and tail loss
        with torch.no_grad():
            predictions = logits_BLV.argmax(dim=-1)
            accuracy = (predictions == gt_BL).float().mean() * 100
            tail_logits = logits_BLV[:, -self.last_l :]
            tail_targets = gt_BL[:, -self.last_l :]
            loss_tail = self.val_loss(
                tail_logits.reshape(-1, V), tail_targets.reshape(-1)
            )
            acc_tail = (tail_logits.argmax(dim=-1) == tail_targets).float().mean() * 100

        parameters = [
            p for p in self.optimizer.param_groups[0]["params"] if p.grad is not None
        ]
        grad_norm = torch.norm(
            torch.stack([p.grad.norm() for p in parameters if p.grad is not None]),
        )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        noise_std = self.trainer_cfg.max_grad_norm * self.noise_multiplier
        signal_to_noise_ratio = grad_norm / noise_std if noise_std > 0 else float("inf")
        effective_batch_size = inp_B3HW.shape[0]
        estimated_noise_impact = (
            noise_std / (grad_norm * np.sqrt(effective_batch_size))
            if grad_norm > 0
            else float("inf")
        )

        current_epsilon = self.privacy_engine.get_epsilon(self.target_delta)

        if dist.is_local_master():
            if stepping:
                print(
                    f"[Step {self.cur_it}] "
                    f"Privacy epsilon: {current_epsilon:.2f} | "
                    f"Num samples this step: {inp_B3HW.shape[0]} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Accuracy: {accuracy.item():.2f}% | "
                    f"Grad norm: {grad_norm:.4f} | "
                    f"SNR: {signal_to_noise_ratio:.4f} | "
                    f"Noise std: {noise_std:.4f} | "
                    f"Noise impact: {estimated_noise_impact:.4f}"
                )
            wandb.log(
                {
                    "train_step/loss": loss.item(),
                    "train_step/train_acc": accuracy.item(),
                    "train_step/tail_loss": loss_tail.item(),
                    "train_step/tail_acc": acc_tail.item(),
                    "train_step/grad_norm": grad_norm,
                    "train_step/privacy_epsilon": (
                        current_epsilon if current_epsilon is not None else 0.0
                    ),
                    "dp_metrics/noise_std": noise_std,
                    "dp_metrics/signal_to_noise_ratio": signal_to_noise_ratio,
                    "dp_metrics/estimated_noise_impact": estimated_noise_impact,
                }
            )

        return (
            grad_norm,
            loss.item(),
            accuracy.item(),
            loss_tail.item(),
            acc_tail.item(),
        )

    def train_step_with_k(
        self,
        stepping: bool,
        imgs_256: torch.Tensor,  # B × 3 × 256 × 256
        imgs_raw,  # Tuple of length B, each C × H × W
        label_B: Union[ITen, FTen],
    ):
        """
        Implements k-augmentation multiplicity when k>1.
        """
        device = dist.get_device()
        k = self.k if stepping else 1  # safety: only loop when stepping
        loss_sum = acc_sum = loss_tail_sum = acc_tail_sum = 0.0

        for _ in range(k):

            # print(f"[DEBUG] [Step {self.cur_it}] k-aug iteration {_+1}/{k}")

            batch_aug = torch.stack(
                [
                    var_default_augmentation(
                        r.to(device),
                        final_reso=self.final_reso,
                        mid_reso_factor=self.mid_reso,
                        hflip=self.hflip,
                    )
                    for r in imgs_raw
                ],
                dim=0,
            )

            B, V = label_B.shape[0], self.vae_local.vocab_size

            self.var.require_backward_grad_sync = False  # sync once per outer step
            gt_idx_Bl = self.vae_local.img_to_idxBl(batch_aug)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            # Moved randomness out of the forward step
            label_B = torch.where(
                torch.rand(B, device=label_B.device) < self.var_wo_ddp.cond_drop_rate,
                self.var_wo_ddp.num_classes,
                label_B,
            )

            req1, req2 = x_BLCv_wo_first_l.size(1), x_BLCv_wo_first_l.size(2)
            label_pack = label_B.unsqueeze(1).unsqueeze(2).expand(-1, req1, -1)
            concat = torch.cat((label_pack, x_BLCv_wo_first_l), dim=2)

            logits = self.var(concat)

            loss = (
                self.train_loss(logits.view(-1, V), gt_BL.view(-1))
                .view(B, -1)
                .mul(self.loss_weight)
                .sum(dim=-1)
                .mean()
            )
            loss.backward()

            with torch.no_grad():
                acc = (logits.argmax(-1) == gt_BL).float().mean() * 100
                tail_logits = logits[:, -self.last_l :]
                tail_targets = gt_BL[:, -self.last_l :]
                loss_tail = self.val_loss(
                    tail_logits.reshape(-1, V), tail_targets.reshape(-1)
                )
                acc_tail = (tail_logits.argmax(-1) == tail_targets).float().mean() * 100

            # ---- accumulate k-views gradients ----
            for p in self.optimizer.params:
                if isinstance(p.grad_sample, list):
                    p.grad_sample = torch.stack(p.grad_sample, dim=0).sum(dim=0)

            loss_sum += loss.item()
            acc_sum += acc.item()
            loss_tail_sum += loss_tail.item()
            acc_tail_sum += acc_tail.item()

        # ---- average gradients over k views ----
        for p in self.optimizer.params:
            if hasattr(p, "grad_sample") and isinstance(p.grad_sample, torch.Tensor):
                p.grad_sample /= k

        # ---- optimiser step & manual grad-norm logging ----
        if self.trainer_cfg.max_grad_norm > 0:
            total_norm = torch.norm(
                torch.stack(
                    [p.grad.norm() for p in self.optimizer.params if p.grad is not None]
                )
            ).item()  # Opacus already clips per-sample grads; norm here is just for logs
        else:
            total_norm = None

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        avg_loss, avg_acc = loss_sum / k, acc_sum / k
        avg_loss_tail, avg_acc_tail = loss_tail_sum / k, acc_tail_sum / k

        noise_std = self.trainer_cfg.max_grad_norm * self.noise_multiplier
        signal_to_noise_ratio = (
            total_norm / noise_std if noise_std > 0 else float("inf")
        )

        effective_batch_size = imgs_256.shape[0]
        estimated_noise_impact = (
            noise_std / (total_norm * np.sqrt(effective_batch_size))
            if total_norm > 0
            else float("inf")
        )
        current_epsilon = self.privacy_engine.get_epsilon(self.target_delta)

        if dist.is_local_master():
            if stepping:
                print(
                    f"[Step {self.cur_it}] "
                    f"Privacy epsilon: {current_epsilon:.2f} | "
                    f"Num samples this step: {imgs_256.shape[0]} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Accuracy: {avg_acc:.2f}% | "
                    f"Grad norm: {total_norm:.4f} | "
                    f"SNR: {signal_to_noise_ratio:.4f} | "
                    f"Noise std: {noise_std:.4f} | "
                    f"Noise impact: {estimated_noise_impact:.4f}"
                )

            wandb.log(
                {
                    "train_step/loss": avg_loss,
                    "train_step/train_acc": avg_acc,
                    "train_step/tail_loss": avg_loss_tail,
                    "train_step/tail_acc": avg_acc_tail,
                    "train_step/grad_norm": total_norm,
                    "train_step/privacy_epsilon": (
                        current_epsilon if current_epsilon is not None else 0.0
                    ),
                    "dp_metrics/noise_std": noise_std,
                    "dp_metrics/signal_to_noise_ratio": signal_to_noise_ratio,
                    "dp_metrics/estimated_noise_impact": estimated_noise_impact,
                }
            )

        return (
            total_norm,
            avg_loss,
            avg_acc,
            avg_loss_tail,
            avg_acc_tail,
        )

    def train(self) -> None:
        self.setup()

        ep = self.start_ep = 0
        pbar = tqdm(range(self.trainer_cfg.epochs), desc="Training Steps")

        dist.barrier()

        start_time = time.time()

        for ep in pbar:
            current_epsilon = self.privacy_engine.get_epsilon(self.target_delta)
            if current_epsilon >= self.trainer_cfg.target_epsilon:
                break

            if hasattr(self.data_loader, "sampler") and hasattr(
                self.data_loader.sampler, "set_epoch"
            ):
                self.data_loader.sampler.set_epoch(ep)

            stats, (sec, remain_time, finish_time) = self.train_one_ep(
                ep=ep,
                is_first_ep=(ep == self.start_ep),
                data_loader=self.data_loader,
            )

            L_mean, L_tail, acc_mean, acc_tail, grad_norm = (
                stats["Lm"],
                stats["Lt"],
                stats["Accm"],
                stats["Acct"],
                stats["tnm"],
            )

            AR_ep_loss = dict(
                L_mean=L_mean, L_tail=L_tail, acc_mean=acc_mean, acc_tail=acc_tail
            )
            is_val_and_also_saving = ((ep + 1) % self.trainer_cfg.checkpoint == 0) or (
                ep + 1
            ) == self.trainer_cfg.epochs

            if is_val_and_also_saving:

                # (
                #     self.val_loss_mean,
                #     self.val_loss_tail,
                #     self.val_acc_mean,
                #     self.val_acc_tail,
                #     tot,
                #     cost,
                # ) = self.eval_ep(self.ld_val)
                # AR_ep_loss.update(
                #     vL_mean=self.val_loss_mean,
                #     vL_tail=self.val_loss_tail,
                #     vacc_mean=self.val_acc_mean,
                #     vacc_tail=self.val_acc_tail,
                # ) Uncomment this if you want to evaluate for DP-SGD Training

                if dist.is_local_master():
                    print(
                        # f" [*] [ep{ep+1}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s"
                        f" [*] [ep{ep+1}]  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f}"
                    )
                self.save_ckpt(ep)

            if dist.is_local_master():
                print(
                    f"     [ep{ep+1}]  (training )  Lm: {L_mean:.3f}, Lt: {L_tail:.3f},  Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}",
                    flush=True,
                )

        total_time = f"{(time.time() - start_time) / 60 / 60:.1f}h"

        print("\n\n")
        print(
            f"  [*] [Training finished]  Total cost: {total_time},   Lm: {L_mean:.3f},   Lt: {L_tail:.3f}"
        )
        print("\n\n")

        gc.collect(), torch.cuda.empty_cache(), dist.barrier()


class DPVARFFTTrainer(DPVARTrainer, FFTVARTrainer):
    def setup_params(self):

        for p in self.var_wo_ddp.parameters():
            p.requires_grad = True

        self.var_wo_ddp.training = True
        self.print_trainable_parameters(self.var_wo_ddp)
        # self.debug_print_all_params()
        return self.var_wo_ddp

    def save_ckpt(self, ep: int) -> None:

        if dist.is_local_master():
            local_out_ckpt = os.path.join(self.output_dir, f"var_ckpt_ep{ep+1}")
            os.makedirs(local_out_ckpt, exist_ok=True)
            print(f"[Saving checkpoint at epoch {ep+1}] ...", end="", flush=True)
            checkpoint = {
                "epoch": ep + 1,
                "trainer": self.state_dict(),
                "dp_eps": self.privacy_engine.get_epsilon(self.target_delta),
                "metrics": {
                    "val_loss_mean": self.val_loss_mean,
                    "val_loss_tail": self.val_loss_tail,
                    "val_acc_mean": self.val_acc_mean,
                    "val_acc_tail": self.val_acc_tail,
                },
            }
            filename = f"var.pth"
            ckpt_path = os.path.join(local_out_ckpt, filename)
            torch.save(checkpoint, ckpt_path)
            print(f"[Checkpoint saved] @ {local_out_ckpt}", flush=True)
        dist.barrier()


class DPVARLoraTrainer(DPVARTrainer):

    def adalin_module_names(self, model):
        """
        We need modules that include "ada_lin.1" but exclude "head_nm"
        because, although "head_nm" is also an "ada_lin.1" module, it
        is not the one we are looking for.
        """
        target_module_names = []
        for name, module in model.named_modules():
            if "ada_lin.1" in name and "head_nm" not in name:
                target_module_names.append(name)
        return target_module_names

    def setup_params(self):

        for p in self.var_wo_ddp.parameters():
            p.requires_grad = False

        self.var_wo_ddp.training = True

        if "ada_lin.1" in self.training_params.target_modules:
            self.training_params.target_modules.remove("ada_lin.1")
            self.training_params.target_modules.extend(
                self.adalin_module_names(self.var_wo_ddp)
            )

        if dist.is_local_master():
            print(f"[*] LoRA Target Modules: {self.training_params.target_modules}")

        self.peft_cfg = LoraConfig(
            r=self.training_params.rank,
            lora_alpha=(self.training_params.rank * 2),
            init_lora_weights="gaussian",
            lora_dropout=self.training_params.lora_dropout,
            target_modules=list(self.training_params.target_modules),
        )

        self.var_wo_ddp = get_peft_model(self.var_wo_ddp, self.peft_cfg)
        self.print_trainable_parameters(self.var_wo_ddp)
        # self.debug_print_all_params()

        return self.var_wo_ddp

    def save_ckpt(self, ep: int) -> None:

        if dist.is_local_master():

            local_out_ckpt = os.path.join(self.output_dir, f"var_ckpt_ep{ep+1}/")
            print(f"[Saving checkpoint at epoch {ep+1}] ...", end="", flush=True)
            os.makedirs(self.output_dir, exist_ok=True)
            self.var_wo_ddp.save_pretrained(
                save_directory=local_out_ckpt,
                is_main_process=True,
                safe_serialization=True,
            )

            print(f"[Checkpoint saved] @ {local_out_ckpt}", flush=True)
        dist.barrier()


class DPVARLNTuningTrainer(DPVARTrainer):

    def adalin_module_names(self, model):
        """
        We need modules that include "ada_lin.1" but exclude "head_nm"
        because, although "head_nm" is also an "ada_lin.1" module, it
        is not the one we are looking for.
        """
        target_module_names = []
        for name, module in model.named_modules():
            if "ada_lin.1" in name and "head_nm" not in name:
                target_module_names.append(name)
        return target_module_names

    def setup_params(self):

        for p in self.var_wo_ddp.parameters():
            p.requires_grad = False

        self.var_wo_ddp.training = True

        self.training_params.target_modules.remove("ada_lin.1")
        self.training_params.target_modules.extend(
            self.adalin_module_names(self.var_wo_ddp)
        )

        if dist.is_local_master():
            print(
                f"[*] LayerNormTuning Target Modules: {self.training_params.target_modules}"
            )

        self.peft_cfg = LNTuningConfig(
            target_modules=list(self.training_params.target_modules),
            peft_type="LN_TUNING",
        )

        self.var_wo_ddp = get_peft_model(self.var_wo_ddp, self.peft_cfg)
        self.print_trainable_parameters(self.var_wo_ddp)
        # self.debug_print_all_params()

        return self.var_wo_ddp

    def save_ckpt(self, ep: int) -> None:

        if dist.is_local_master():

            local_out_ckpt = os.path.join(self.output_dir, f"var_ckpt_ep{ep+1}/")
            print(f"[Saving checkpoint at epoch {ep+1}] ...", end="", flush=True)
            os.makedirs(self.output_dir, exist_ok=True)
            self.var_wo_ddp.save_pretrained(
                save_directory=local_out_ckpt,
                is_main_process=True,
                safe_serialization=True,
            )

            print(f"[Checkpoint saved] @ {local_out_ckpt}", flush=True)
        dist.barrier()
