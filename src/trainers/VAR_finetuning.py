import os, time, gc
import os.path as osp

import wandb
from tqdm.auto import tqdm
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.transforms import InterpolationMode, transforms
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS

from peft import LoraConfig, LNTuningConfig, get_peft_model

from src.utils import dist
from src.local_datasets import datasets
from src.trainers.trainer import Trainer
from src.trainers.utils import NullDDP, filter_params
from src.utils.var import VAR, VQVAE, VectorQuantizer2, build_vae_var
from src.local_datasets.utils import pil_loader, print_aug, normalize_01_into_pm1

from submodules.VAR.utils.amp_sc import AmpOptimizer
from submodules.VAR.utils.data_sampler import (
    EvalDistributedSampler,
    DistInfiniteBatchSampler,
)

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(Trainer):

    def load_models(self):
        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt, var_ckpt = (
            "vae_ch160v4096z32.pth",
            f"var_d{self.model_cfg.model_depth}.pth",
        )
        if dist.is_local_master():
            if not os.path.exists(f"out/model_checkpoints/var/{vae_ckpt}"):
                os.makedirs("out/model_checkpoints/var/", exist_ok=True)
                os.system(
                    f"wget {hf_home}/{vae_ckpt} -O out/model_checkpoints/var/{vae_ckpt}"
                )
            if not os.path.exists(f"out/model_checkpoints/var/{var_ckpt}"):
                os.makedirs("out/model_checkpoints/var/", exist_ok=True)
                os.system(
                    f"wget {hf_home}/{var_ckpt} -O out/model_checkpoints/var/{var_ckpt}"
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
            print("Debugging Enabled, Printing all parameters.")
            print("=" * 50)
            for name, param in self.var_wo_ddp.named_parameters():
                if param.requires_grad:
                    print(name, param.size())
            print("=" * 50)
            print("=" * 50)

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

        names, paras, para_groups = filter_params(
            self.var_wo_ddp,
            nowd_keys={
                "cls_token",
                "start_token",
                "task_token",
                "cfg_uncond",
                "pos_embed",
                "pos_1LC",
                "pos_start",
                "start_pos",
                "lvl_embed",
                "gamma",
                "beta",
                "ada_gss",
                "moe_bias",
                "scale_mul",
            },
        )

        opt_clz = {
            "adam": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=True),
            "adamw": partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=True),
        }[self.training_params.opt.lower().strip()]

        opt_kw = dict(
            lr=self.training_params.lr,
            weight_decay=self.training_params.weight_decay,
        )

        self.var_opt = AmpOptimizer(
            mixed_precision=2,
            optimizer=opt_clz(
                params=para_groups,
                **opt_kw,
            ),
            names=names,
            paras=paras,
            grad_clip=self.training_params.grad_clip,
            n_gradient_accumulation=self.training_params.n_gradient_accumulation,
        )

        del names, paras, para_groups
        return self.var_opt

    def setup(self) -> None:

        dist.barrier()
        self.setup_model()
        self.setup_params()
        self.vae_local = self.vae_local.to(dist.get_device())
        self.var_wo_ddp = self.var_wo_ddp.to(dist.get_device())
        dist.barrier()

        self.var: DDP = (DDP if dist.initialized() else NullDDP)(
            self.var_wo_ddp,
            device_ids=[dist.get_local_rank()],
            find_unused_parameters=(
                True if dist.initialized() else False
            ),  # Useful when using DDP with LoRA
            broadcast_buffers=False,
        )

        dist.barrier()

        self.build_optimizer()
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
            except Exception as e:
                pass
            finally:
                self.var_wo_ddp.base_model.rng = torch.Generator(
                    device=dist.get_device()
                )

        if dist.is_local_master():
            if self.training_params.n_gradient_accumulation > 1:
                print(
                    f"Gradiant Accumulated after {self.training_params.n_gradient_accumulation} steps"
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

    def build_var_dataset(
        self,
        data_path: str,
        final_reso: int,
        hflip=False,
        mid_reso=1.125,
    ):
        mid_reso = round(
            mid_reso * final_reso
        )  # first resize to mid_reso, then crop to final_reso
        train_aug, val_aug = [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.LANCZOS
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ], [
            transforms.Resize(
                mid_reso, interpolation=InterpolationMode.LANCZOS
            ),  # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(),
            normalize_01_into_pm1,
        ]
        if hflip:
            train_aug.insert(0, transforms.RandomHorizontalFlip())
        train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)

        train_set = DatasetFolder(
            root=osp.join(data_path, "train"),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=train_aug,
        )
        try:
            # Check if val folder name is val or validation
            val_path = (
                osp.join(data_path, "val")
                if osp.exists(osp.join(data_path, "val"))
                else osp.join(data_path, "validation")
            )
            val_set = DatasetFolder(
                root=val_path,
                loader=pil_loader,
                extensions=IMG_EXTENSIONS,
                transform=val_aug,
            )
        except Exception as e:
            print(
                f"Looking for 'val/validation' dir failed, using 'test' folder instead. Dataset {e=}, {data_path=}"
            )
            val_set = DatasetFolder(
                root=osp.join(data_path, "test"),
                loader=pil_loader,
                extensions=IMG_EXTENSIONS,
                transform=val_aug,
            )

        num_classes = self.dataset_cfg.num_classes
        print(f"[Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}")
        print_aug(train_aug, "[train]")
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

        return self.dataset_train, self.dataset_val

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
            persistent_workers=True,
            pin_memory=True,
            batch_size=(self.model_cfg.batch_size),
            sampler=EvalDistributedSampler(
                self.dataset_val,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
            ),
            shuffle=False,
            drop_last=False,
        )

        self.ld_train = DataLoader(
            self.dataset_train,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            generator=get_different_generator_for_each_rank(),
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(self.dataset_train),
                glb_batch_size=self.global_batch_size,
                same_seed_for_all_ranks=0,
                shuffle=True,
                fill_last=True,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
            ),
        )

        if dist.is_local_master():
            print("=" * 50)
            print(
                f"[*] Global Batch Size: {(self.global_batch_size * self.training_params.n_gradient_accumulation)}"
            )
            print(
                f"[*] Batch Size Per GPU: {(self.global_batch_size * self.training_params.n_gradient_accumulation) // dist.get_world_size()}"
            )
            print(
                f"[*] Num Iters Per Epoch: {self.ld_train.batch_sampler.iters_per_ep}"
            )
            print(
                f"[*] Total Iters: {self.ld_train.batch_sampler.iters_per_ep * self.trainer_cfg.epochs}"
            )
            print("=" * 50)

        del self.dataset_train
        del self.dataset_val

        return self.ld_train, self.ld_val

    def get_param_norm(model):
        total_norm = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total_norm += param.data.norm(2).item() ** 2
        return total_norm**0.5

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
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
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

    def train_step(
        self,
        stepping: bool,
        inp_B3HW: FTen,
        label_B: Union[ITen, FTen],
    ) -> Tuple[
        Optional[Union[Ten, float]], Optional[float], float, float, float, float
    ]:
        """Optimizer Scaler is set to None so `scale_log2` will always be None"""
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()

            # Calculate accuracy
            with torch.no_grad():
                predictions = logits_BLV.argmax(dim=-1)
                accuracy = (predictions == gt_BL).float().mean() * 100

            # Calculate tail loss and tail accuracy
            with torch.no_grad():
                tail_logits = logits_BLV[:, -self.last_l :]
                tail_targets = gt_BL[:, -self.last_l :]
                loss_tail = self.val_loss(
                    tail_logits.reshape(-1, V), tail_targets.reshape(-1)
                )
                acc_tail = (
                    tail_logits.argmax(dim=-1) == tail_targets
                ).float().mean() * 100

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(
            loss=loss, stepping=stepping
        )

        if dist.is_local_master():
            lr = self.var_opt.optimizer.param_groups[0]["lr"]
            if stepping:
                print(
                    f"[Step {self.cur_it}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Accuracy: {accuracy.item():.2f}% | "
                    f"Tail Loss: {loss_tail.item():.4f} | "
                    f"Tail Acc: {acc_tail.item():.2f}% | "
                    f"Grad Norm: {grad_norm:.4f} | "
                    f"LR: {lr:.6f}"
                )
            else:
                print(
                    f"[Step {self.cur_it}] "
                    f"Loss: {loss.item():.4f} | "
                    f"Accuracy: {accuracy.item():.2f}% | "
                    f"Tail Loss: {loss_tail.item():.4f} | "
                    f"Tail Acc: {acc_tail.item():.2f}% | "
                    f"LR: {lr:.6f} | "
                )
            wandb.log(
                {
                    "train_step/loss": loss.item(),
                    "train_step/train_acc": accuracy.item(),
                    "train_step/tail_loss": loss_tail.item(),
                    "train_step/tail_acc": acc_tail.item(),
                    "train_step/grad_norm": grad_norm,
                    "train_step/scale_log2": scale_log2,
                }
            )

        return (
            grad_norm,
            scale_log2,
            loss.item(),
            accuracy.item(),
            loss_tail.item(),
            acc_tail.item(),
        )

    def train_one_ep(
        self,
        ep: int,
        is_first_ep: bool,
        data_loader,
        iters_train: int,
    ):
        """Train for one epoch"""

        from submodules.VAR.utils.lr_control import lr_wd_annealing

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

        g_it = ep * iters_train
        data_it = iter(data_loader)
        max_iter = self.trainer_cfg.epochs * iters_train

        running_loss = 0.0
        running_acc = 0.0
        running_loss_tail = 0.0
        running_acc_tail = 0.0

        for param_group in self.var_opt.optimizer.param_groups:
            param_group["lr"] = self.training_params.lr

        for it, (inp, label) in enumerate(data_it):
            if it >= iters_train:
                break

            inp = inp.to(dist.get_device(), non_blocking=True)
            label = label.to(dist.get_device(), non_blocking=True)

            self.cur_it = f"{it+1}/{iters_train}"

            stepping = (g_it + 1) % self.training_params.n_gradient_accumulation == 0
            step_cnt += int(stepping)

            grad_norm, scale_log2, loss_val, acc_val, loss_tail_val, acc_tail_val = (
                self.train_step(
                    stepping=stepping,
                    inp_B3HW=inp,
                    label_B=label,
                )
            )

            running_loss += loss_val
            running_acc += acc_val
            running_loss_tail += loss_tail_val
            running_acc_tail += acc_tail_val

            g_it = ep * iters_train + (it + 1)

            # Print progress every 100 iterations
            if (it + 1) % 100 == 0 and dist.is_local_master():
                elapsed = time.time() - start_time
                steps_per_sec = (it + 1) / elapsed
                remaining_steps = iters_train - (it + 1)
                eta = remaining_steps / steps_per_sec

                print(
                    f"\n[Epoch {ep+1} Progress] "
                    f"{it+1}/{iters_train} steps | "
                    f"Steps/sec: {steps_per_sec:.2f} | "
                    f"ETA: {eta/60:.2f}min"
                )

        # Calculate epoch statistics
        stats = {
            "Lm": running_loss / iters_train,
            "Lt": running_loss_tail / iters_train,
            "Accm": running_acc / iters_train,
            "Acct": running_acc_tail / iters_train,
            "tnm": grad_norm,
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
                    "epoch": ep,
                }
            )

        return stats, (elapsed, remain_time, finish_time)

    def get_config(self):
        return {
            "patch_nums": self.model_cfg.patch_nums,
            "label_smooth": self.label_smooth,
        }

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

    def state_dict(self):
        state = {"config": self.get_config()}
        # Only save VAR model state
        m = getattr(self, "var_wo_ddp")
        if m is not None:
            if hasattr(m, "_orig_mod"):
                m = m._orig_mod
            state["var_wo_ddp"] = m.state_dict()
        return state

    def save_ckpt(self, ep: int) -> None:
        NotImplementedError

    def load_state_dict(self, state, strict=True):
        # Only load VAR model state
        m = getattr(self, "var_wo_ddp")
        if m is not None:
            if hasattr(m, "_orig_mod"):
                m = m._orig_mod
            ret = m.load_state_dict(state["var_wo_ddp"], strict=strict)
            if ret is not None:
                missing, unexpected = ret
                print(f"[VARTrainer.load_state_dict] var_wo_ddp missing: {missing}")
                print(
                    f"[VARTrainer.load_state_dict] var_wo_ddp unexpected: {unexpected}"
                )

        config: dict = state.pop("config", None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f"[VAR.load_state_dict] config mismatch: this.{k}={v} (ckpt.{k}={config.get(k, None)})"
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)

    def train(self) -> None:
        self.setup()

        ep = self.start_ep = 0
        pbar = tqdm(range(self.trainer_cfg.epochs), desc="Training Epochs")

        dist.barrier()

        start_time = time.time()

        for ep in pbar:
            if hasattr(self.ld_train, "sampler") and hasattr(
                self.ld_train.sampler, "set_epoch"
            ):
                self.ld_train.sampler.set_epoch(ep)

            stats, (sec, remain_time, finish_time) = self.train_one_ep(
                ep=ep,
                is_first_ep=(ep == self.start_ep),
                data_loader=self.ld_train,
                iters_train=len(self.ld_train),
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
                (
                    self.val_loss_mean,
                    self.val_loss_tail,
                    self.val_acc_mean,
                    self.val_acc_tail,
                    tot,
                    cost,
                ) = self.eval_ep(self.ld_val)
                AR_ep_loss.update(
                    vL_mean=self.val_loss_mean,
                    vL_tail=self.val_loss_tail,
                    vacc_mean=self.val_acc_mean,
                    vacc_tail=self.val_acc_tail,
                )
                if dist.is_local_master():
                    print(
                        f" [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Val cost: {cost:.2f}s"
                    )
                self.save_ckpt(ep)

            if dist.is_local_master():
                print(
                    f"     [ep{ep}]  (training )  Lm: {L_mean:.3f}, Lt: {L_tail:.3f},  Acc m&t: {acc_mean:.2f} {acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}",
                    flush=True,
                )

        total_time = f"{(time.time() - start_time) / 60 / 60:.1f}h"

        print("\n\n")
        print(
            f"  [*] [Training finished]  Total cost: {total_time},   Lm: {L_mean:.3f},   Lt: {L_tail:.3f}"
        )
        print("\n\n")

        gc.collect(), torch.cuda.empty_cache(), dist.barrier()


class FFTVARTrainer(VARTrainer):

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


class LoraVARTrainer(VARTrainer):

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
        self.debug_print_all_params()

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


class LNTuningVARTrainer(VARTrainer):

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
        self.debug_print_all_params()

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
