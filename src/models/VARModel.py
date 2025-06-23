import sys

sys.path.append("./submodules/VAR")
sys.path.append("./submodules/VAR/models")

from submodules.VAR.models import build_vae_var
import torch
from torch import nn
from torch import Tensor as T
from omegaconf import DictConfig
import os
from peft import PeftModel

import logging

logging.disable(logging.INFO)


class VARModel(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        model_cfg: DictConfig,
        dataset_cfg: DictConfig,
        finetuning_cfg: DictConfig,
        trainer_cfg: DictConfig,
        action_cfg: DictConfig,
    ):
        super(VARModel, self).__init__()
        vae, var = build_vae_var(
            V=model_cfg.V,
            Cvae=model_cfg.Cvae,
            ch=model_cfg.ch,
            num_classes=dataset_cfg.num_classes,
            share_quant_resi=model_cfg.share_quant_resi,
            device=action_cfg.device,
            patch_nums=model_cfg.patch_nums,
            depth=model_cfg.model_depth,
            shared_aln=model_cfg.shared_aln,
            flash_if_available=True,
            fused_if_available=True,
        )
        self.vae = vae
        self.var = var

        self.config = config
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
        self.finetuning_cfg = finetuning_cfg
        self.training_params = finetuning_cfg.training_params
        self.trainer_cfg = trainer_cfg
        self.action_cfg = action_cfg

        self.load_models()

        print(f"Model Checkpoint Loaded successfully.\n")

    def load_models(self):
        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt = "vae_ch160v4096z32.pth"
        var_ckpt = f"var_d{self.model_cfg.model_depth}.pth"

        out_dir = "out/model_checkpoints/var"
        os.makedirs(out_dir, exist_ok=True)

        vae_pth = os.path.join(out_dir, vae_ckpt)
        var_local = os.path.join(out_dir, var_ckpt)

        if not os.path.exists(vae_pth):
            os.system(f"wget {hf_home}/{vae_ckpt} -O {vae_pth}")
        if not os.path.exists(var_local):
            os.system(f"wget {hf_home}/{var_ckpt} -O {var_local}")

        self.vae.load_state_dict(torch.load(vae_pth, map_location="cpu"))
        var_ckpt = torch.load(var_local, map_location="cpu")
        missing_keys, unexpected_keys = self.var.load_state_dict(var_ckpt, strict=False)

        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print("Following keys are missing or unexpected in the loaded model:")
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

    def get_path(self) -> str:
        path = os.path.join(
            self.config.path_to_models,
            self.model_cfg.name,
            self.finetuning_cfg.name,
            self.trainer_cfg.name,
            self.dataset_cfg.name,
            str(self.model_cfg.subdir) if hasattr(self.model_cfg, "subdir") else "",
            f"var_ckpt_ep{self.trainer_cfg.checkpoint}",
        )
        print(f"VAR Checkpoint Path: {path}")
        return path


class FFTVARModel(VARModel):

    def load_models(self):
        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt = "vae_ch160v4096z32.pth"
        out_dir = "out/model_checkpoints/var"
        os.makedirs(out_dir, exist_ok=True)

        vae_pth = os.path.join(out_dir, vae_ckpt)
        if not os.path.exists(vae_pth):
            os.system(f"wget {hf_home}/{vae_ckpt} -O {vae_pth}")

        var_pth = self.get_path()
        var_ckpt = torch.load(f"{var_pth}/var.pth", map_location="cpu")
        var_ckpt_state_dict = var_ckpt["trainer"]["var_wo_ddp"]

        self.vae.load_state_dict(torch.load(vae_pth, map_location="cpu"))
        unexpected_keys, missing_keys = self.var.load_state_dict(
            var_ckpt_state_dict, strict=False
        )

        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print("Following keys are missing or unexpected in the loaded model:")
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

        self.vae.eval(), self.var.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.var.parameters():
            p.requires_grad = False


class LoraVARModel(VARModel):

    def load_models(self):
        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt = "vae_ch160v4096z32.pth"
        var_ckpt = f"var_d{self.model_cfg.model_depth}.pth"

        out_dir = "out/model_checkpoints/var"
        os.makedirs(out_dir, exist_ok=True)

        vae_pth = os.path.join(out_dir, vae_ckpt)
        var_local = os.path.join(out_dir, var_ckpt)

        if not os.path.exists(vae_pth):
            os.system(f"wget {hf_home}/{vae_ckpt} -O {vae_pth}")
        if not os.path.exists(var_local):
            os.system(f"wget {hf_home}/{var_ckpt} -O {var_local}")

        self.vae.load_state_dict(torch.load(vae_pth, map_location="cpu"))

        var_ckpt = torch.load(var_local, map_location="cpu")
        if "class_emb.weight" in var_ckpt:
            del var_ckpt["class_emb.weight"]
        if "class_emb.bias" in var_ckpt:
            del var_ckpt["class_emb.bias"]
        self.var.load_state_dict(var_ckpt, strict=False)
        adapter_path = self.get_path()

        self.var = PeftModel.from_pretrained(
            self.var,
            adapter_path,
            is_trainable=False,
        )

        self.var = self.var.merge_and_unload(
            progressbar=True,
            safe_merge=True,
            adapter_names=["default"],
        )
        print(f"[*] LoRA Adapter Weights Merged Successfully for faster inference.\n")

        self.vae.eval(), self.var.eval()


class LNTuningVARModel(VARModel):

    def load_models(self):
        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt = "vae_ch160v4096z32.pth"
        var_ckpt = f"var_d{self.model_cfg.model_depth}.pth"

        out_dir = "out/model_checkpoints/var"
        os.makedirs(out_dir, exist_ok=True)

        vae_pth = os.path.join(out_dir, vae_ckpt)
        var_local = os.path.join(out_dir, var_ckpt)

        if not os.path.exists(vae_pth):
            os.system(f"wget {hf_home}/{vae_ckpt} -O {vae_pth}")
        if not os.path.exists(var_local):
            os.system(f"wget {hf_home}/{var_ckpt} -O {var_local}")

        self.vae.load_state_dict(torch.load(vae_pth, map_location="cpu"))

        var_ckpt = torch.load(var_local, map_location="cpu")
        if "class_emb.weight" in var_ckpt:
            del var_ckpt["class_emb.weight"]
        if "class_emb.bias" in var_ckpt:
            del var_ckpt["class_emb.bias"]
        self.var.load_state_dict(var_ckpt, strict=False)
        adapter_path = self.get_path()

        self.var = PeftModel.from_pretrained(
            self.var,
            adapter_path,
            is_trainable=False,
        )

        self.var = self.var.merge_and_unload(
            progressbar=True,
            safe_merge=True,
            adapter_names=["default"],
        )
        print(
            f"[*] LayerNormTuning Adapter Weights Merged Successfully for faster inference.\n"
        )

        self.vae.eval(), self.var.eval()
