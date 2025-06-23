# REF: https://github.com/FoundationVision/VAR/tree/main

"""
Utility module for various helper functions compiled together.
"""

import torch
from typing import List, Tuple, Dict, Union, Optional
from src.utils import dist
from pprint import pformat, pprint


class NullDDP(torch.nn.Module):
    """
    Null DDP module to handle the case when DDP is not used.
    """

    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def filter_params(model, nowd_keys=()) -> Tuple[
    List[str],
    List[torch.nn.Parameter],
    List[Dict[str, Union[torch.nn.Parameter, float]]],
]:
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    count, numel = 0, 0
    for name, para in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if not para.requires_grad:
            names_no_grad.append(name)
            continue  # frozen weights
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)

        if para.ndim == 1 or name.endswith("bias") or any(k in name for k in nowd_keys):
            cur_wd_sc, group_name = 0.0, "ND"
        else:
            cur_wd_sc, group_name = 1.0, "D"
        cur_lr_sc = 1.0
        if group_name not in para_groups:
            para_groups[group_name] = {
                "params": [],
                "wd_sc": cur_wd_sc,
                "lr_sc": cur_lr_sc,
            }
            para_groups_dbg[group_name] = {
                "params": [],
                "wd_sc": cur_wd_sc,
                "lr_sc": cur_lr_sc,
            }
        para_groups[group_name]["params"].append(para)
        para_groups_dbg[group_name]["params"].append(name)

    for g in para_groups_dbg.values():
        g["params"] = pformat(", ".join(g["params"]), width=200)

    print(
        f"[get_param_groups] param_groups = \n{pformat(para_groups_dbg, indent=2, width=240)}\n"
    )

    for rk in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == rk:
            print(
                f"[get_param_groups][rank{dist.get_rank()}] {type(model).__name__=} {count=}, {numel=}",
                flush=True,
            )
    print("")

    if len(names_no_grad) > 0:
        print(
            f"[get_param_groups] Found {len(names_no_grad)} frozen parameters, make sure this is intended due to LoRA.\n\n"
        )

    return names, paras, list(para_groups.values())
