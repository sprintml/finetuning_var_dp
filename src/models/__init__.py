"""Module for Training and Eval implementations.

This module provides different VAR model implementations:
- Standard VAR model with training wrapper
- FFT VAR model with training wrapper
- Lora VAR model with training wrapper
- LNTuning VAR model with training wrapper

Usage:
    For training:
        Use the `gen_models` dictionary with TrainVARWrapper
    For inference:
        Use the `eval_models` dictionary for Sampling and Evaluation
"""

from typing import Dict, Type
from torch import nn

from src.models.VARModel import (
    VARModel,
    FFTVARModel,
    LoraVARModel,
    LNTuningVARModel,
)

from src.models.TrainModel import TrainVARWrapper

TrainModelRegistry = Dict[str, Dict[str, Type[nn.Module]]]
gen_models: TrainModelRegistry = {
    **{
        var_gen: {
            "var_fft": TrainVARWrapper,
            "var_lora": TrainVARWrapper,
            "var_lntuning": TrainVARWrapper,
        }
        for var_gen in ["var_16", "var_20", "var_24", "var_30"]
    },
}

EvalModelRegistry = Dict[str, Dict[str, Type[VARModel]]]
eval_models: EvalModelRegistry = {
    **{
        var_eval: {
            "none": VARModel,
            "var_fft": FFTVARModel,
            "var_lora": LoraVARModel,
            "var_lntuning": LNTuningVARModel,
        }
        for var_eval in ["var_16", "var_20", "var_24", "var_30"]
    }
}
