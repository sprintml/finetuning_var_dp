from src.trainers.utils import NullDDP
from src.trainers.trainer import Trainer

from src.trainers.VAR_finetuning import (
    FFTVARTrainer as FFTVAR,
    LoraVARTrainer as LoraVAR,
    LNTuningVARTrainer as LNTuningVAR,
)

from src.trainers.VAR_dp_trainer import (
    DPVARFFTTrainer as DPFFTVAR,
    DPVARLoraTrainer as DPLoraVAR,
    DPVARLNTuningTrainer as DPLNTuningVAR,
)

from src.trainers.VAR_profiler import (
    FFTVARProfiler,
    LoraVARProfiler,
    LNTuningVARProfiler,
)

from typing import Dict

dp_trainers = {
    "var_fft": DPFFTVAR,
    "var_lora": DPLoraVAR,
    "var_lntuning": DPLNTuningVAR,
}

trainers: Dict[str, Dict[str, Trainer]] = {
    "var_finetuning": {
        "var_fft": FFTVAR,
        "var_lora": LoraVAR,
        "var_lntuning": LNTuningVAR,
    },
    "var_profiler": {
        "var_fft": FFTVARProfiler,
        "var_lora": LoraVARProfiler,
        "var_lntuning": LNTuningVARProfiler,
    },
    "dp-eps-1": dp_trainers,
    "dp-eps-10": dp_trainers,
    "dp-eps-20": dp_trainers,
    "dp-eps-50": dp_trainers,
    "dp-eps-100": dp_trainers,
    "dp-eps-500": dp_trainers,
    "dp-eps-1000": dp_trainers,
    "dp-eps-100000": dp_trainers,
}
