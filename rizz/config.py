import torch
from dataclasses import dataclass
from pathlib import Path

def device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

# TODO better way ?
def ov_device():
    if torch.xpu.is_available():
        return "GPU"
    else:
         return "CPU"

@dataclass
class RizzConfig:
        resources = Path(__file__).parent.parent.resolve() / "resources"
        dtype = torch.bfloat16
        device = device()
        ov_device = ov_device()

        print(f"Using device: {device}")
        print(f"Using OpenVINO device: {ov_device}")
