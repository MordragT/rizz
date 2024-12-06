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

@dataclass
class RizzConfig:
        resources = Path(__file__).parent.parent.resolve() / "resources"
        dtype = torch.bfloat16
        device = device()
