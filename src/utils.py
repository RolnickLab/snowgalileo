import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_SEED
from .masking import MaskedOutput

data_dir = Path(__file__).parent.parent / "data"

logging_dir = Path(__file__).parent.parent / "logs"

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


# From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int = DEFAULT_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def masked_ouptput_np_to_tensor(d_x, s_x, d_m, s_m, month) -> MaskedOutput:
    """converts eval task"""
    d_x_torch = torch.as_tensor(d_x, dtype=torch.float32)
    s_x_torch = torch.as_tensor(s_x, dtype=torch.float32)
    d_m_torch = torch.as_tensor(d_m, dtype=torch.float32)
    s_m_torch = torch.as_tensor(s_m, dtype=torch.float32)
    month_torch = torch.as_tensor(month, dtype=torch.long)
    return MaskedOutput(d_x_torch, s_x_torch, d_m_torch, s_m_torch, month_torch)


class AverageMeter:
    """computes and stores the average and current value"""

    average: float
    sum: float
    count: int

    def __init__(self):
        self.average = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count
