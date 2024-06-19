import random
import torchvision.transforms.functional as F
import torch
from typing import Tuple
from einops import rearrange

def flip_and_rotate_space(
        space_time_x: torch.Tensor,
        space_x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    # 
    space_time_x = rearrange(space_time_x, 'b h w t')
    if random.random() <= 0.5:
        space_time_x = F.vflip(space_time_x)
        space_x = F.vflip(space_x)
    
    if random.random() <= 0.5:
        space_time_x = F.hflip(space_time_x)
        space_x = F.hflip(space_x)

