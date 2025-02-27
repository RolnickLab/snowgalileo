import random
from typing import Dict, Tuple

import torch
import torchvision.transforms.v2.functional as F
from einops import rearrange


class FlipAndRotateSpace(object):
    """
    For now, lets have no parameters
    Choose 1 of 8 transformations and apply it to space_time_x and space_x
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.transformations = [
            self.no_transform,  # No transformation
            self.rotate_90,  # 90-degree rotation
            self.rotate_180,  # 180-degree rotation
            self.rotate_270,  # 270-degree rotation
            self.hflip,  # Horizontal flip
            self.vflip,  # Vertical flip
            self.hflip_rotate_90,  # Horizontal flip of 90-degree rotated image
            self.vflip_rotate_90,  # Vertical flip of 90-degree rotated image
        ]

    def no_transform(self, x):
        return x

    def rotate_90(self, x):
        return F.rotate(x, 90)

    def rotate_180(self, x):
        return F.rotate(x, 180)

    def rotate_270(self, x):
        return F.rotate(x, 270)

    def hflip(self, x):
        return F.hflip(x)

    def vflip(self, x):
        return F.vflip(x)

    def hflip_rotate_90(self, x):
        return F.hflip(F.rotate(x, 90))

    def vflip_rotate_90(self, x):
        return F.vflip(F.rotate(x, 90))

    def apply(
        self,
        space_time_h_x: torch.Tensor,
        space_time_m_x: torch.Tensor,
        space_time_l_x: torch.Tensor,
        space_x: torch.Tensor,
        valid_data_mask_s_t_h: torch.Tensor,
        valid_data_mask_s_t_m: torch.Tensor,
        valid_data_mask_s_t_l: torch.Tensor,
        valid_data_mask_sp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return space_time_h_x, space_time_m_x, space_time_l_x, space_x, valid_data_mask_s_t_h, valid_data_mask_s_t_m, valid_data_mask_s_t_l, valid_data_mask_sp

        space_time_h_x = rearrange(
            space_time_h_x.float(), "b h w t c -> b t c h w"
        )  # rearrange for transforms
        space_time_m_x = rearrange(
            space_time_m_x.float(), "b h w t c -> b t c h w"
        )
        space_time_l_x = rearrange(
            space_time_l_x.float(), "b h w t c -> b t c h w"
        )
        space_x = rearrange(space_x.float(), "b h w c -> b c h w")  # rearrange for transforms
        valid_data_mask_s_t_h = rearrange(
            valid_data_mask_s_t_h.float(), "b h w t c -> b t c h w"
        )
        valid_data_mask_s_t_m = rearrange(
            valid_data_mask_s_t_m.float(), "b h w t c -> b t c h w"
        )
        valid_data_mask_s_t_l = rearrange(
            valid_data_mask_s_t_l.float(), "b h w t c -> b t c h w"
        )
        valid_data_mask_sp = rearrange(valid_data_mask_sp.float(), "b h w c -> b c h w")  # rearrange for transforms

        transformation = random.choice(self.transformations)

        space_time_h_x = rearrange(
            transformation(space_time_h_x), "b t c h w -> b h w t c"
        )  # rearrange back
        space_time_m_x = rearrange(
            transformation(space_time_m_x), "b t c h w -> b h w t c"
        )
        space_time_l_x = rearrange(
            transformation(space_time_l_x), "b t c h w -> b h w t c"
        )
        space_x = rearrange(transformation(space_x), "b c h w -> b h w c")  # rearrange back
        valid_data_mask_s_t_h = rearrange(
            transformation(valid_data_mask_s_t_h), "b t c h w -> b h w t c"
        )
        valid_data_mask_s_t_m = rearrange(
            transformation(valid_data_mask_s_t_m), "b t c h w -> b h w t c"
        )
        valid_data_mask_s_t_l = rearrange(
            transformation(valid_data_mask_s_t_l), "b t c h w -> b h w t c"
        )
        valid_data_mask_sp = rearrange(transformation(valid_data_mask_sp), "b c h w -> b h w c")  # rearrange back

        return space_time_h_x.half(), space_time_m_x.half(), space_time_l_x.half(), space_x.half(), valid_data_mask_s_t_h.half(), valid_data_mask_s_t_m.half(), valid_data_mask_s_t_l.half(), valid_data_mask_sp.half()


class Augmentation(object):
    def __init__(self, aug_config: Dict):
        self.flip_and_rotate = FlipAndRotateSpace(enabled=aug_config.get("flip+rotate", False))

    def apply(
        self,
        space_time_h_x: torch.Tensor,
        space_time_m_x: torch.Tensor,
        space_time_l_x: torch.Tensor,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        static_x: torch.Tensor,
        months: torch.Tensor,
        valid_data_mask_s_t_h: torch.Tensor,
        valid_data_mask_s_t_m: torch.Tensor,
        valid_data_mask_s_t_l: torch.Tensor,
        valid_data_mask_sp: torch.Tensor,
        valid_data_mask_t: torch.Tensor,
        valid_data_mask_st: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        space_time_h_x, space_time_m_x, space_time_l_x, space_x, valid_data_mask_s_t_h, valid_data_mask_s_t_m, valid_data_mask_s_t_l, valid_data_mask_sp = self.flip_and_rotate.apply(space_time_h_x, space_time_m_x, space_time_l_x, space_x, valid_data_mask_s_t_h, valid_data_mask_s_t_m, valid_data_mask_s_t_l, valid_data_mask_sp)

        return space_time_h_x, space_time_m_x, space_time_l_x, space_x, time_x, static_x, months, valid_data_mask_s_t_h, valid_data_mask_s_t_m, valid_data_mask_s_t_l, valid_data_mask_sp, valid_data_mask_t, valid_data_mask_st
