import random
from typing import Dict, Tuple

import torch
import torchvision.transforms.v2.functional as F
from einops import rearrange


class DownstreamFlipAndRotateSpace(object):
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
        space_time_h_m: torch.Tensor,
        space_time_m_m: torch.Tensor,
        space_time_l_m: torch.Tensor,
        space_m: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not self.enabled:
            return (
                space_time_h_x,
                space_time_m_x,
                space_time_l_x,
                space_x,
                space_time_h_m,
                space_time_m_m,
                space_time_l_m,
                space_m,
                label,
            )

        space_time_h_x = rearrange(
            space_time_h_x.float(), "h w t c -> t c h w"
        )  # rearrange for transforms
        space_time_m_x = rearrange(space_time_m_x.float(), "h w t c -> t c h w")
        space_time_l_x = rearrange(space_time_l_x.float(), "h w t c -> t c h w")
        space_x = rearrange(space_x.float(), "h w c -> c h w")  # rearrange for transforms
        space_time_h_m = rearrange(space_time_h_m.float(), "h w t c -> t c h w")
        space_time_m_m = rearrange(space_time_m_m.float(), "h w t c -> t c h w")
        space_time_l_m = rearrange(space_time_l_m.float(), "h w t c -> t c h w")
        space_m = rearrange(space_m.float(), "h w c -> c h w")  # rearrange for transforms
        label = torch.unsqueeze(label, dim=0)

        transformation = random.choice(self.transformations)

        space_time_h_x = rearrange(
            transformation(space_time_h_x), "t c h w -> h w t c"
        )  # rearrange back
        space_time_m_x = rearrange(transformation(space_time_m_x), "t c h w -> h w t c")
        space_time_l_x = rearrange(transformation(space_time_l_x), "t c h w -> h w t c")
        space_x = rearrange(transformation(space_x), "c h w -> h w c")  # rearrange back
        space_time_h_m = rearrange(transformation(space_time_h_m), "t c h w -> h w t c")
        space_time_m_m = rearrange(transformation(space_time_m_m), "t c h w -> h w t c")
        space_time_l_m = rearrange(transformation(space_time_l_m), "t c h w -> h w t c")
        space_m = rearrange(transformation(space_m), "c h w -> h w c")  # rearrange back
        label = torch.squeeze(transformation(label))

        return (
            space_time_h_x.half(),
            space_time_m_x.half(),
            space_time_l_x.half(),
            space_x.half(),
            space_time_h_m.half(),
            space_time_m_m.half(),
            space_time_l_m.half(),
            space_m.half(),
            label,
        )


class DownstreamAugmentation(object):
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.flip_and_rotate = DownstreamFlipAndRotateSpace(enabled=self.enabled)

    def apply(
        self,
        space_time_h_x: torch.Tensor,
        space_time_m_x: torch.Tensor,
        space_time_l_x: torch.Tensor,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        static_x: torch.Tensor,
        months: torch.Tensor,
        space_time_h_m: torch.Tensor,
        space_time_m_m: torch.Tensor,
        space_time_l_m: torch.Tensor,
        space_m: torch.Tensor,
        time_m: torch.Tensor,
        static_m: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            space_time_h_x,
            space_time_m_x,
            space_time_l_x,
            space_x,
            space_time_h_m,
            space_time_m_m,
            space_time_l_m,
            space_m,
            label,
        ) = self.flip_and_rotate.apply(
            space_time_h_x,
            space_time_m_x,
            space_time_l_x,
            space_x,
            space_time_h_m,
            space_time_m_m,
            space_time_l_m,
            space_m,
            label,
        )

        return (
            space_time_h_x,
            space_time_m_x,
            space_time_l_x,
            space_x,
            time_x,
            static_x,
            months,
            space_time_h_m,
            space_time_m_m,
            space_time_l_m,
            space_m,
            time_m,
            static_m,
            label,
        )
