import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT(nn.Module):

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"HaarDWT 要求输入为四维 NCHW 张量，当前为 {x.ndim} 维")

        height, width = x.shape[-2:]
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "HaarDWT 要求输入的高度和宽度均为偶数，"
                f"当前为 {height}x{width}"
            )

        top_left = x[:, :, 0::2, 0::2]
        top_right = x[:, :, 0::2, 1::2]
        bottom_left = x[:, :, 1::2, 0::2]
        bottom_right = x[:, :, 1::2, 1::2]

        ll = (top_left + top_right + bottom_left + bottom_right) * 0.5
        lh = (top_left + top_right - bottom_left - bottom_right) * 0.5
        hl = (top_left - top_right + bottom_left - bottom_right) * 0.5
        hh = (top_left - top_right - bottom_left + bottom_right) * 0.5

        fl = ll
        fh = torch.cat((lh, hl, hh), dim=1)
        return fl, fh


class HaarIDWT(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"HaarIDWT 要求输入为四维 NCHW 张量，当前为 {x.ndim} 维"
            )

        channels = x.shape[1]
        if channels <= 0 or channels % 4 != 0:
            raise ValueError(
                "HaarIDWT 要求输入通道数为正整数且可被 4 整除，"
                f"当前为 {channels}"
            )

        ll, lh, hl, hh = x.chunk(4, dim=1)

        top_left = (ll + lh + hl + hh) * 0.5
        top_right = (ll + lh - hl - hh) * 0.5
        bottom_left = (ll - lh + hl - hh) * 0.5
        bottom_right = (ll - lh - hl + hh) * 0.5

        spatial_samples = torch.stack(
            (top_left, top_right, bottom_left, bottom_right),
            dim=2,
        ).flatten(1, 2)
        return F.pixel_shuffle(spatial_samples, upscale_factor=2)
