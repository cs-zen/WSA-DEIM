import torch
import torch.nn as nn

from ..core import register


__all__ = ['SNI']


@register()
class SNI(nn.Module):

    def __init__(self, up_factor: int = 2) -> None:
        super().__init__()
        if (
            not isinstance(up_factor, int)
            or isinstance(up_factor, bool)
            or up_factor <= 0
        ):
            raise ValueError(
                f'up_factor must be a positive integer, got {up_factor!r}'
            )

        self.up_factor = up_factor
        self.up = nn.Upsample(
            scale_factor=up_factor,
            mode='nearest',
        )
        self.alpha = 1.0 / (up_factor ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.alpha * x
        return x
