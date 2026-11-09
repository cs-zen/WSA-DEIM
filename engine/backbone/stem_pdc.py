import math

import torch
from torch import nn
from torch.nn import functional as F


class CentralPixelDifference(nn.Module):

    def __init__(self, in_channels=3, filters_per_channel=4):
        super().__init__()
        if in_channels < 1 or filters_per_channel < 1:
            raise ValueError("PDC channel counts must be positive")
        self.in_channels = in_channels
        self.out_channels = in_channels * filters_per_channel
        self.weight = nn.Parameter(torch.empty(self.out_channels, 1, 8))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def effective_kernel(self):
        center = -self.weight.sum(dim=-1, keepdim=True)
        return torch.cat((self.weight[..., :4], center, self.weight[..., 4:]), -1).reshape(
            self.out_channels, 1, 3, 3)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            working = x if x.dtype == torch.float64 else x.float()
            kernel = self.effective_kernel().to(dtype=working.dtype)
            return F.conv2d(F.pad(working, (1, 1, 1, 1), mode='replicate'),
                            kernel, groups=self.in_channels)


class PDCResidualStem(nn.Module):

    def __init__(self, spatial, in_channels=3, out_channels=16,
                 filters_per_channel=4, initial_residual_scale=0.1,
                 max_residual_scale=0.5):
        super().__init__()
        if not 0 < initial_residual_scale < max_residual_scale <= 1:
            raise ValueError("Require 0 < initial_residual_scale < max_residual_scale <= 1")
        self.spatial = spatial
        with torch.random.fork_rng(devices=[]):
            self.pdc = CentralPixelDifference(in_channels, filters_per_channel)
            self.pack = nn.PixelUnshuffle(4)
            self.detail_projection = nn.Conv2d(self.pdc.out_channels * 16,
                                               out_channels, 1, bias=False)
            self.detail_norm = nn.BatchNorm2d(out_channels)
        self.max_residual_scale = float(max_residual_scale)
        logit = math.log(initial_residual_scale / (max_residual_scale - initial_residual_scale))
        self.residual_beta = nn.Parameter(torch.full((1, out_channels, 1, 1), logit))

    def residual_scale(self):
        return self.max_residual_scale * self.residual_beta.sigmoid()

    def detail_features(self, x):
        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError("PDC stem requires height and width divisible by four")
        return self.detail_norm(self.detail_projection(self.pack(self.pdc(x))))

    def forward(self, x):
        spatial = self.spatial(x)
        detail = self.detail_features(x).to(dtype=spatial.dtype)
        return spatial + self.residual_scale().to(dtype=spatial.dtype) * detail
