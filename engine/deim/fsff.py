# Modified for the WSA-DEIM full-model code release (2026).

import torch
from torch import nn

from ..core import register
from .hybrid_encoder import ConvNormLayer
from .sni import SNI

__all__ = ['GLCAM', 'FSFF']


@register()
class GLCAM(nn.Module):

    def __init__(self, channels=256, act='silu'):
        super().__init__()
        if (not isinstance(channels, int) or isinstance(channels, bool)
                or channels <= 0 or channels % 8):
            raise ValueError('GLCAM channels must be a positive multiple of 8')
        self.channels = channels
        self.hidden_channels = hidden = channels // 2
        self.quarter_channels = quarter = hidden // 4
        self.local_branch = nn.Sequential(
            ConvNormLayer(channels, hidden, 1, 1, act=act),
            ConvNormLayer(hidden, hidden, 3, 1, g=hidden, act=act),
            ConvNormLayer(hidden, hidden, 1, 1, act=None),
        )
        self.global_projection = ConvNormLayer(channels, hidden, 1, 1, act=act)
        self.frequency_depthwise = ConvNormLayer(
            2 * quarter, 2 * quarter, 3, 1, g=2 * quarter, act=act,
        )
        self.multi_scale_pre = ConvNormLayer(
            quarter, quarter, 3, 1, g=quarter, act=act,
        )
        self.multi_scale_branches = nn.ModuleList([
            ConvNormLayer(quarter, quarter, kernel, 1, g=quarter, act=act)
            for kernel in (5, 7, 11)
        ])
        self.output_projection = ConvNormLayer(hidden, channels, 1, 1, act=None)
        self.gate = nn.Sigmoid()

    def _frequency_forward(self, x):
        original_dtype = x.dtype
        fft_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        spectrum = torch.fft.fft2(fft_input, dim=(-2, -1), norm='ortho')
        cartesian = torch.cat((spectrum.real, spectrum.imag), dim=1)
        cartesian = cartesian.to(self.frequency_depthwise.conv.weight.dtype)
        filtered = self.frequency_depthwise(cartesian)
        real, imag = filtered.chunk(2, dim=1)
        if real.dtype in (torch.float16, torch.bfloat16):
            real, imag = real.float(), imag.float()
        reconstructed = torch.fft.ifft2(
            torch.complex(real, imag), dim=(-2, -1), norm='ortho',
        ).real
        return reconstructed.to(dtype=original_dtype)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'GLCAM expects NCHW input with {self.channels} channels')
        local = self.local_branch(x)
        residual = self.global_projection(x)
        frequency, multi_scale, identity = torch.split(
            residual,
            (self.quarter_channels, self.quarter_channels, self.hidden_channels // 2),
            dim=1,
        )
        frequency = self._frequency_forward(frequency)
        shared = self.multi_scale_pre(multi_scale)
        multi_scale = shared
        for branch in self.multi_scale_branches:
            multi_scale = multi_scale + branch(shared)
        global_response = torch.cat((frequency, multi_scale, identity), dim=1) + residual
        return self.gate(self.output_projection(local + global_response))


@register()
class FSFF(nn.Module):

    def __init__(self, channels=256, act='silu'):
        super().__init__()
        self.channels = channels
        self.upsample = SNI(up_factor=2)
        self.attention = GLCAM(channels=channels, act=act)

    def forward(self, shallow, deep):
        if shallow.ndim != 4 or deep.ndim != 4:
            raise ValueError('FSFF expects two NCHW tensors')
        if (shallow.shape[:2] != deep.shape[:2]
                or shallow.shape[1] != self.channels
                or shallow.device != deep.device):
            raise ValueError('FSFF inputs must have matching batch, channels and device')
        if shallow.shape[-2:] != (2 * deep.shape[-2], 2 * deep.shape[-1]):
            raise ValueError('FSFF shallow input must be exactly twice the deep spatial size')
        deep_up = self.upsample(deep)
        mask = self.attention(shallow + deep_up)
        return shallow * mask + deep_up * (1.0 - mask)
