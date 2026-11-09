# Modified for the WSA-DEIM full-model code release (2026).

"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_activation
from .sni import SNI
from .irpe import IRPE_DETR_CONFIG, IRPEMultiheadAttention
from ..backbone.haar import HaarDWT, HaarIDWT
from ..core import register

class HFEM(nn.Module):

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, int)
            or channels <= 0
            or channels % 2 != 0
        ):
            raise ValueError(
                'channels must be a positive even integer, '
                f'got {channels!r}'
            )

        self.channels = channels
        self.branch_channels = channels // 2
        self.pool_branch = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=3,
                stride=1,
                padding=1,
                count_include_pad=True,
            ),
            nn.Conv2d(
                self.branch_channels,
                self.branch_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.SiLU(inplace=True),
        )
        self.conv_branch = nn.Sequential(
            nn.Conv2d(
                self.branch_channels,
                self.branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.SiLU(inplace=True),
        )
        self.fuse_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f'HFEM expects a four-dimensional NCHW tensor, got {x.ndim}D'
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                'HFEM input channels must match the configured width '
                f'{self.channels}, got {x.shape[1]}'
            )

        pooled_input, convolved_input = torch.split(
            x,
            (self.branch_channels, self.branch_channels),
            dim=1,
        )
        averaged = self.pool_branch[0](pooled_input)
        detail = pooled_input - averaged
        pooled = self.pool_branch[2](
            pooled_input + self.pool_branch[1](detail)
        )
        convolved = self.conv_branch(convolved_input)
        fused = self.fuse_conv(torch.cat((pooled, convolved), dim=1))
        return x + fused


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size-1)//2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            groups=g,
            padding=padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 irpe_config=None,
                 initialize_irpe=True):
        super().__init__()
        self.normalize_before = normalize_before
        if irpe_config not in (None, IRPE_DETR_CONFIG):
            raise ValueError(
                'TransformerEncoderLayer irpe_config must be None or '
                f'{IRPE_DETR_CONFIG!r}, got {irpe_config!r}'
            )
        self.irpe_config = irpe_config
        if self.irpe_config is None:
            self.self_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout,
                batch_first=True,
            )
        else:
            self.self_attn = IRPEMultiheadAttention(
                d_model,
                nhead,
                dropout,
                batch_first=True,
                initialize_rpe=False,
            )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        if self.irpe_config is not None and initialize_irpe:
            self.self_attn.reset_rpe_parameters()

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        src,
        src_mask=None,
        pos_embed=None,
        spatial_shape=None,
    ) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        attention_kwargs = {
            'value': src,
            'attn_mask': src_mask,
        }
        if self.irpe_config is not None:
            attention_kwargs['spatial_shape'] = spatial_shape
        src, _ = self.self_attn(q, k, **attention_kwargs)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        src_mask=None,
        pos_embed=None,
        spatial_shape=None,
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                pos_embed=pos_embed,
                spatial_shape=spatial_shape,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


@register()
class Haar_Spatial_Alignment(nn.Module):

    def __init__(self, channels=256, reduction=16):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}')
        if reduction <= 0:
            raise ValueError(f'reduction must be positive, got {reduction}')
        self.channels = channels
        self.reduction = reduction
        joint_channels = 5 * channels
        bottleneck_channels = max(1, joint_channels // reduction)
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_fc1 = nn.Linear(joint_channels, bottleneck_channels)
        self.channel_act = nn.ReLU(inplace=True)
        self.channel_fc2 = nn.Linear(bottleneck_channels, joint_channels)
        self.sigmoid = nn.Sigmoid()

    def _validate_inputs(self, large: torch.Tensor, small: torch.Tensor) -> None:
        if large.ndim != 4 or small.ndim != 4:
            raise ValueError(
                'Haar_Spatial_Alignment expects two NCHW tensors, '
                f'got {large.ndim}D and {small.ndim}D inputs'
            )
        if large.shape[0] != small.shape[0]:
            raise ValueError(
                'Haar_Spatial_Alignment requires matching batch sizes, '
                f'got {large.shape[0]} and {small.shape[0]}'
            )
        if large.shape[1] != small.shape[1]:
            raise ValueError(
                'Haar_Spatial_Alignment requires matching channel counts, '
                f'got {large.shape[1]} and {small.shape[1]}'
            )
        if large.shape[1] != self.channels:
            raise ValueError(
                'Haar_Spatial_Alignment input channels must match the configured '
                f'channels={self.channels}, got {large.shape[1]}'
            )
        expected_large_size = (small.shape[-2] * 2, small.shape[-1] * 2)
        if large.shape[-2:] != expected_large_size:
            raise ValueError(
                'Haar_Spatial_Alignment requires the large feature to be '
                'exactly twice the small feature spatially, '
                f'got {tuple(large.shape[-2:])} and {tuple(small.shape[-2:])}'
            )

    def forward(
        self,
        large: torch.Tensor,
        small: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(large, small)

        low_frequency, high_frequency = self.dwt(large)
        wavelet = torch.cat((low_frequency, high_frequency), dim=1)
        joint = torch.cat((wavelet, small), dim=1)
        descriptor = self.gap(joint).flatten(1)
        descriptor = self.channel_fc1(descriptor)
        descriptor = self.channel_act(descriptor)
        descriptor = self.channel_fc2(descriptor)
        descriptor = descriptor.reshape(descriptor.shape[0], -1, 1, 1)
        gates = self.sigmoid(descriptor)
        wavelet_gate, small_gate = torch.split(
            gates,
            (wavelet.shape[1], small.shape[1]),
            dim=1,
        )

        aligned_large = self.idwt(wavelet * wavelet_gate)
        aligned_small = small * small_gate
        return aligned_large, aligned_small


@register()
class MultiScaleDWEnhancement(nn.Module):

    def __init__(self, channels=256):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}')
        self.channels = channels
        self.depthwise_convs = nn.ModuleList([
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            )
            for kernel_size in (3, 5, 7)
        ])
        self.weight_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.weight_norm = nn.BatchNorm2d(channels)
        self.weight_gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                'MultiScaleDWEnhancement expects an NCHW tensor, '
                f'got a {x.ndim}D input'
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                'MultiScaleDWEnhancement input channels must match the '
                f'configured channels={self.channels}, got {x.shape[1]}'
            )

        response_3x3 = self.depthwise_convs[0](x)
        response_5x5 = self.depthwise_convs[1](response_3x3)
        response_7x7 = self.depthwise_convs[2](response_3x3)
        multi_scale = response_3x3 + response_5x5 + response_7x7
        weight = self.weight_gate(
            self.weight_norm(self.weight_conv(multi_scale))
        )
        return x * (1.0 + weight)


@register()
class UpsampleFusion(nn.Module):

    def __init__(self, channels=256):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}')
        self.channels = channels
        self.upsample = SNI(up_factor=2)
        self.fuse_conv = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.fuse_norm = nn.BatchNorm2d(channels)
        self.fuse_act = nn.SiLU(inplace=True)

    def _validate_inputs(
        self,
        large: torch.Tensor,
        small: torch.Tensor,
    ) -> None:
        if large.ndim != 4 or small.ndim != 4:
            raise ValueError(
                'UpsampleFusion expects two NCHW tensors, '
                f'got {large.ndim}D and {small.ndim}D inputs'
            )
        if large.shape[0] != small.shape[0]:
            raise ValueError(
                'UpsampleFusion requires matching batch sizes, '
                f'got {large.shape[0]} and {small.shape[0]}'
            )
        if large.shape[1] != small.shape[1]:
            raise ValueError(
                'UpsampleFusion requires matching channel counts, '
                f'got {large.shape[1]} and {small.shape[1]}'
            )
        if large.shape[1] != self.channels:
            raise ValueError(
                'UpsampleFusion input channels must match the configured '
                f'channels={self.channels}, got {large.shape[1]}'
            )
        expected_large_size = (small.shape[-2] * 2, small.shape[-1] * 2)
        if large.shape[-2:] != expected_large_size:
            raise ValueError(
                'UpsampleFusion requires large to be exactly twice small '
                'spatially, '
                f'got {tuple(large.shape[-2:])} and '
                f'{tuple(small.shape[-2:])}'
            )

    def forward(
        self,
        large: torch.Tensor,
        small: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(large, small)

        small_upsampled = self.upsample(small)
        if small_upsampled.shape[-2:] != large.shape[-2:]:
            raise RuntimeError(
                'UpsampleFusion upsampler produced an unexpected spatial size: '
                f'{tuple(small_upsampled.shape[-2:])} versus '
                f'{tuple(large.shape[-2:])}'
            )
        fused = torch.cat((large, small_upsampled), dim=1)
        return self.fuse_act(self.fuse_norm(self.fuse_conv(fused)))


@register()
class DownsampleFusion(nn.Module):

    def __init__(self, channels=256):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}')
        self.channels = channels
        self.downsample = SCDown(channels, channels, 3, 2)
        self.fuse_conv = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.fuse_norm = nn.BatchNorm2d(channels)
        self.fuse_act = nn.SiLU(inplace=True)

    def _validate_inputs(
        self,
        large: torch.Tensor,
        small: torch.Tensor,
    ) -> None:
        if large.ndim != 4 or small.ndim != 4:
            raise ValueError(
                'DownsampleFusion expects two NCHW tensors, '
                f'got {large.ndim}D and {small.ndim}D inputs'
            )
        if large.shape[0] != small.shape[0]:
            raise ValueError(
                'DownsampleFusion requires matching batch sizes, '
                f'got {large.shape[0]} and {small.shape[0]}'
            )
        if large.shape[1] != small.shape[1]:
            raise ValueError(
                'DownsampleFusion requires matching channel counts, '
                f'got {large.shape[1]} and {small.shape[1]}'
            )
        if large.shape[1] != self.channels:
            raise ValueError(
                'DownsampleFusion input channels must match the configured '
                f'channels={self.channels}, got {large.shape[1]}'
            )
        expected_large_size = (small.shape[-2] * 2, small.shape[-1] * 2)
        if large.shape[-2:] != expected_large_size:
            raise ValueError(
                'DownsampleFusion requires large to be exactly twice small '
                f'spatially, got {tuple(large.shape[-2:])} and '
                f'{tuple(small.shape[-2:])}'
            )

    def forward(
        self,
        large: torch.Tensor,
        small: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(large, small)

        large_downsampled = self.downsample(large)
        if large_downsampled.shape[-2:] != small.shape[-2:]:
            raise RuntimeError(
                'DownsampleFusion SCDown produced an unexpected spatial size: '
                f'{tuple(large_downsampled.shape[-2:])} versus '
                f'{tuple(small.shape[-2:])}'
        )
        fused = torch.cat((large_downsampled, small), dim=1)
        return self.fuse_act(self.fuse_norm(self.fuse_conv(fused)))


@register()
class HaarFPN(nn.Module):

    __inject__ = ['post_fusion']

    def __init__(self, num_levels=4, hidden_dim=256, act='silu',
                 alignment_reduction=16, post_fusion=None):
        super().__init__()
        from .fsff import FSFF
        if num_levels != 4 or post_fusion is None:
            raise ValueError('The full SACFF neck requires four levels and MSCA')
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.alignments = nn.ModuleList([
            Haar_Spatial_Alignment(hidden_dim, alignment_reduction) for _ in range(2)
        ])
        self.lateral_convs = nn.ModuleList([
            ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1) for _ in range(3)
        ])
        self.upsample_fusions = nn.ModuleList([UpsampleFusion(hidden_dim) for _ in range(2)])
        self.fsff = FSFF(hidden_dim, act=act)
        self.post_fusion_blocks = nn.ModuleList([copy.deepcopy(post_fusion) for _ in range(3)])

    def _forward_lateral(self, edge_idx, feature):
        lateral = self.lateral_convs[edge_idx]
        if feature.is_cuda and torch.is_autocast_enabled():
            with torch.autocast(device_type='cuda', enabled=False):
                projected = lateral(feature.float())
            return projected.to(dtype=feature.dtype)
        return lateral(feature)

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError('SACFF expects P2, P3, P4 and P5')
        current, coarsest_output = feats[-1], feats[-1]
        inner_outs = []
        for edge_idx, feat_idx in enumerate((3, 2, 1)):
            small = self._forward_lateral(edge_idx, current)
            large = feats[feat_idx - 1]
            if edge_idx == 2:
                fused = self.fsff(large, small)
            else:
                aligned_large, aligned_small = self.alignments[edge_idx](large, small)
                if edge_idx == 0:
                    coarsest_output = aligned_small
                fused = self.upsample_fusions[edge_idx](aligned_large, aligned_small)
            current = self.post_fusion_blocks[edge_idx](fused)
            inner_outs.insert(0, current)
        return inner_outs + [coarsest_output]


@register()
class HaarPAN(nn.Module):

    __inject__ = ['post_fusion']

    def __init__(self, num_levels=4, hidden_dim=256, post_fusion=None):
        super().__init__()
        if num_levels != 4 or post_fusion is None:
            raise ValueError('The full SACFF neck requires four levels and MSCA')
        self.num_levels, self.hidden_dim = num_levels, hidden_dim
        self.downsample_fusions = nn.ModuleList([DownsampleFusion(hidden_dim) for _ in range(3)])
        self.post_fusion_blocks = nn.ModuleList([copy.deepcopy(post_fusion) for _ in range(3)])

    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError('SACFF expects four feature levels')
        outs = [feats[0]]
        for i in range(3):
            fused = self.downsample_fusions[i](outs[-1], feats[i + 1])
            outs.append(self.post_fusion_blocks[i](fused))
        return outs


@register()
class HybridEncoder(nn.Module):

    __share__ = ['eval_spatial_size']
    __inject__ = ['fpn', 'pan']

    def __init__(self, in_channels=(64, 256, 512, 1024), feat_strides=(4, 8, 16, 32),
                 hidden_dim=256, nhead=8, dim_feedforward=1024, dropout=0.0,
                 enc_act='gelu', pe_temperature=10000, eval_spatial_size=None,
                 fpn=None, pan=None):
        super().__init__()
        if list(in_channels) != [64, 256, 512, 1024] or list(feat_strides) != [4, 8, 16, 32]:
            raise ValueError('WSA-DEIM requires the four HGNetv2-B0 feature levels')
        if hidden_dim != 256 or fpn is None or pan is None:
            raise ValueError('WSA-DEIM requires width 256 and the complete SACFF neck')
        self.in_channels, self.feat_strides = list(in_channels), list(feat_strides)
        self.hidden_dim = hidden_dim
        self.out_channels, self.out_strides = [hidden_dim] * 4, list(feat_strides)
        self.f5_branch_channels = (64, 128, 64)
        self.pe_temperature, self.eval_spatial_size = pe_temperature, eval_spatial_size
        self.input_proj = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(ch, hidden_dim, 1, bias=False)),
                ('norm', nn.BatchNorm2d(hidden_dim)),
            ])) for ch in in_channels
        ])
        layer = TransformerEncoderLayer(128, nhead, dim_feedforward, dropout,
                                        enc_act, irpe_config=IRPE_DETR_CONFIG,
                                        initialize_irpe=False)
        self.encoder = nn.ModuleList([TransformerEncoder(copy.deepcopy(layer), 1)])
        if eval_spatial_size:
            self.pos_embed3 = self.build_2d_sincos_position_embedding(
                eval_spatial_size[1] // 32, eval_spatial_size[0] // 32, 128, pe_temperature)
        self.f5_hfem = HFEM(64)
        self.fpn, self.pan = fpn, pan
        for layer in self.encoder[0].layers:
            layer.self_attn.reset_rpe_parameters()

    @staticmethod
    def build_2d_sincos_position_embedding(
        width,
        height,
        embed_dim=256,
        temperature=10000.,
    ):
        grid_width = torch.arange(int(width), dtype=torch.float32)
        grid_height = torch.arange(int(height), dtype=torch.float32)
        grid_width, grid_height = torch.meshgrid(
            grid_width,
            grid_height,
            indexing='ij',
        )
        if embed_dim % 4 != 0:
            raise ValueError(
                f'embed_dim must be divisible by 4, got {embed_dim}'
            )
        position_dim = embed_dim // 4
        omega = torch.arange(position_dim, dtype=torch.float32) / position_dim
        omega = 1. / (temperature ** omega)
        output_width = grid_width.flatten()[..., None] @ omega[None]
        output_height = grid_height.flatten()[..., None] @ omega[None]
        return torch.cat(
            (
                output_width.sin(),
                output_width.cos(),
                output_height.sin(),
                output_height.cos(),
            ),
            dim=1,
        )[None, :, :]


    def forward(self, feats):
        if len(feats) != 4:
            raise ValueError('WSA-DEIM requires four backbone outputs')
        projected = [projection(feature) for projection, feature in zip(self.input_proj, feats)]
        identity, attention, detail = projected[-1].split(self.f5_branch_channels, dim=1)
        h, w = attention.shape[-2:]
        tokens = attention.flatten(2).permute(0, 2, 1)
        if self.training or self.eval_spatial_size is None:
            position = self.build_2d_sincos_position_embedding(w, h, 128, self.pe_temperature)
        else:
            position = self.pos_embed3
        encoded = self.encoder[0](tokens, pos_embed=position.to(tokens), spatial_shape=(h, w))
        encoded = encoded.permute(0, 2, 1).reshape(-1, 128, h, w).contiguous()
        projected[-1] = torch.cat((identity, encoded, self.f5_hfem(detail)), dim=1)
        return self.pan(self.fpn(projected))
