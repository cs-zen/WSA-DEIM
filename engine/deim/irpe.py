# Modified for the WSA-DEIM code release.
"""Contextual Product iRPE used by the F5 AIFI self-attention.

The relative-position bucket mapping and contextual key formulation are
adapted from Microsoft's official iRPE DETR implementation at commit
4a13c4091e78f9abd2160e7e01c02e48c1cf8fb9 (Apache-2.0).  This local version
keeps only the native PyTorch, shared-head, K-only Product variant required by
``rpe-2.0-product-ctx-1-k``; it does not depend on the optional CUDA extension.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "IRPE_DETR_CONFIG",
    "piecewise_index",
    "build_product_position_index",
    "ContextualProductRPEKey",
    "IRPEMultiheadAttention",
]


IRPE_DETR_CONFIG = "rpe-2.0-product-ctx-1-k"


def piecewise_index(
    relative_position: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    gamma: float = 16.0,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:

    if not torch.is_tensor(relative_position):
        raise TypeError("relative_position must be a torch.Tensor")
    if not (0 < alpha <= beta < gamma):
        raise ValueError(
            "piecewise_index requires 0 < alpha <= beta < gamma, got "
            f"alpha={alpha}, beta={beta}, gamma={gamma}"
        )

    relative_abs = relative_position.abs()
    inside = relative_abs <= alpha
    outside = ~inside

    index = relative_position.clone()
    if index.dtype in (torch.float32, torch.float64):
        index = index.round().to(dtype)
    else:
        index = index.to(dtype)

    outside_position = relative_position[outside]
    outside_abs = relative_abs[outside]
    magnitude = (
        alpha
        + torch.log(outside_abs / alpha)
        / math.log(gamma / alpha)
        * (beta - alpha)
    )
    magnitude = magnitude.round().clip(max=beta)
    compressed = (torch.sign(outside_position) * magnitude).to(dtype)
    index[outside] = compressed

    return index


def build_product_position_index(
    height: int,
    width: int,
    *,
    device: Optional[torch.device] = None,
    alpha: float = 2.0,
    beta: float = 4.0,
    gamma: float = 16.0,
) -> torch.Tensor:

    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")

    beta_int = int(beta)
    axis_bucket_count = 2 * beta_int + 1
    row = torch.arange(height, device=device, dtype=torch.long)
    column = torch.arange(width, device=device, dtype=torch.long)
    row, column = torch.meshgrid(row, column, indexing="ij")
    coordinates = torch.stack((row.reshape(-1), column.reshape(-1)), dim=0)
    relative = coordinates[:, :, None] - coordinates[:, None, :]
    relative = piecewise_index(
        relative,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        dtype=torch.long,
    )
    row_bucket = relative[0] + beta_int
    column_bucket = relative[1] + beta_int
    return (row_bucket * axis_bucket_count + column_bucket).contiguous()


class ContextualProductRPEKey(nn.Module):

    def __init__(
        self,
        head_dim: int,
        ratio: float = 2.0,
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if (
            isinstance(head_dim, bool)
            or not isinstance(head_dim, int)
            or head_dim <= 0
        ):
            raise ValueError(f"head_dim must be a positive integer, got {head_dim!r}")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            raise TypeError(f"ratio must be a real number, got {type(ratio).__name__}")
        if not math.isclose(float(ratio), 2.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{IRPE_DETR_CONFIG} requires ratio=2.0, got {ratio!r}"
            )

        self.head_dim = head_dim
        self.ratio = 2.0
        self.alpha = 2.0
        self.beta = 4.0
        self.gamma = 16.0
        self.axis_bucket_count = 9
        self.num_buckets = 81
        self.shared_head = True
        self.rpe_on = "k"
        self.lookup_table_weight = nn.Parameter(
            torch.zeros(1, head_dim, self.num_buckets)
        )
        self._relative_position_index_cache: dict[
            Tuple[int, int, str, Optional[int]], torch.Tensor
        ] = {}
        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lookup_table_weight)

    def clear_index_cache(self) -> None:
        self._relative_position_index_cache.clear()

    def _apply(self, fn):
        self.clear_index_cache()
        return super()._apply(fn)

    def get_bucket_index(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        device = torch.device(device)
        key = (height, width, device.type, device.index)
        cached = self._relative_position_index_cache.get(key)
        if cached is None:
            cached = build_product_position_index(
                height,
                width,
                device=device,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )
            if len(self._relative_position_index_cache) >= 8:
                self._relative_position_index_cache.pop(
                    next(iter(self._relative_position_index_cache))
                )
            self._relative_position_index_cache[key] = cached
        return cached

    def forward(
        self,
        scaled_query: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if scaled_query.ndim != 4:
            raise ValueError(
                "ContextualProductRPEKey expects [B, heads, H*W, head_dim], "
                f"got shape {tuple(scaled_query.shape)}"
            )
        batch, heads, length, head_dim = scaled_query.shape
        if head_dim != self.head_dim:
            raise ValueError(
                f"scaled query head_dim must be {self.head_dim}, got {head_dim}"
            )
        if length != height * width:
            raise ValueError(
                f"scaled query length {length} does not match {height}x{width}"
            )
        if scaled_query.device != self.lookup_table_weight.device:
            raise ValueError(
                "scaled query and iRPE lookup table must be on the same device, "
                f"got {scaled_query.device} and {self.lookup_table_weight.device}"
            )

        query_to_bucket = torch.matmul(
            scaled_query,
            self.lookup_table_weight,
        )
        bucket_index = self.get_bucket_index(height, width, scaled_query.device)
        gather_index = bucket_index.view(1, 1, length, length).expand(
            batch,
            heads,
            length,
            length,
        )
        return torch.gather(query_to_bucket, dim=-1, index=gather_index)


class IRPEMultiheadAttention(nn.MultiheadAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        *,
        initialize_rpe: bool = True,
    ) -> None:
        if not batch_first:
            raise ValueError("IRPEMultiheadAttention requires batch_first=True")
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        self.rpe_k = ContextualProductRPEKey(
            self.head_dim,
            ratio=2.0,
            initialize=initialize_rpe,
        )

    def reset_rpe_parameters(self) -> None:
        self.rpe_k.reset_parameters()

    @staticmethod
    def _add_attention_mask(
        logits: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, query_length, key_length = logits.shape
        if attn_mask.dtype == torch.uint8:
            raise TypeError("uint8 attention masks are not supported; use bool")
        if attn_mask.ndim == 2:
            if tuple(attn_mask.shape) != (query_length, key_length):
                raise ValueError(
                    "2D attn_mask must have shape "
                    f"{(query_length, key_length)}, got {tuple(attn_mask.shape)}"
                )
            mask = attn_mask.view(1, 1, query_length, key_length)
        elif attn_mask.ndim == 3:
            if tuple(attn_mask.shape) != (
                batch * heads,
                query_length,
                key_length,
            ):
                raise ValueError(
                    "3D attn_mask must have shape "
                    f"{(batch * heads, query_length, key_length)}, "
                    f"got {tuple(attn_mask.shape)}"
                )
            mask = attn_mask.reshape(batch, heads, query_length, key_length)
        else:
            raise ValueError(
                f"attn_mask must be 2D or 3D, got {attn_mask.ndim}D"
            )

        mask = mask.to(device=logits.device)
        if mask.dtype == torch.bool:
            return logits.masked_fill(mask, float("-inf"))
        if not torch.is_floating_point(mask):
            raise TypeError(
                f"attn_mask must be bool or floating point, got {mask.dtype}"
            )
        return logits + mask.to(dtype=logits.dtype)

    @staticmethod
    def _add_key_padding_mask(
        logits: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, _, key_length = logits.shape
        if tuple(key_padding_mask.shape) != (batch, key_length):
            raise ValueError(
                "key_padding_mask must have shape "
                f"{(batch, key_length)}, got {tuple(key_padding_mask.shape)}"
            )
        if key_padding_mask.dtype == torch.uint8:
            raise TypeError("uint8 key padding masks are not supported; use bool")
        mask = key_padding_mask.to(device=logits.device).view(
            batch,
            1,
            1,
            key_length,
        )
        if mask.dtype == torch.bool:
            return logits.masked_fill(mask, float("-inf"))
        if not torch.is_floating_point(mask):
            raise TypeError(
                "key_padding_mask must be bool or floating point, "
                f"got {mask.dtype}"
            )
        return logits + mask.to(dtype=logits.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        *,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                "IRPEMultiheadAttention expects batch-first 3D query/key/value"
            )
        if spatial_shape is None or len(spatial_shape) != 2:
            raise ValueError("spatial_shape=(height, width) is required for iRPE")
        height, width = spatial_shape
        if (
            isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
        ):
            raise ValueError(
                f"spatial_shape must contain positive integers, got {spatial_shape!r}"
            )

        batch, query_length, embed_dim = query.shape
        key_batch, key_length, key_dim = key.shape
        value_batch, value_length, value_dim = value.shape
        if (
            batch != key_batch
            or batch != value_batch
            or query_length != key_length
            or key_length != value_length
            or embed_dim != self.embed_dim
            or key_dim != self.embed_dim
            or value_dim != self.embed_dim
        ):
            raise ValueError(
                "iRPE is defined here for same-width self-attention; got "
                f"query={tuple(query.shape)}, key={tuple(key.shape)}, "
                f"value={tuple(value.shape)}"
            )
        if query_length != height * width:
            raise ValueError(
                f"token length {query_length} does not match {height}x{width}"
            )

        if self.in_proj_weight is None:
            raise RuntimeError("iRPE attention requires packed same-width QKV weights")
        query_weight, key_weight, value_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            query_bias = key_bias = value_bias = None
        else:
            query_bias, key_bias, value_bias = self.in_proj_bias.chunk(3, dim=0)

        projected_query = F.linear(query, query_weight, query_bias)
        projected_key = F.linear(key, key_weight, key_bias)
        projected_value = F.linear(value, value_weight, value_bias)

        projected_query = projected_query.view(
            batch,
            query_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        projected_key = projected_key.view(
            batch,
            key_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        projected_value = projected_value.view(
            batch,
            value_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scaled_query = projected_query * (self.head_dim ** -0.5)
        attention_logits = torch.matmul(
            scaled_query,
            projected_key.transpose(-2, -1),
        )
        attention_logits = attention_logits + self.rpe_k(
            scaled_query,
            height,
            width,
        )

        if is_causal:
            causal_mask = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=attention_logits.device,
            ).triu(diagonal=1)
            attention_logits = self._add_attention_mask(
                attention_logits,
                causal_mask,
            )
        if attn_mask is not None:
            attention_logits = self._add_attention_mask(
                attention_logits,
                attn_mask,
            )
        if key_padding_mask is not None:
            attention_logits = self._add_key_padding_mask(
                attention_logits,
                key_padding_mask,
            )

        attention_weights = F.softmax(attention_logits, dim=-1)
        dropped_weights = F.dropout(
            attention_weights,
            p=self.dropout,
            training=self.training,
        )
        attention_output = torch.matmul(dropped_weights, projected_value)
        attention_output = attention_output.transpose(1, 2).reshape(
            batch,
            query_length,
            self.embed_dim,
        )
        attention_output = self.out_proj(attention_output)

        if not need_weights:
            returned_weights = None
        elif average_attn_weights:
            returned_weights = dropped_weights.mean(dim=1)
        else:
            returned_weights = dropped_weights
        return attention_output, returned_weights
