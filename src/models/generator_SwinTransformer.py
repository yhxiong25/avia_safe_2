"""Swin-style generator for ERA5 cloud forecasting.

Interface matches ``ConvGRUUNet3DGenerator`` and ``SimVP3DGenerator``:
    x     : [B, input_steps * input_vars, D, H, W]
    y_hat : [B, forecast_steps * output_vars, D, H, W]

Data flow
---------
1. A shared 3x3x3 stem reads each historical time step. The last frame is kept
   as a full-resolution skip so sparse cloud cores are not rebuilt only from
   coarse tokens.
2. Latitude and longitude are patched by a stride-(1, 2, 2) convolution.
   Pressure depth stays 13.
3. Shifted-window attention mixes the horizontal field. A vertical attention
   then mixes the 13 pressure levels at each horizontal location.
4. Twelve learnable lead queries cross-attend to the historical latent states.
   Leads are produced together, not by a recurrent rollout.
5. The latent is interpolated back to the original grid, fused with the last
   observed frame, and projected to the output variables.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in reversed(range(1, max_groups + 1)):
        if channels % groups == 0:
            return groups
    return 1


def _pad_to_multiple(size: int, multiple: int) -> int:
    return (multiple - size % multiple) % multiple


def window_partition(x: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
    """[B, H, W, C] -> [B * n_windows, window_h * window_w, C]."""
    batch, height, width, channels = x.shape
    window_h, window_w = window_size
    x = x.view(batch, height // window_h, window_h, width // window_w, window_w, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_h * window_w, channels)


def window_reverse(windows: torch.Tensor, window_size: tuple[int, int], height: int, width: int) -> torch.Tensor:
    """[B * n_windows, window_h * window_w, C] -> [B, H, W, C]."""
    window_h, window_w = window_size
    channels = windows.shape[-1]
    batch = int(windows.shape[0] / (height * width / window_h / window_w))
    x = windows.view(batch, height // window_h, width // window_w, window_h, window_w, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, channels)


class WindowAttention(nn.Module):
    """Multi-head self-attention inside one horizontal window, with relative bias."""

    def __init__(self, dim: int, window_size: tuple[int, int], num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.window_size = window_size
        window_h, window_w = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_h - 1) * (2 * window_w - 1), num_heads)
        )

        coords_h = torch.arange(window_h)
        coords_w = torch.arange(window_w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_h - 1
        relative[:, :, 1] += window_w - 1
        relative[:, :, 0] *= 2 * window_w - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query * self.scale) @ key.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(length, length, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch // num_windows, num_windows, self.num_heads, length, length)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, length, length)
        attn = attn.softmax(dim=-1)
        out = (attn @ value).transpose(1, 2).reshape(batch, length, channels)
        return self.proj(out)


class SwinBlock(nn.Module):
    """One Swin block. Even blocks use a regular window; odd blocks shift it."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _attention_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        window_h, window_w = self.window_size
        shift_h, shift_w = self.shift_size
        image_mask = torch.zeros((1, height, width, 1), device=device)
        height_slices = (slice(0, -window_h), slice(-window_h, -shift_h), slice(-shift_h, None))
        width_slices = (slice(0, -window_w), slice(-window_w, -shift_w), slice(-shift_w, None))
        region = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[:, height_slice, width_slice, :] = region
                region += 1
        windows = window_partition(image_mask, self.window_size).squeeze(-1)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, _ = x.shape
        shortcut = x
        x = self.norm1(x)
        shifted = self.shift_size[0] > 0 or self.shift_size[1] > 0
        if shifted:
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            mask = self._attention_mask(height, width, x.device)
        else:
            mask = None
        windows = window_partition(x, self.window_size)
        windows = self.attn(windows, mask=mask)
        x = window_reverse(windows, self.window_size, height, width)
        if shifted:
            x = torch.roll(x, shifts=self.shift_size, dims=(1, 2))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class VerticalAttention(nn.Module):
    """Attention across pressure levels at each horizontal location."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).reshape(batch * height * width, depth, channels)
        shortcut = tokens
        tokens = self.norm(tokens)
        qkv = self.qkv(tokens).reshape(tokens.shape[0], depth, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query * self.scale) @ key.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ value).transpose(1, 2).reshape(tokens.shape[0], depth, channels)
        out = self.proj(out)
        out = (shortcut + out).reshape(batch, height, width, depth, channels)
        return out.permute(0, 4, 3, 1, 2).contiguous()


class LeadQueryTranslator(nn.Module):
    """Map a short history to every lead with learnable queries.

    At each pressure-latitude-longitude location the keys are the historical
    latent states. Each forecast lead has its own query, so the 12 leads are
    decoded together and can specialize.
    """

    def __init__(self, dim: int, forecast_steps: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.forecast_steps = forecast_steps
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lead_query = nn.Parameter(torch.empty(1, forecast_steps, dim))
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.trunc_normal_(self.lead_query, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, depth, height, width = z.shape
        history = z.permute(0, 3, 4, 5, 1, 2).reshape(batch * depth * height * width, steps, channels)
        persistence = history[:, -1:, :]
        query = self.lead_query.expand(history.shape[0], -1, -1)
        query = self.q_proj(self.norm_q(query + persistence))
        key_value = self.kv_proj(self.norm_kv(history))
        key, value = key_value.chunk(2, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            count, length, _ = tensor.shape
            tensor = tensor.reshape(count, length, self.num_heads, channels // self.num_heads)
            return tensor.permute(0, 2, 1, 3)

        query, key, value = split_heads(query), split_heads(key), split_heads(value)
        attn = (query * self.scale) @ key.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ value).transpose(1, 2).reshape(history.shape[0], self.forecast_steps, channels)
        out = persistence + self.out_proj(out)
        out = out + self.mlp(self.norm_mlp(out))
        out = out.reshape(batch, depth, height, width, self.forecast_steps, channels)
        return out.permute(0, 4, 5, 1, 2, 3).contiguous()


class SwinTransformer3DGenerator(nn.Module):
    """Horizontal Swin encoder plus lead-query decoder for ERA5 cloud fields.

    ``base_channels`` follows the training script. The attention width is
    ``4 * base_channels``, so ``--base_channels_g 64`` builds a 256-d Swin.
    """

    def __init__(
        self,
        in_channels: int = 18,
        out_channels: int = 48,
        base_channels: int = 32,
        input_steps: int = 2,
        input_vars: int = 9,
        output_vars: int = 4,
        forecast_steps: int | None = None,
        depth: int = 8,
        num_heads: int = 4,
        window_size: tuple[int, int] = (7, 6),
        patch_size: tuple[int, int] = (2, 2),
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if in_channels != input_steps * input_vars:
            raise ValueError(
                f"in_channels={in_channels} is inconsistent with "
                f"input_steps*input_vars={input_steps * input_vars}."
            )
        if forecast_steps is None:
            if out_channels % output_vars != 0:
                raise ValueError("out_channels must be divisible by output_vars.")
            forecast_steps = out_channels // output_vars
        if forecast_steps * output_vars != out_channels:
            raise ValueError(
                f"forecast_steps * output_vars must equal out_channels. "
                f"Got {forecast_steps} * {output_vars} != {out_channels}."
            )
        dim = base_channels * 4
        if dim % num_heads != 0:
            raise ValueError(f"embed dim={dim} must be divisible by num_heads={num_heads}.")
        if depth < 1:
            raise ValueError("depth must be >= 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_steps = input_steps
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.forecast_steps = forecast_steps
        self.window_size = window_size
        self.patch_size = patch_size

        self.stem = nn.Sequential(
            nn.Conv3d(input_vars, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(dim), dim),
            nn.SiLU(inplace=True),
        )
        self.patch_embed = nn.Conv3d(
            dim,
            dim,
            kernel_size=(1, patch_size[0], patch_size[1]),
            stride=(1, patch_size[0], patch_size[1]),
        )
        blocks = []
        for index in range(depth):
            shift = (0, 0) if index % 2 == 0 else (window_size[0] // 2, window_size[1] // 2)
            blocks.append(SwinBlock(dim, num_heads, window_size, shift_size=shift, mlp_ratio=mlp_ratio))
        self.blocks = nn.ModuleList(blocks)
        self.vertical = VerticalAttention(dim, num_heads)
        self.translator = LeadQueryTranslator(dim, forecast_steps, num_heads, mlp_ratio=mlp_ratio)
        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(dim), dim),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(dim), dim),
            nn.SiLU(inplace=True),
            nn.Conv3d(dim, output_vars, kernel_size=1),
        )

    def _encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """[B, C, D, H_pad, W_pad] -> [B, C, D, h, w]."""
        batch, _, depth, height, width = frame.shape
        x = self.patch_embed(frame)
        _, _, _, latent_h, latent_w = x.shape
        if latent_h % self.window_size[0] != 0 or latent_w % self.window_size[1] != 0:
            raise ValueError(
                f"Latent map {(latent_h, latent_w)} is not divisible by window {self.window_size}."
            )
        x = x.permute(0, 2, 3, 4, 1).reshape(batch * depth, latent_h, latent_w, -1)
        for block in self.blocks:
            x = block(x)
        channels = x.shape[-1]
        x = x.reshape(batch, depth, latent_h, latent_w, channels).permute(0, 4, 1, 2, 3).contiguous()
        return self.vertical(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected input channel {self.in_channels}, got {channels}.")

        x = x.view(batch, self.input_steps, self.input_vars, depth, height, width)
        encoded = []
        last_skip = None
        for step in range(self.input_steps):
            stem = self.stem(x[:, step])
            last_skip = stem
            pad_h = _pad_to_multiple(height, self.patch_size[0] * self.window_size[0])
            pad_w = _pad_to_multiple(width, self.patch_size[1] * self.window_size[1])
            if pad_h or pad_w:
                stem = F.pad(stem, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
            encoded.append(self._encode_frame(stem))

        latent = torch.stack(encoded, dim=1)
        future = self.translator(latent)
        lead_batch = batch * self.forecast_steps
        future = future.reshape(lead_batch, future.shape[2], depth, future.shape[4], future.shape[5])
        future = F.interpolate(future, size=(depth, height + pad_h, width + pad_w), mode="trilinear", align_corners=False)
        future = future[..., :height, :width]
        future = self.refine(future)

        skip = last_skip.unsqueeze(1).expand(-1, self.forecast_steps, -1, -1, -1, -1)
        skip = skip.reshape(lead_batch, last_skip.shape[1], depth, height, width)
        y = self.fuse(torch.cat([future, skip], dim=1))
        return y.reshape(batch, self.forecast_steps * self.output_vars, depth, height, width)
