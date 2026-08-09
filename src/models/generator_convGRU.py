"""ConvGRU-based generator for ERA5 cloud-variable forecasting.

This file is designed as a drop-in alternative to ``UNet3DGenerator`` in
``src.models.generator``.

Task tensor convention
----------------------
Input to the model is already flattened by the Dataset/DataLoader:
    x: [B, 18, D, H, W]
       18 = input_steps(2) * input_variables(9)
       D  = pressure levels, normally 13
       H  = latitude, normally 41
       W  = longitude, normally 71

Output from the model keeps the same convention used by the current training
code:
    y_hat: [B, out_channels, D, H, W]
           out_channels = forecast_steps * output_variables
           normally forecast_steps * 4 cloud variables.

Main idea
---------
The current U-Net treats the two historical times as ordinary channels.
This ConvGRU generator explicitly restores the time dimension, encodes each
historical time step with a shared 3D encoder, updates a ConvGRU hidden state
in latent space, and then evolves that hidden state to generate future lead
steps sequentially.

Architecture
------------
Past encoder:
    x_t -> 3D U-Net encoder -> bottleneck feature b_t

Temporal memory:
    h_t = ConvGRUCell3D(b_t, h_{t-1})

Future rollout:
    h_{t+k} = ConvGRUCell3D(phi(h_{t+k-1}), h_{t+k-1})
    y_{t+k} = shared 3D U-Net decoder(h_{t+k}, latest skips)

The pressure-level dimension is kept unchanged during pooling; only latitude
and longitude are downsampled, matching the original anisotropic 3D U-Net.

TODO
1，需要生成一个batch的样本，然后ide里面，一行行运行一下ConvGRU的处理流程，学习ConvGRU的逻辑和原理，在去阅读相关资料，掌握RNN家族的模型

"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small building blocks, kept self-contained so this file can be placed directly
# under src/models/ without requiring changes to generator.py.
# -----------------------------------------------------------------------------

def _group_count(channels: int, max_groups: int = 8) -> int:
    """Return the largest group number <= max_groups that divides channels."""
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
    """Two 3D convolutions with GroupNorm and SiLU.

    GroupNorm is used instead of BatchNorm because this task often uses small
    batch sizes and 3D tensors. It is therefore less sensitive to per-GPU batch
    size, especially in DDP training.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock3D(nn.Module):
    """Downsample only latitude-longitude; keep pressure-level depth unchanged."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    """Upsample latitude-longitude to the skip feature size and fuse."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # ``size=skip.shape[2:]`` guarantees exact recovery of odd H/W sizes
        # such as 41 x 71 after repeated pooling.
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -----------------------------------------------------------------------------
# ConvGRU core
# -----------------------------------------------------------------------------

class ConvGRUCell3D(nn.Module):
    """3D convolutional GRU cell.

    Difference from ordinary GRU:
        Ordinary GRU uses linear layers on vectors.
        ConvGRU replaces linear layers with Conv3d, so the hidden state remains
        a structured field: [B, C, D, H, W].

    Equations:
        z = sigmoid(W_z * [x, h])              update gate
        r = sigmoid(W_r * [x, h])              reset gate
        h_tilde = tanh(W_h * [x, r * h])       candidate state
        h_new = (1 - z) * h + z * h_tilde

    Here * denotes 3D convolution, not matrix multiplication.
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        gate_channels = input_channels + hidden_channels
        self.conv_z = nn.Conv3d(gate_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_r = nn.Conv3d(gate_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_h = nn.Conv3d(gate_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(
                x.shape[0],
                self.hidden_channels,
                x.shape[2],
                x.shape[3],
                x.shape[4],
                dtype=x.dtype,
                device=x.device,
            )

        xh = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))

        xrh = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_h(xrh))

        # z controls how much of the new candidate state replaces old memory.
        h_new = (1.0 - z) * h + z * h_tilde
        return h_new


# -----------------------------------------------------------------------------
# ConvGRU-U-Net generator
# -----------------------------------------------------------------------------

class ConvGRUUNet3DGenerator(nn.Module):
    """ConvGRU-enhanced 3D U-Net generator.

    This class intentionally accepts the same core arguments as the existing
    ``UNet3DGenerator``:
        in_channels, out_channels, base_channels

    Therefore the main training file only needs to replace the imported class
    and the constructor call can remain almost unchanged.

    Parameters
    ----------
    in_channels:
        Flattened input channels. Default 18 = 2 historical times * 9 variables.
    out_channels:
        Flattened output channels. Default 48 = 12 leads * 4 cloud variables.
        For reduced-lead training, set this to ``4 * cfg.lead_steps``.
    base_channels:
        Width of the U-Net encoder/decoder.
    input_steps:
        Number of historical input steps. Current dataset uses 2.
    input_vars:
        Number of variables per input step. Current dataset uses 9.
    output_vars:
        Number of cloud variables per forecast lead. Current task uses 4.
    forecast_steps:
        Optional explicit lead-step count. If None, inferred from
        out_channels // output_vars.
    """

    def __init__(
        self,
        in_channels: int = 18,
        out_channels: int = 48,
        base_channels: int = 32,
        input_steps: int = 2,
        input_vars: int = 9,
        output_vars: int = 4,
        forecast_steps: Optional[int] = None,
    ) -> None:
        super().__init__()

        if in_channels != input_steps * input_vars:
            raise ValueError(
                f"in_channels={in_channels} is inconsistent with "
                f"input_steps*input_vars={input_steps * input_vars}."
            )
        if out_channels % output_vars != 0:
            raise ValueError("out_channels must be divisible by output_vars.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_steps = input_steps
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.forecast_steps = forecast_steps or (out_channels // output_vars)

        if self.forecast_steps * output_vars != out_channels:
            raise ValueError(
                "forecast_steps * output_vars must equal out_channels. "
                f"Got {self.forecast_steps} * {output_vars} != {out_channels}."
            )

        c = base_channels
        hidden_ch = c * 16

        # Shared encoder for each historical time step.
        self.enc1 = ConvBlock3D(input_vars, c)
        self.enc2 = DownBlock3D(c, c * 2)
        self.enc3 = DownBlock3D(c * 2, c * 4)
        self.enc4 = DownBlock3D(c * 4, c * 8)
        self.bottleneck = DownBlock3D(c * 8, hidden_ch)

        # ConvGRU memory over encoded historical bottleneck features.
        self.past_gru = ConvGRUCell3D(input_channels=hidden_ch, hidden_channels=hidden_ch)

        # Latent transition used for future autoregressive rollout.
        # It converts the previous hidden state into the input of the future GRU.
        self.future_context = nn.Sequential(
            nn.Conv3d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_ch), hidden_ch),
            nn.SiLU(inplace=True),
        )
        self.future_gru = ConvGRUCell3D(input_channels=hidden_ch, hidden_channels=hidden_ch)

        # Shared decoder. It is reused for every forecast lead time.
        self.up4 = UpBlock3D(hidden_ch, c * 8, c * 8)
        self.up3 = UpBlock3D(c * 8, c * 4, c * 4)
        self.up2 = UpBlock3D(c * 4, c * 2, c * 2)
        self.up1 = UpBlock3D(c * 2, c, c)
        self.out_conv = nn.Conv3d(c, output_vars, kernel_size=1)

    def _encode_one_step(self, x_t: torch.Tensor):
        """Encode one historical time step.

        Parameters
        ----------
        x_t:
            [B, input_vars, D, H, W]

        Returns
        -------
        b:
            bottleneck feature [B, base*16, D, H/16, W/16 approximately]
        skips:
            tuple of skip features from the encoder.
        """
        s1 = self.enc1(x_t)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        return b, (s1, s2, s3, s4)

    def _decode_one_lead(self, h: torch.Tensor, skips) -> torch.Tensor:
        """Decode one future lead from the current ConvGRU hidden state."""
        s1, s2, s3, s4 = skips
        x = self.up4(h, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out_conv(x)  # [B, output_vars, D, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Input:
            x: [B, input_steps*input_vars, D, H, W]

        Output:
            y_hat: [B, forecast_steps*output_vars, D, H, W]
        """
        B, C, D, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected input channel {self.in_channels}, got {C}.")

        # Restore the explicit historical time dimension:
        # [B, 18, D, H, W] -> [B, 2, 9, D, H, W].
        x_seq = x.contiguous().view(B, self.input_steps, self.input_vars, D, H, W)

        h_state = None
        latest_skips = None

        # Encode historical inputs in chronological order and update memory.
        for t in range(self.input_steps):
            b_t, skips_t = self._encode_one_step(x_seq[:, t])
            h_state = self.past_gru(b_t, h_state)
            latest_skips = skips_t

        # Generate future lead steps sequentially in latent space.
        outputs = []
        for _ in range(self.forecast_steps):
            future_in = self.future_context(h_state)
            h_state = self.future_gru(future_in, h_state)
            y_step = self._decode_one_lead(h_state, latest_skips)
            outputs.append(y_step)

        # [B, T, 4, D, H, W] -> [B, T*4, D, H, W]
        y = torch.stack(outputs, dim=1)
        y = y.reshape(B, self.forecast_steps * self.output_vars, D, H, W)
        return y


# Alias with a short name, convenient for training scripts.
# ConvGRUGenerator = ConvGRUUNet3DGenerator


# if __name__ == "__main__":
#     # Lightweight shape test. Run:
#     #     python generator_convGRU.py
#     model = ConvGRUUNet3DGenerator(in_channels=18, out_channels=16, base_channels=8)
#     x = torch.randn(2, 18, 13, 41, 71)
#     y = model(x)
#     print("input :", tuple(x.shape))
#     print("output:", tuple(y.shape), "expected (2, 16, 13, 41, 71)")
