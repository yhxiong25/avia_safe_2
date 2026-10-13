"""
generator_simVP.py

A compact 3D SimVP-style generator for the ERA5 cloud forecasting GAN task.

Task interface kept identical to the existing UNet3DGenerator:
    Input : x     [B, 18, D, H, W]
            18 = input_steps(2) * input_vars(9)
    Output: y_hat [B, forecast_steps * output_vars, D, H, W]
            output_vars = 4 cloud variables: CIWC, CLWC, CRWC, CSWC

Core idea of SimVP:
    Spatial Encoder  : encode each historical time step independently.
    Temporal Translator: transform historical latent states to future latent states.
    Spatial Decoder  : decode each future latent state to cloud variables.

This implementation uses Conv3d because pressure level is treated as the 3D depth
axis, while downsampling is only applied to latitude-longitude dimensions.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Small utility functions
# -----------------------------------------------------------------------------

def _group_count(channels: int, max_groups: int = 8) -> int:
    """Return the largest group number <= max_groups that divides channels.

    GroupNorm is more stable than BatchNorm for small batch 3D training.
    This helper avoids invalid settings such as GroupNorm(8, 10).
    """
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


# -----------------------------------------------------------------------------
# Basic 3D CNN blocks
# -----------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """Two Conv3d + GroupNorm + SiLU layers.

    The kernel is isotropic 3x3x3, but later downsampling keeps the pressure-level
    dimension unchanged. Therefore the model can learn vertical structures while
    avoiding excessive loss of pressure-level resolution.
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
    """Downsample latitude-longitude only; keep pressure-level depth unchanged."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Inception3DBlock(nn.Module):
    """A lightweight multi-scale 3D convolution block for latent translation.

    SimVP commonly uses an Inception-like translator. Here three branches use
    different latitude-longitude kernel sizes:
        3x3, 5x5, 7x7 in horizontal space,
    while pressure-level kernel size is kept at 3.

    This lets the temporal translator see cloud structures at several horizontal
    scales without using a recurrent network or Transformer.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        b = channels // 3
        b3 = channels - 2 * b

        self.branch3 = nn.Conv3d(
            channels, b, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False
        )
        self.branch5 = nn.Conv3d(
            channels, b, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False
        )
        self.branch7 = nn.Conv3d(
            channels, b3, kernel_size=(3, 7, 7), padding=(1, 3, 3), bias=False
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)
        x = self.fuse(x)
        return x + residual


# -----------------------------------------------------------------------------
# SimVP modules: spatial encoder, temporal translator, spatial decoder
# -----------------------------------------------------------------------------

class SpatialEncoder3D(nn.Module):
    """Shared encoder applied to each input historical time step.

    Input : [B*T_in, input_vars, D, H, W]
    Output: [B*T_in, latent_channels, D, H/4, W/4]
    """

    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.stem = ConvBlock3D(in_channels, c)
        self.down1 = DownBlock3D(c, c * 2)
        self.down2 = DownBlock3D(c * 2, c * 4)
        self.latent_channels = c * 4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int], Tuple[int, int, int]]:
        # orig_shape and mid_shape are used by the decoder to recover exact size.
        orig_shape = x.shape[2:]          # [D, H, W]
        x = self.stem(x)                  # [B*T, c, D, H, W]
        mid_shape = x.shape[2:]           # [D, H, W], before downsampling
        x = self.down1(x)                 # [B*T, 2c, D, H/2, W/2]
        x = self.down2(x)                 # [B*T, 4c, D, H/4, W/4]
        return x, mid_shape, orig_shape


class TemporalTranslator3D(nn.Module):
    """Map historical latent states to future latent states.

    Input : [B, T_in * C_latent, D, h, w]
    Output: [B, T_out * C_latent, D, h, w]

    This is the key SimVP module: time is translated in latent space, not by a
    recurrent hidden state. It is simpler and easier to parallelize than RNN-based
    models such as ConvGRU/ConvLSTM.
    """

    def __init__(
        self,
        latent_channels: int,
        input_steps: int,
        forecast_steps: int,
        hidden_channels: int | None = None,
        depth: int = 4,
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels or latent_channels

        layers: list[nn.Module] = [
            nn.Conv3d(input_steps * latent_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        ]
        for _ in range(depth):
            layers.append(Inception3DBlock(hidden_channels))
        layers.append(nn.Conv3d(hidden_channels, forecast_steps * latent_channels, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialDecoder3D(nn.Module):
    """Shared decoder applied to each future latent state.

    Input : [B*T_out, latent_channels, D, h, w]
    Output: [B*T_out, output_vars, D, H, W]

    No U-Net skip connection is used here because SimVP separates spatial encoding,
    latent temporal translation and spatial decoding. This makes it a clean baseline
    for testing whether latent-space temporal translation improves the GAN generator.
    """

    def __init__(self, latent_channels: int, out_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.conv_latent = ConvBlock3D(latent_channels, c * 2)
        self.conv_mid = ConvBlock3D(c * 2, c)
        self.conv_out = ConvBlock3D(c, c)
        self.out_conv = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(
        self,
        z: torch.Tensor,
        mid_shape: Sequence[int],
        orig_shape: Sequence[int],
    ) -> torch.Tensor:
        z = self.conv_latent(z)
        z = F.interpolate(z, size=mid_shape, mode="trilinear", align_corners=False)
        z = self.conv_mid(z)
        z = F.interpolate(z, size=orig_shape, mode="trilinear", align_corners=False)
        z = self.conv_out(z)
        return self.out_conv(z)


# -----------------------------------------------------------------------------
# Full SimVP generator
# -----------------------------------------------------------------------------

class SimVP3DGenerator(nn.Module):
    """3D SimVP-style generator for ERA5 cloud forecasting.

    Expected input/output for current GAN training code:
        x     : [B, 18, D, H, W]
        y_hat : [B, forecast_steps * 4, D, H, W]

    Internally:
        [B, 18, D, H, W]
            -> [B, input_steps=2, input_vars=9, D, H, W]
            -> shared spatial encoder per historical time
            -> latent temporal translator
            -> shared spatial decoder per future lead time
            -> [B, forecast_steps*4, D, H, W]
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
        translator_depth: int = 4,
        translator_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()

        if in_channels != input_steps * input_vars:
            raise ValueError(
                f"in_channels must equal input_steps*input_vars, got "
                f"{in_channels} vs {input_steps}*{input_vars}."
            )

        if forecast_steps is None:
            if out_channels % output_vars != 0:
                raise ValueError("out_channels must be divisible by output_vars.")
            forecast_steps = out_channels // output_vars

        expected_out = forecast_steps * output_vars
        if out_channels != expected_out:
            raise ValueError(
                f"out_channels must equal forecast_steps*output_vars, got "
                f"{out_channels} vs {forecast_steps}*{output_vars}."
            )

        self.input_steps = input_steps
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.forecast_steps = forecast_steps

        self.encoder = SpatialEncoder3D(input_vars, base_channels=base_channels)
        latent_channels = self.encoder.latent_channels

        self.translator = TemporalTranslator3D(
            latent_channels=latent_channels,
            input_steps=input_steps,
            forecast_steps=forecast_steps,
            hidden_channels=translator_hidden_channels,
            depth=translator_depth,
        )

        self.decoder = SpatialDecoder3D(
            latent_channels=latent_channels,
            out_channels=output_vars,
            base_channels=base_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, input_steps*input_vars, D, H, W].

        Returns
        -------
        torch.Tensor
            Shape [B, forecast_steps*output_vars, D, H, W].
        """
        B, C, D, H, W = x.shape
        expected_c = self.input_steps * self.input_vars
        if C != expected_c:
            raise ValueError(f"Expected input channel {expected_c}, got {C}.")

        # Restore the explicit historical time dimension.
        # [B, 18, D, H, W] -> [B, 2, 9, D, H, W]
        x = x.view(B, self.input_steps, self.input_vars, D, H, W)

        # Encode each historical time step with the same spatial encoder.
        # [B, T, V, D, H, W] -> [B*T, V, D, H, W]
        x = x.reshape(B * self.input_steps, self.input_vars, D, H, W)
        z, mid_shape, orig_shape = self.encoder(x)

        # [B*T, C_latent, D, h, w] -> [B, T*C_latent, D, h, w]
        _, C_latent, d_lat, h_lat, w_lat = z.shape
        z = z.view(B, self.input_steps, C_latent, d_lat, h_lat, w_lat)
        z = z.reshape(B, self.input_steps * C_latent, d_lat, h_lat, w_lat)

        # Translate historical latent sequence into future latent sequence.
        z = self.translator(z)

        # [B, T_out*C_latent, D, h, w] -> [B*T_out, C_latent, D, h, w]
        z = z.view(B, self.forecast_steps, C_latent, d_lat, h_lat, w_lat)
        z = z.reshape(B * self.forecast_steps, C_latent, d_lat, h_lat, w_lat)

        # Decode every future lead with the shared spatial decoder.
        y = self.decoder(z, mid_shape=mid_shape, orig_shape=orig_shape)

        # [B*T_out, 4, D, H, W] -> [B, T_out*4, D, H, W]
        y = y.view(B, self.forecast_steps, self.output_vars, D, H, W)
        y = y.reshape(B, self.forecast_steps * self.output_vars, D, H, W)
        return y


# # Backward-compatible alias if you prefer a longer explicit class name.
# SimVP3DGeneratorForCloudForecast = SimVP3DGenerator


# if __name__ == "__main__":
#     # Minimal shape test. This does not train the model.
#     model = SimVP3DGenerator(
#         in_channels=18,
#         out_channels=16,
#         base_channels=16,
#         input_steps=2,
#         input_vars=9,
#         output_vars=4,
#         forecast_steps=4,
#     )
#     x = torch.randn(2, 18, 13, 41, 71)
#     y = model(x)
#     print("input :", tuple(x.shape))
#     print("output:", tuple(y.shape))
