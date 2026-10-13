from typing import Union

import numpy as np
import torch
import torch.nn as nn


class LatitudeWeightedCharbonnierLoss(nn.Module):
    """Latitude-weighted Charbonnier reconstruction loss.

    Expected tensor shape: [B, C, D, H, W], where H is latitude.

    Extreme-value weight is applied in physical mixing-ratio space:
        w(y) = 1 + alpha * (y / y_ref) ** beta
    because training targets are stored as z = log10(1 + q / q0), while y_ref
    is specified in kg/kg.
    """

    def __init__(
        self,
        latitudes: Union[np.ndarray, torch.Tensor],
        eps: float = 1e-3,
        extreme_alpha: float = 0.0,
        extreme_beta: float = 0.5,
        y_ref: float = 1e-5,
        w_max: float = 100.0,
        q0: float = 1e-8,
    ) -> None:
        super().__init__()
        lat = torch.as_tensor(latitudes, dtype=torch.float32)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-12)
        self.register_buffer("weights", weights.view(1, 1, 1, -1, 1))
        self.eps = eps
        self.extreme_alpha = float(extreme_alpha)
        self.extreme_beta = float(extreme_beta)
        self.y_ref = float(y_ref)
        self.w_max = float(w_max)
        self.q0 = float(q0)

    def _physical_target(self, target: torch.Tensor) -> torch.Tensor:
        q = self.q0 * (torch.pow(10.0, target.float()) - 1.0)
        return q.clamp_min(0.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target   # diff数组维度： B, C, D, H, W
        loss = torch.sqrt(diff.square() + self.eps**2)  # 平方和开根号   loss数组维度： B, C, D, H, W
        weighted = loss * self.weights   # weighted数组维度： B, C, D, H, W
        if self.extreme_alpha == 0.0:
            return weighted.mean()   # 返回weighted数组在所有维度上的平均值
        y_phys = self._physical_target(target)  # 云要素反标准化。
        w_ext = 1.0 + self.extreme_alpha * (y_phys / self.y_ref).pow(self.extreme_beta)
        w_ext = w_ext.clamp_max(self.w_max)   # 大于这个值的数值取这个数值
        return (weighted * w_ext).mean()   # Latitude weighted之后的5维数组，再乘一个，云极值加权的5维数组，最后所有维度上求均值。


N_CLOUD = 4
N_TEMP = 1


def split_cloud_temp(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split [B, lead*(4+1), D, H, W] into cloud and temperature channels.

    Channel order inside each lead is the four cloud variables, then temperature.
    """
    batch, channels, depth, height, width = tensor.shape
    n_out = N_CLOUD + N_TEMP
    if channels % n_out != 0:
        raise ValueError(f"channels {channels} is not divisible by {n_out}")
    lead = channels // n_out   
    view = tensor.view(batch, lead, n_out, depth, height, width)
    cloud = view[:, :, :N_CLOUD].reshape(batch, lead * N_CLOUD, depth, height, width)
    temp = view[:, :, N_CLOUD:].reshape(batch, lead * N_TEMP, depth, height, width)
    return cloud, temp


class CompositeGeneratorLoss(nn.Module):
    """Cloud extreme-weighted Charbonnier plus a small plain temperature term."""

    def __init__(
        self,
        latitudes: Union[np.ndarray, torch.Tensor],
        recon_eps: float = 1e-3,
        extreme_alpha: float = 0.0,
        extreme_beta: float = 0.5,
        y_ref: float = 1e-5,
        w_max: float = 100.0,
        q0: float = 1e-8,
        lambda_temp: float = 0.2,
    ) -> None:
        super().__init__()
        self.recon_fn = LatitudeWeightedCharbonnierLoss(
            latitudes=latitudes,
            eps=recon_eps,
            extreme_alpha=extreme_alpha,
            extreme_beta=extreme_beta,
            y_ref=y_ref,
            w_max=w_max,
            q0=q0,
        )
        self.temp_fn = LatitudeWeightedCharbonnierLoss(
            latitudes=latitudes,
            eps=recon_eps,
            extreme_alpha=0.0,
        )
        self.lambda_temp = float(lambda_temp)

    def compute_components(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_cloud, pred_temp = split_cloud_temp(preds)
        target_cloud, target_temp = split_cloud_temp(targets)
        loss_cloud = self.recon_fn(pred_cloud, target_cloud)
        loss_temp = self.temp_fn(pred_temp, target_temp)
        return {
            "total": loss_cloud + self.lambda_temp * loss_temp,
            "recon": loss_cloud,
            "weighted_recon": loss_cloud,
            "temp": loss_temp,
        }

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_components(preds, targets)["total"]
