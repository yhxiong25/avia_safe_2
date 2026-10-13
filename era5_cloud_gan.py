"""
era5_cloud_gan.py

A PyTorch implementation template for a conditional GAN that forecasts ERA5 China-region   # conditional GAN？  条件GAN具体什么含义？   就是指，discriminator，判别的时候不是单纯用真实场，discriminator 不是孤立地看未来云场，而是把历史输入和未来输出拼在一起看。   在给定历史大气状态 x 的条件下，未来云场 y 是否合理。
cloud microphysics variables.

Task
----
Input  : historical 2 time steps, 9 variables, 13 pressure levels, 41 x 71 grid
         x_raw shape = [2, 9, 13, 41, 71]
Target : future 12 time steps, 4 cloud variables, 13 pressure levels, 41 x 71 grid
         y_raw shape = [12, 4, 13, 41, 71]

For the 3D U-Net generator, time and variable are folded into channels while pressure level
is treated as the 3D depth dimension:
    x = [2*9, 13, 41, 71]   = [18, 13, 41, 71]
    y = [12*4, 13, 41, 71]  = [48, 13, 41, 71]

The discriminator is a conditional 3D PatchGAN:
    D(concat(x, y)) -> patch logits

Data split
----------
Train: windows fully within 2018-01-01 00UTC to 2020-12-31 18UTC
Val  : windows fully within 2021-01-01 00UTC to 2021-12-31 18UTC
Test : windows fully within 2022-01-01 00UTC to 2022-12-31 18UTC

Typical usage
-------------
python era5_cloud_gan.py \
  --zarr_path /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr \
  --save_dir ./checkpoints_cloud_gan \
  --batch_size 4 \
  --pretrain_epochs 30 \
  --gan_epochs 50 \
  --max_val_samples 2048       #？ 一年的最大样本数量？ 2048？    # 365 * 4 = 1460   真实一年大概 1447 个样本左右。   2048只是一个安全参数
"""

from __future__ import annotations

# 为什么没用到logging？
import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional,Union   # 这两个库，真要认真看看。

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


import xarray as xr  # 没必要使用try 和 except   。   


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


INPUT_VARIABLES = [   # 文件名路径？ 
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
]

CLOUD_VARIABLES = [
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
]

SPLIT_RANGES = {
    "train": ("2018-01-01T00:00:00", "2020-12-31T18:00:00"),
    "val": ("2021-01-01T00:00:00", "2021-12-31T18:00:00"),
    "test": ("2022-01-01T00:00:00", "2022-12-31T18:00:00"),
}


# -----------------------------------------------------------------------------
# Reproducibility utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:   # 设置随机种子
    np.random.seed(seed)    # 这三个随机种子，各自具体什么作用和含义？
    torch.manual_seed(seed)   # torch.manual_seed(seed) is used in PyTorch to set the seed for generating random numbers on all devices (CPU and GPU), ensuring reproducibility of experiments. It returns a torch.Generator object that can be reused for controlled randomness.  确保每次生成的随机数是一样的。
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子 为所有可用的 GPU 设置随机数生成器的种子，以确保多 GPU 环境下的结果可复现。
    torch.backends.cudnn.benchmark = True   # 用于启用 cuDNN 的自动优化功能。它会根据硬件和输入数据的大小，选择最优的卷积算法，从而提升模型的运行效率。；一般场景下，只要简单地在 PyTorch 程序开头将其值设置为 True，就可以大大提升卷积神经网络的运行速度。；适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
# 把所有随机性设置随机种子

# -----------------------------------------------------------------------------
# Dataset and dataloaders
# -----------------------------------------------------------------------------

class ERA5CloudWindowDataset(Dataset):
    """Windowed dataset built directly from era5_china_2018_2022.zarr.

    It does not materialize all samples as tensors. Each __getitem__ slices one
    continuous time window from the Zarr-backed xarray Dataset.

    Returns
    -------
    x: torch.FloatTensor
        Shape [18, 13, 41, 71] for 3D U-Net generator.
    y: torch.FloatTensor
        Shape [48, 13, 41, 71].
    meta: dict
        Includes anchor_time, input_times, target_times.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        split: str,
        input_steps: int = 2,
        forecast_steps: int = 12,
        input_variables: Iterable[str] = INPUT_VARIABLES,
        target_variables: Iterable[str] = CLOUD_VARIABLES,
        normalize: bool = True,
        flatten_time_variable: bool = True,
        max_samples: Optional[int] = None,  #这个参数可以是 int，也可以是 None。
        sample_seed: int = 42,
        consolidated: bool = True,
    ) -> None:     # 括号内是可输入的参数，None？  -> None 是返回值类型注解,  表示不返回实际值。
        super().__init__()
        if split not in SPLIT_RANGES:
            raise ValueError(f"split must be one of {list(SPLIT_RANGES)}, got {split!r}")
        if input_steps != 2:
            raise ValueError("This implementation assumes input_steps=2 for anchor logic.")

        self.zarr_path = str(zarr_path)
        self.split = split
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.input_variables = list(input_variables)
        self.target_variables = list(target_variables)
        self.normalize = normalize
        self.flatten_time_variable = flatten_time_variable
        self.consolidated = consolidated
        self._ds = None  # Lazily opened per worker.

        ds = xr.open_zarr(self.zarr_path, consolidated=self.consolidated)
        self.times = ds["time"].values.astype("datetime64[ns]")
        self.latitudes = ds["latitude"].values.astype("float32")
        self.longitudes = ds["longitude"].values.astype("float32")
        self.pressure_levels = ds["pressure_level"].values.astype("float32")
        variable_names = [str(v) for v in ds["variable"].values.tolist()]
        self.variable_names = variable_names

        self.input_var_idx = self._names_to_indices(self.input_variables, variable_names)
        self.target_var_idx = self._names_to_indices(self.target_variables, variable_names)

        if "mean" not in ds or "std" not in ds:
            raise KeyError("Zarr file must contain mean and std variables for normalization.")

        # Store normalization arrays as NumPy, not dask/xarray objects, so workers are safe.
        self.mean_all = ds["mean"].values.astype("float32")  # [variable, level]
        self.std_all = ds["std"].values.astype("float32")
        self.std_all = np.where(self.std_all < 1e-12, 1.0, self.std_all).astype("float32")

        self.indices = self._build_valid_anchor_indices()
        if max_samples is not None and max_samples > 0 and len(self.indices) > max_samples:
            rng = np.random.default_rng(sample_seed)   # 创建随机数生成器并设置种子； 为什么要来个随机数生成器？
            self.indices = np.sort(rng.choice(self.indices, size=max_samples, replace=False))  # 从验证集中随机抽取固定样本??

        ds.close()

    @staticmethod   # ？ 什么作用和意思？   staticmethod 表示这个函数不需要访问 self，也不需要访问类本身 cls
    def _names_to_indices(names: list[str], all_names: list[str]) -> list[int]:  # 将变量的文件路径名转化为要素名？  --应该是返回，要素名的序号。
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        missing = [name for name in names if name not in name_to_idx]  # 
        if missing:
            raise KeyError(
                "Missing variables in Zarr variable coordinate: "
                f"{missing}. Available variables: {all_names}"
            )
        return [name_to_idx[name] for name in names]

    def _build_valid_anchor_indices(self) -> np.ndarray:   # 干嘛的？
        start_s, end_s = SPLIT_RANGES[self.split]
        start = np.datetime64(start_s, "ns")
        end = np.datetime64(end_s, "ns")
        n = len(self.times)
        valid = []
        # anchor i uses input [i-1, i] and target [i+1, ..., i+forecast_steps]
        for i in range(1, n - self.forecast_steps):
            first_time = self.times[i - 1]
            last_time = self.times[i + self.forecast_steps]
            if first_time >= start and last_time <= end:
                valid.append(i)
        if not valid:
            raise RuntimeError(f"No valid window found for split={self.split}")
        return np.asarray(valid, dtype=np.int64)

    def _ensure_open(self):
        if self._ds is None:
            self._ds = xr.open_zarr(self.zarr_path, consolidated=self.consolidated)
        return self._ds

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ds"] = None  # xarray objects are opened lazily inside each worker.
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        ds = self._ensure_open()
        i = int(self.indices[idx])

        # x_raw: [2, 9, 13, 41, 71]
        x = ds["data"].isel(
            time=slice(i - 1, i + 1),
            variable=self.input_var_idx,
        ).values.astype("float32")

        # y_raw: [12, 4, 13, 41, 71]
        y = ds["data"].isel(
            time=slice(i + 1, i + 1 + self.forecast_steps),
            variable=self.target_var_idx,
        ).values.astype("float32")

        if self.normalize:  # 在dataloader里面完成标准化
            x_mean = self.mean_all[self.input_var_idx][None, :, :, None, None]
            x_std = self.std_all[self.input_var_idx][None, :, :, None, None]
            y_mean = self.mean_all[self.target_var_idx][None, :, :, None, None]
            y_std = self.std_all[self.target_var_idx][None, :, :, None, None]
            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std

        if self.flatten_time_variable:
            # [T, V, D, H, W] -> [T*V, D, H, W]
            x = np.ascontiguousarray(x.reshape(self.input_steps * len(self.input_var_idx), *x.shape[2:]))
            y = np.ascontiguousarray(y.reshape(self.forecast_steps * len(self.target_var_idx), *y.shape[2:]))

        meta = {
            "anchor_time": str(self.times[i]),
            "input_times": [str(t) for t in self.times[i - 1:i + 1]],
            "target_start": str(self.times[i + 1]),
            "target_end": str(self.times[i + self.forecast_steps]),
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


@dataclass   # 什么含义和作用？ 自动为类生成特殊方法如__init__、__repr__、__eq__等，简化了类的创建过程。
class DataLoaders:   # 什么意思？ 
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: ERA5CloudWindowDataset
    val_dataset: ERA5CloudWindowDataset
    test_dataset: ERA5CloudWindowDataset


def make_dataloaders(
    zarr_path: str | Path,   # 
    batch_size: int = 4,
    num_workers: int = 4,  # 加载数据的子进程
    max_val_samples: int = 2048,  # 为什么是2048？ 2048怎么来的？
    max_test_samples: Optional[int] = None,
    seed: int = 42,
) -> DataLoaders:
    train_ds = ERA5CloudWindowDataset(zarr_path, split="train", max_samples=None, sample_seed=seed)
    val_ds = ERA5CloudWindowDataset(zarr_path, split="val", max_samples=max_val_samples, sample_seed=seed)
    test_ds = ERA5CloudWindowDataset(zarr_path, split="test", max_samples=max_test_samples, sample_seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return DataLoaders(train_loader, val_loader, test_loader, train_ds, val_ds, test_ds)


# -----------------------------------------------------------------------------
# Loss functions and metrics
# -----------------------------------------------------------------------------

class LatitudeWeightedCharbonnierLoss(nn.Module):
    """  训练模型用的损失函数是： 损失函数用的BCE 加上  LatitudeWeightedCharbonnierLoss！！， 所以下面两个是单纯的评估指标。  这个是损失函数，就不用加上no grad
    Latitude-weighted Charbonnier L1 loss.

    Expected tensor shape: [B, C, D, H, W], where H is latitude.
    """
    def __init__(self, latitudes: np.ndarray | torch.Tensor, eps: float = 1e-3) -> None:
        super().__init__()
        lat = torch.as_tensor(latitudes, dtype=torch.float32)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-12)
        self.register_buffer("weights", weights.view(1, 1, 1, -1, 1))   # register_buffer ?? 干嘛的？self.register_buffer(name, tensor) 用于在模型中注册一组不会参与梯度更新但会随模型一起保存和加载的张量。它常用于保存固定常量
        # view， 改变张量的形状。   -1表示，在该维度上，应该分配多少个分配多少个。反正让数据个数不变就行。
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)  # torch.sqrt返回一个tensor
        return (loss * self.weights).mean()


@torch.no_grad()   # 是 PyTorch 中用于临时禁用梯度计算的上下文管理器或装饰器，常用于推理阶段以减少内存占用并加快计算速度。 所有张量运算的结果都会自动设置 requires_grad=False，即使输入张量原本需要梯度   #  针对的是，无需反向传播的任务
def latitude_weighted_rmse(pred: torch.Tensor, target: torch.Tensor, latitudes: torch.Tensor | np.ndarray) -> torch.Tensor:
    lat = torch.as_tensor(latitudes, dtype=pred.dtype, device=pred.device)
    weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    weights = weights / weights.mean().clamp_min(1e-12)
    weights = weights.view(1, 1, 1, -1, 1)   # 
    mse = ((pred - target) ** 2 * weights).mean()
    return torch.sqrt(mse)


@torch.no_grad()
def csi_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Critical Success Index in the current tensor space.   # CSI必须要自定义一个threshold

    If pred/target are normalized, threshold should also be in normalized space.
    For physical CSI, denormalize first and use a physical threshold.
    """
    pred_event = pred > threshold
    target_event = target > threshold
    hits = (pred_event & target_event).sum().float()
    misses = (~pred_event & target_event).sum().float()
    false_alarms = (pred_event & ~target_event).sum().float()
    denom = hits + misses + false_alarms
    return hits / denom.clamp_min(1.0)


# -----------------------------------------------------------------------------
# Models: 4-level 3D U-Net generator and 3D PatchGAN discriminator
# -----------------------------------------------------------------------------

def _group_count(channels: int, max_groups: int = 8) -> int:   # ？\
    '''如果 channels 不能被 8 整除，就尝试 7、6、5……直到找到能整除的组数
    '''
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
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
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3DGenerator(nn.Module):
    """4-level anisotropic 3D U-Net generator.

    Input : [B, 18, 13, 41, 71]
    Output: [B, 48, 13, 41, 71]
    """

    def __init__(self, in_channels: int = 18, out_channels: int = 48, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3D(in_channels, c)
        self.enc2 = DownBlock3D(c, c * 2)
        self.enc3 = DownBlock3D(c * 2, c * 4)
        self.enc4 = DownBlock3D(c * 4, c * 8)
        self.bottleneck = DownBlock3D(c * 8, c * 16)

        self.up4 = UpBlock3D(c * 16, c * 8, c * 8)
        self.up3 = UpBlock3D(c * 8, c * 4, c * 4)
        self.up2 = UpBlock3D(c * 4, c * 2, c * 2)
        self.up1 = UpBlock3D(c * 2, c, c)
        self.out_conv = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out_conv(x)


def _sn(module: nn.Module, use_spectral_norm: bool) -> nn.Module:    # use_spectral_norm  ? 什么东西？ 谱归一化？ 
    '''用于在 神经网络层的权重参数 上施加谱归一化约束，使其满足 1-Lipschitz 连续性，常用于 GAN 判别器 以稳定训练。
    什么是谱归一化？    谱归一化会约束权重矩阵的最大奇异值，简单理解是限制判别器不要变化得太剧烈。
      这里的discriminator具体是什么逻辑啊？
    '''
    return nn.utils.spectral_norm(module) if use_spectral_norm else module



class PatchGAN3DDiscriminator(nn.Module):
    """Conditional 3D PatchGAN discriminator.

    Input is concat([x, y], dim=1):    # 为什么要合并？
        x: [B, 18, 13, 41, 71]
        y: [B, 48, 13, 41, 71]
        concat: [B, 66, 13, 41, 71]
    Output: patch logits [B, 1, D', H', W'].
    """

    def __init__(self, in_channels: int = 66, base_channels: int = 32, spectral_norm: bool = True) -> None:  # 默认要用谱标准化。
        super().__init__()
        c = base_channels   # 就是卷积的核的数量
        layers = [
            _sn(nn.Conv3d(in_channels, c, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c, c * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False), spectral_norm),
            nn.GroupNorm(_group_count(c * 2), c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c * 2, c * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False), spectral_norm),
            nn.GroupNorm(_group_count(c * 4), c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=False), spectral_norm),
            nn.GroupNorm(_group_count(c * 8), c * 8),    # 分组进行标准化？具体什么含义？
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c * 8, 1, kernel_size=3, stride=1, padding=1), spectral_norm),
        ]
        self.net = nn.Sequential(*layers)   # 为什么要加个*  ？  *layers 把列表“拆开”

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))   # 就是把这个x，y的tensor合并，然后在经过上面定义的模型架构，   这里的x，和y具体是什么？


# -----------------------------------------------------------------------------
# Training and validation
# -----------------------------------------------------------------------------

@dataclass     
class TrainConfig:   ## 单纯定义参数和参数类型？  不能在命令行中运行文件时，在通过args自定义一些参数？   
    '''是不是在这里定义各种超参数不用在linux命令行运行程序也可以？   那在命令行中运行这个程序，带上超参数，谁的优先级更高呢？   这里训练模式，有输出各种训练超参数的设置吗？
TrainConfig 是一个 dataclass，用来集中保存训练超参数。
    命令行参数优先。
    '''
    zarr_path: str     #  /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr/
    save_dir: str = "./logs/checkpoints_cloud_gan"    # /home/yx/projects/common_data/sais2025/test1/logs/某个test文件夹/    需要在正式存储内容之前，检查一下文件夹名字是否存在，并创建一下。
    batch_size: int = 4
    num_workers: int = 4   # 什么意思？  子进程？
    max_val_samples: int = 2048
    max_test_samples: Optional[int] = None
    pretrain_epochs: int = 30  # 与训练的轮次
    gan_epochs: int = 50   #正式训练的轮次
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5   # 
    beta2: float = 0.999 #   adam优化器的动量相关的参数
    lambda_adv: float = 1e-3
    base_channels_g: int = 32  # generator的卷积核数量
    base_channels_d: int = 32  # 
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"   # 先在一个GPU上训练，之后再看并行
    amp: bool = True  # 什么意思？ 自动混合精度训练（Automatic Mixed Precision） ？


def save_checkpoint(path: str | Path, generator: nn.Module, discriminator: Optional[nn.Module], extra: dict) -> None:
    ckpt = {
        "generator": generator.state_dict(),
        "extra": extra,
    }
    if discriminator is not None:   # 按照字典保存各个模型的   .state_dict()具体是怎样的呢？ 需要自己在ide里面试试
        ckpt["discriminator"] = discriminator.state_dict()
    torch.save(ckpt, path)   # 保存模型。  torch保存字典？ 


def _batch_to_device(batch, device: str):  # ？
    x, y, meta = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), meta


@torch.no_grad()
def validate_generator(
    generator: nn.Module,
    loader: DataLoader,
    criterion: LatitudeWeightedCharbonnierLoss,
    latitudes: np.ndarray,
    device: str,
) -> dict:
    '''具体是在干嘛呢？  返回某一个批次，generator的指标
    '''
    generator.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_csi = 0.0
    n = 0
    for batch in loader:
        x, y, _ = _batch_to_device(batch, device)
        pred = generator(x)
        loss = criterion(pred, y)
        rmse = latitude_weighted_rmse(pred, y, latitudes)
        csi = csi_score(pred, y, threshold=0.0)
        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_rmse += rmse.item() * bs
        total_csi += csi.item() * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),
        "rmse": total_rmse / max(n, 1),
        "csi_norm0": total_csi / max(n, 1),
    }


def pretrain_generator(
    generator: nn.Module,
    loaders: DataLoaders,
    criterion: LatitudeWeightedCharbonnierLoss,
    cfg: TrainConfig,
) -> float:
    '''
    '''
    device = cfg.device
    generator.to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_g, betas=(0.9, 0.95), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))
    best_val = math.inf
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)   # 反复确认创建了save_dir三次

    for epoch in range(1, cfg.pretrain_epochs + 1):
        generator.train()
        running = 0.0
        seen = 0
        for batch in loaders.train:
            x, y, _ = _batch_to_device(batch, device)
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.startswith("cuda"))):  # amp到底是什么含义？ 这一句代码又是什么意思呢？
                pred = generator(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt_g)
            scaler.update()
            bs = x.shape[0]
            running += loss.item() * bs
            seen += bs

        val_metrics = validate_generator(generator, loaders.val, criterion, loaders.val_dataset.latitudes, device)
        train_loss = running / max(seen, 1)
        print(   # 打印预训练各个批次的训练信息
            f"[Pretrain {epoch:03d}/{cfg.pretrain_epochs}] "
            f"train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_rmse={val_metrics['rmse']:.6f} val_csi_norm0={val_metrics['csi_norm0']:.4f}"
        )
        if val_metrics["rmse"] < best_val:   # best_val是 无穷大。  这里这里的第一轮肯定要保存。 然后best_val就更新为非inf的值了。
            best_val = val_metrics["rmse"]
            save_checkpoint(      #  
                save_dir / "best_pretrain_generator.pt",
                generator,
                None,
                {"epoch": epoch, "val_metrics": val_metrics, "stage": "pretrain"},
            )
            print(f"  -> saved best pretrain generator, val_rmse={best_val:.6f}")
    return best_val


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    loaders: DataLoaders,
    recon_criterion: LatitudeWeightedCharbonnierLoss,
    cfg: TrainConfig,
) -> float:
    '''生成的日志文件，数据文件，存哪里去了？

    '''
    device = cfg.device
    generator.to(device)
    discriminator.to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2), weight_decay=1e-4)   # beta是什么意思来着？AdamW在 Adam 的基础上改进了权重衰减（Weight Decay）。。 Adam，Adam W这些以后要花时间。
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2), weight_decay=1e-4)  # weight_decay？
    adv_criterion = nn.BCEWithLogitsLoss()   # 损失函数用的BCE 加上  LatitudeWeightedCharbonnierLoss！！
    scaler_g = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))
    scaler_d = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.startswith("cuda")))  # 什么意思？
    best_val = math.inf
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)   # 

    for epoch in range(1, cfg.gan_epochs + 1):
        generator.train()
        discriminator.train()
        sum_d = 0.0
        sum_g = 0.0
        sum_recon = 0.0
        seen = 0

        for batch in loaders.train:   # 这里反应的是，GAN的训练逻辑。
            x, y, _ = _batch_to_device(batch, device)
            bs = x.shape[0]

            # -----------------------
            # 1. Update discriminator
            # -----------------------
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_y = generator(x)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.startswith("cuda"))):
                real_logits = discriminator(x, y)
                fake_logits = discriminator(x, fake_y.detach())
                real_labels = torch.ones_like(real_logits)
                fake_labels = torch.zeros_like(fake_logits)
                d_loss_real = adv_criterion(real_logits, real_labels)
                d_loss_fake = adv_criterion(fake_logits, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ------------------
            # 2. Update generator
            # ------------------
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.startswith("cuda"))):   # torch.cuda.amp.autocast ? 
                fake_y = generator(x)
                fake_logits = discriminator(x, fake_y)
                adv_loss_g = adv_criterion(fake_logits, torch.ones_like(fake_logits))
                recon_loss = recon_criterion(fake_y, y)
                g_loss = recon_loss + cfg.lambda_adv * adv_loss_g
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            sum_d += d_loss.item() * bs
            sum_g += g_loss.item() * bs
            sum_recon += recon_loss.item() * bs
            seen += bs

        val_metrics = validate_generator(generator, loaders.val, recon_criterion, loaders.val_dataset.latitudes, device)
        print(
            f"[GAN {epoch:03d}/{cfg.gan_epochs}] "
            f"D={sum_d/max(seen,1):.6f} G={sum_g/max(seen,1):.6f} "
            f"Recon={sum_recon/max(seen,1):.6f} val_rmse={val_metrics['rmse']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} val_csi_norm0={val_metrics['csi_norm0']:.4f}"
        )
        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            save_checkpoint(
                save_dir / "best_gan.pt",
                generator,
                discriminator,
                {"epoch": epoch, "val_metrics": val_metrics, "stage": "gan"},
            )
            print(f"  -> saved best GAN checkpoint, val_rmse={best_val:.6f}")
    return best_val


@torch.no_grad()
def evaluate_test(generator: nn.Module, loaders: DataLoaders, criterion: LatitudeWeightedCharbonnierLoss, cfg: TrainConfig) -> dict:
    '''
    '''
    generator.to(cfg.device)
    metrics = validate_generator(generator, loaders.test, criterion, loaders.test_dataset.latitudes, cfg.device)
    print("[Test]", json.dumps(metrics, indent=2))
    return metrics


def load_generator_checkpoint(generator: nn.Module, path: str | Path, device: str) -> dict:
    '''
    '''
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    return ckpt.get("extra", {})


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def parse_args() -> TrainConfig:    # 这里的  ->什么含义？    
    p = argparse.ArgumentParser(description="Train a conditional GAN for ERA5 China cloud forecasting.")
    p.add_argument("--zarr_path", type=str, required=True, )  # 这里应该加一个default   required=True 表示命令行必须传入这个参数。
    p.add_argument("--save_dir", type=str, default="./logs/checkpoints_cloud_gan")   # 
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_val_samples", type=int, default=2048)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--gan_epochs", type=int, default=50)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--lambda_adv", type=float, default=1e-3)
    p.add_argument("--base_channels_g", type=int, default=32)
    p.add_argument("--base_channels_d", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true")
    args = p.parse_args()
    return TrainConfig(
        zarr_path=args.zarr_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        pretrain_epochs=args.pretrain_epochs,
        gan_epochs=args.gan_epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_adv=args.lambda_adv,
        base_channels_g=args.base_channels_g,    # 这也需要自定义？  都是默认32，不要去定义这个
        base_channels_d=args.base_channels_d,   # 就是定义卷积核的数量，第一次卷积操作的卷积核数量，之后卷积的数量在依次翻倍
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,
    )


def main() -> None:
    cfg = parse_args()    # 这里就是，通过修改parse_args()这个函数，来修改训练的超参数？ 但是argparse一般是用来在命令行中指定超参数用的吗？ 
    set_seed(cfg.seed)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)   # 此处就是在检验保存路径是否存在，不存在就创建一个
    '''parents = True : 允许递归创建多级目录，
    exist_ok = True: 目录存在的时候，这里也不会报错
    '''

    print("[INFO] Building dataloaders...")
    loaders = make_dataloaders(
        cfg.zarr_path,   # 
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_val_samples=cfg.max_val_samples,
        max_test_samples=cfg.max_test_samples,
        seed=cfg.seed,
    )
    print(f"[INFO] train samples: {len(loaders.train_dataset)}")
    print(f"[INFO] val samples:   {len(loaders.val_dataset)}")
    print(f"[INFO] test samples:  {len(loaders.test_dataset)}")

    # One batch shape check.
    xb, yb, meta = next(iter(loaders.train))
    print(f"[INFO] x batch shape: {tuple(xb.shape)}  expected [B,18,13,41,71]")
    print(f"[INFO] y batch shape: {tuple(yb.shape)}  expected [B,48,13,41,71]")
    print(f"[INFO] first sample meta: {meta}")

    generator = UNet3DGenerator(in_channels=18, out_channels=48, base_channels=cfg.base_channels_g)
    discriminator = PatchGAN3DDiscriminator(in_channels=18 + 48, base_channels=cfg.base_channels_d)
    criterion = LatitudeWeightedCharbonnierLoss(loaders.train_dataset.latitudes).to(cfg.device)

    print("[INFO] Pretraining generator...")
    pretrain_generator(generator, loaders, criterion, cfg)   # generator预训练，只使用CharbonnierLoss训练。

    pretrain_path = Path(cfg.save_dir) / "best_pretrain_generator.pt"   # 在保存路径中，保存预训练的最佳结果
    if pretrain_path.exists():
        extra = load_generator_checkpoint(generator, pretrain_path, cfg.device)   # 加载预训练的模型
        print(f"[INFO] Loaded best pretrained generator: {extra}")

    print("[INFO] Training GAN...")
    train_gan(generator, discriminator, loaders, criterion, cfg)   # 正式训练GAN

    best_gan_path = Path(cfg.save_dir) / "best_gan.pt"
    if best_gan_path.exists():
        extra = load_generator_checkpoint(generator, best_gan_path, cfg.device)
        print(f"[INFO] Loaded best GAN generator: {extra}")
        metrics = evaluate_test(generator, loaders, criterion, cfg)
        with open(Path(cfg.save_dir) / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
