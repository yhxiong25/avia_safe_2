# -*- coding: utf-8 -*-
"""
Average_CSI_evaluation

功能：
    读取训练好的 Generator 模型；
    遍历 test set 一整年的所有样本；
    计算每个 forecast lead time 上的 CSI；
    保存 CSV 结果和 CSI-lead time 曲线图。

默认评估：
    4 个云变量：
        CIWC, CLWC, CRWC, CSWC

    默认高度层：
        500 hPa

    阈值选取：
        指定高度层上，每个云变量所有非零值的 75% 分位数，
        再保留 2 位有效数字向上取整。

注意：
    DataLoader 输出的 pred/y 是标准化后的值；
    CSI 评估前需要反标准化回原始物理量 kg kg^-1。
"""

import os
import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib
matplotlib.use("Agg")    # 不使用GUI，直接使用Agg绘图引擎，将绘图结果保存为图片文件。
import matplotlib.pyplot as plt

import sys
PROJECT_ROOT = '/home/yx/projects/common_data/sais2025/era5_cloud_gan'
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)   # 为了能勾索引.src目录中去，调用项目源码

# 这里假定你的训练脚本中类名为 UNet3DGenerator
# 如果你的类名不同，请相应修改导入名称

from src.models.generator import UNet3DGenerator
from src.metrics.metrics import csi_score    # 除了csi score，还有其它score吗？ 
from src.data.data_loader import DataLoaders,make_dataloaders
from src.utils.utilities import ceil_to_sigfig

# ------------------------------------------------------------
# 4 个目标云变量顺序必须与 Dataset 输出顺序一致：
# [CIWC, CLWC, CRWC, CSWC]
# ------------------------------------------------------------
CLOUD_VARIABLES = [
    "specific_cloud_ice_water_content",      # CIWC
    "specific_cloud_liquid_water_content",   # CLWC
    "specific_rain_water_content",           # CRWC
    "specific_snow_water_content",           # CSWC
]

CLOUD_SHORT_NAMES = ["CIWC", "CLWC", "CRWC", "CSWC"]

THRESHOLDS = {"CIWC_500":8.9e-06,"CLWC_500":3.2e-05,"CRWC_500":3.0e-07,"CSWC_500":2.2e-05}   
# THRESHOLDS = [8.9e-06,3.2e-05,3.0e-07,2.2e-05]  # 四个云变量的阈值


def ceil_to_sigfig(x: float, sigfig: int = 2) -> float:
    """   每一次运行还要重新用很多时间算一次阈值。很不划算。 需要改！！！
    将数值按有效数字向上取整。

    例如：
        8.83e-06 -> 8.9e-06
        3.12e-05 -> 3.2e-05
        2.01e-07 -> 2.1e-07
    """
    if x == 0 or not np.isfinite(x):
        return x
    x_abs = abs(x) 
    exponent = np.floor(np.log10(x_abs))  # ?    小数点第二位向上取整，为什么能这样？
    scale = 10 ** (sigfig - 1 - exponent)   #?  scale : 1d7
    return float(np.ceil(x_abs * scale) / scale)   # ? 


class Average_CSI_evaluation:
    """
    计算模型在 test set 上的平均 CSI 随 lead time 的变化。

    Parameters
    ----------
    model_path : str
        best_gan.pt 或 best_pretrain_generator.pt 路径。

    zarr_path : str
        era5_china_2018_2022.zarr 路径。

    batch_size : int
        test loader 的 batch size。

    num_workers : int
        DataLoader workers。建议先用 0，避免 xarray/zarr 多进程读取卡住。

    level_hpa : float
        评估的高度层，例如 500。

    base_channels : int
        Generator 初始通道数，需与训练时一致。

    threshold_percentile : float
        阈值分位数，默认 75。

    threshold_sigfig : int
        阈值向上取整时保留的有效数字位数，默认 2。
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        zarr_path: Union[str, Path] = "/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr",
        batch_size: int = 4,
        num_workers: int = 0,
        device: str = "cuda",
        level_hpa: float = 500.0,
        base_channels: int = 32,
        threshold_percentile: float = 75.0,
        threshold_sigfig: int = 2,
        max_test_samples: int = None,
        seed: int = 42,
    ) -> None:

        self.model_path = Path(model_path)
        self.model_dir = self.model_path.parent

        self.zarr_path = str(zarr_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.level_hpa = level_hpa
        self.base_channels = base_channels
        self.threshold_percentile = threshold_percentile
        self.threshold_sigfig = threshold_sigfig
        self.max_test_samples = max_test_samples
        self.seed = seed

        # 在模型所在文件夹下创建 figure 文件夹
        self.figure_dir = self.model_dir / "figure"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # 结果文件
        self.csv_path = self.figure_dir / f"average_CSI_{int(level_hpa)}hPa.csv"
        self.fig_path = self.figure_dir / f"average_CSI_{int(level_hpa)}hPa.png"
        self.threshold_json_path = self.figure_dir / f"CSI_thresholds_{int(level_hpa)}hPa.json"

    # --------------------------------------------------------
    # 1. 读取 zarr，计算指定高度层的四个云变量阈值
    # --------------------------------------------------------
    def get_thresholds(self):
        """  这个函数用不了
        读取 zarr 文件，计算指定高度层上四个云变量的阈值。

        阈值定义：
            所有非零值的 75% 分位数；
            再保留 2 位有效数字向上取整。

        返回
        ----
        thresholds_raw : np.ndarray
            shape [4]，顺序为 [CIWC, CLWC, CRWC, CSWC]。

        mean_4 : np.ndarray
            shape [4]，该高度层四个云变量的 mean。

        std_4 : np.ndarray
            shape [4]，该高度层四个云变量的 std。

        level_idx : int
            该高度层在 pressure_level 维度中的索引。
        """
        ds = xr.open_zarr(self.zarr_path, consolidated=True, chunks=None)

        pressure_levels = np.asarray(ds["pressure_level"].values, dtype=np.float32)
        level_idx = int(np.argmin(np.abs(pressure_levels - self.level_hpa)))
        level_actual = float(pressure_levels[level_idx])

        var_names = [str(v) for v in ds["variable"].values]
        name_to_idx = {name: i for i, name in enumerate(var_names)}

        thresholds_raw = []
        mean_4 = []
        std_4 = []
        details = {}
        print(name_to_idx)
        for full_name, short_name in zip(CLOUD_VARIABLES, CLOUD_SHORT_NAMES):
            if full_name not in name_to_idx:
                raise ValueError(f"{full_name} 不在 zarr variable 坐标中。")

            var_idx = name_to_idx[full_name]  # 这个var_idx是int类型，从0开始

            arr = ds["data"].isel(
                variable=var_idx,
                pressure_level=level_idx,
            ).values.astype(np.float32)

            finite_mask = np.isfinite(arr)  # 判断是不是有限的值
            positive_values = arr[finite_mask & (arr > 0)]  # 提取有限且大于0的值

            if positive_values.size == 0:
                raise ValueError(f"{short_name}_{int(level_actual)} 没有非零值，无法计算阈值。")

            # q75 = float(np.nanpercentile(positive_values, self.threshold_percentile))
            # threshold_raw = ceil_to_sigfig(q75, sigfig=self.threshold_sigfig)

            mean = float(ds["mean"].isel(variable=var_idx, pressure_level=level_idx).values)
            std = float(ds["std"].isel(variable=var_idx, pressure_level=level_idx).values)

            threshold_raw = THRESHOLDS[short_name+'_500']
            thresholds_raw.append(THRESHOLDS[short_name+'_500'])   # 数值append啥？ 
            mean_4.append(mean)
            std_4.append(std)

            key = f"{short_name}_{int(level_actual)}"
            details[key] = {
                "variable_full_name": full_name,
                "variable_index": int(var_idx),
                "level_hpa": level_actual,
                "level_index": int(level_idx),
                "positive_count": int(positive_values.size),
                "threshold_raw": threshold_raw,
                "mean": mean,
                "std": std,
                "threshold_norm": float((threshold_raw - mean) / max(std, 1e-12)),
            }

            print(
                f"[THR] {key}: "
                # f"q75={q75:.8e}, "
                f"thr_raw={threshold_raw:.8e}, "
                f"thr_norm={details[key]['threshold_norm']:.6f}"
            )

            del arr, positive_values

        # 保存阈值信息，方便复查
        with open(self.threshold_json_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=4, ensure_ascii=False)

        print(f"[INFO] thresholds saved to: {self.threshold_json_path}")
        return (
            np.asarray(thresholds_raw, dtype=np.float32),
            np.asarray(mean_4, dtype=np.float32),
            np.asarray(std_4, dtype=np.float32),
            level_idx,
            level_actual,
            details,
        )

    # --------------------------------------------------------
    # 2. 加载 Generator 模型
    # --------------------------------------------------------
    def load_generator(self):
        """
        加载训练好的 Generator 参数。

        需要保证：
            base_channels 与训练时一致；
            in_channels=18；
            out_channels=48。
        """
        model = UNet3DGenerator(
            in_channels=18,
            out_channels=48,
            base_channels=self.base_channels,
        ).to(self.device)

        ckpt = torch.load(self.model_path, map_location=self.device)

        if isinstance(ckpt, dict) and "generator" in ckpt:
            state_dict = ckpt["generator"]
        else:
            state_dict = ckpt

        # 兼容 DataParallel / DDP 保存的 module. 前缀
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("module.", "")] = v

        model.load_state_dict(cleaned, strict=True)
        model.eval()

        print(f"[INFO] Generator loaded from: {self.model_path}")

        return model

    # --------------------------------------------------------
    # 3. 构建 test loader
    # --------------------------------------------------------
    def build_test_loader(self):
        """
        复用训练脚本中的 make_dataloaders。
        这里只使用 loaders.test。
        """
        loaders = make_dataloaders(
            self.zarr_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            max_val_samples=16,
            max_test_samples=self.max_test_samples,
            seed=self.seed,
        )

        print(f"[INFO] test samples: {len(loaders.test_dataset)}")

        return loaders.test

    # --------------------------------------------------------
    # 4. 计算每个 lead time 的 CSI    不止是csi，还有其它指数。
    # --------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        """
        遍历 test set，计算每个 lead time 的 CSI。

        计算方式：
            对每个 lead time 和每个变量分别累加 TP、FP、FN；
            最后 CSI = TP / (TP + FP + FN)。

        返回
        ----
        df : pandas.DataFrame
            包含每个 lead time 的平均 CSI 和各变量 CSI。
        """
        model = self.load_generator()
        test_loader = self.build_test_loader()

        thresholds_raw, mean_4, std_4, level_idx, level_actual, _ = self.get_thresholds()

        # 转成 torch tensor，用于 GPU 计算
        # shape 设计为 [1, 1, 4, 1, 1]，对应 [B, lead, var, lat, lon]
        thr = torch.as_tensor(thresholds_raw, dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)
        mean = torch.as_tensor(mean_4, dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)
        std = torch.as_tensor(std_4, dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)

        # 累加每个 lead time、每个云变量的 TP/FP/FN
        tp = torch.zeros(12, 4, dtype=torch.float64, device=self.device)
        fp = torch.zeros(12, 4, dtype=torch.float64, device=self.device)
        fn = torch.zeros(12, 4, dtype=torch.float64, device=self.device)

        print("[INFO] Start evaluating CSI on test set...")

        for batch_id, batch in enumerate(test_loader):
            x, y, _ = batch

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            pred = model(x)

            B, C, D, H, W = pred.shape

            # [B, 48, 13, H, W] -> [B, 12, 4, 13, H, W]
            pred = pred.view(B, 12, 4, D, H, W)
            y = y.view(B, 12, 4, D, H, W)

            # 只取指定高度层，例如 500 hPa
            pred_l = pred[:, :, :, level_idx, :, :]  # [B, 12, 4, H, W]
            y_l = y[:, :, :, level_idx, :, :]        # [B, 12, 4, H, W]

            # 反标准化回原始物理量 kg kg^-1
            pred_raw = pred_l * std + mean
            y_raw = y_l * std + mean

            # 云变量物理上非负，防止模型输出反标准化后出现负值
            pred_raw = torch.clamp(pred_raw, min=0.0)

            pred_event = pred_raw >= thr
            obs_event = y_raw >= thr

            # 按 batch、lat、lon 累加，保留 lead 和 variable 两个维度
            # 输出 shape: [12, 4]
            tp += (pred_event & obs_event).sum(dim=(0, 3, 4)).double()
            fp += (pred_event & (~obs_event)).sum(dim=(0, 3, 4)).double()
            fn += ((~pred_event) & obs_event).sum(dim=(0, 3, 4)).double()

            if (batch_id + 1) % 50 == 0:
                print(f"[INFO] evaluated batches: {batch_id + 1}")

        denom = tp + fp + fn
        csi = torch.where(denom > 0, tp / denom, torch.nan)  # [12, 4]
        mean_csi = torch.nanmean(csi, dim=1)                 # [12]

        csi_np = csi.detach().cpu().numpy()
        mean_csi_np = mean_csi.detach().cpu().numpy()

        # 保存结果表
        rows = []
        for lead_idx in range(12):
            row = {
                "lead_step": lead_idx + 1,
                "lead_hour": (lead_idx + 1) * 6,
                "CSI_mean_4vars": float(mean_csi_np[lead_idx]),
            }

            for var_i, short_name in enumerate(CLOUD_SHORT_NAMES):
                row[f"CSI_{short_name}"] = float(csi_np[lead_idx, var_i])

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)

        print(f"[INFO] CSI csv saved to: {self.csv_path}")

        self.plot_csi_curve(df, level_actual)

        return df

    # --------------------------------------------------------
    # 5. 绘制 CSI 随 lead time 的变化
    # --------------------------------------------------------
    def plot_csi_curve(self, df: pd.DataFrame, level_actual: float):
        """
        绘制平均 CSI 和四个云变量各自 CSI 随 lead time 的变化。
        """
        plt.figure(figsize=(9, 6))

        colors = {
            "CIWC": "#5B6CFF",
            "CLWC": "#2CA25F",
            "CRWC": "#F28E2B",
            "CSWC": "#4ECDC4",
        }

        for short_name in CLOUD_SHORT_NAMES:
            plt.plot(
                df["lead_hour"],
                df[f"CSI_{short_name}"],
                marker="o",
                linewidth=1.5,
                alpha=0.85,
                color=colors[short_name],
                label=short_name,
            )

        plt.xlabel("Forecast lead time (hour)", fontsize=12)
        plt.ylabel("Critical Success Index (CSI)", fontsize=12)
        plt.title(
            f"Average CSI vs Forecast Lead Time\n"
            f"Level: {int(level_actual)} hPa | Test set",
            fontsize=13,
            fontweight="bold"
        )

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.xticks(df["lead_hour"].values)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] CSI figure saved to: {self.fig_path}")



if __name__ == "__main__":
    evaluator = Average_CSI_evaluation(
        model_path="/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test5AdvancedLoss/best_pretrain_generator.pt",
        zarr_path="/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr",
        batch_size=4,
        num_workers=0,
        device="cuda",
        level_hpa=500.0,
        base_channels=32,
    )

    df = evaluator.evaluate()
    print(df)
