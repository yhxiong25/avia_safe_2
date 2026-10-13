# -*- coding: utf-8 -*-
"""
evaluate_gan_forecast.py

功能：
    加载训练好的 GAN Generator；
    在 test set 上计算指定高度层的 CSI, POD, FAR, ACC, RMSE；
    绘制每个指标随 forecast lead time 的变化曲线。

每张图：
    一个指标 + 一个高度层；
    4 条曲线分别表示 CIWC, CLWC, CRWC, CSWC。
"""

import os
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.data_loader import make_dataloaders
from src.models.generator import UNet3DGenerator

from src.data.fixed_info import INPUT_VARIABLES,CLOUD_SHORT_NAMES,THRESHOLDS,THRESHOLDS_normalized



class GANForecastEvaluator:
    """
    对训练好的 GAN Generator 进行 test set 评估和绘图。

    Parameters
    ----------
    model_path : str
        训练好的 best_gan.pt 路径。

    save_dir : str
        图像和评估结果保存目录，例如：
        /home/yx/projects/common_data/sais2025/test1/logs/GAN_test6/evaluation/

    zarr_path : str
        ERA5 zarr 数据路径。

    level_hpa : float
        需要评估的高度层，例如 500, 200, 850, 1000。

    batch_size : int
        test loader 的 batch size。

    num_workers : int
        DataLoader 进程数。你之前 zarr 多进程容易卡住，因此默认 0。

    device : str
        例如 "cuda:0", "cuda:1", "cpu"。

    base_channels_g : int
        Generator 的 base channels，必须和训练时一致。
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        zarr_path: str = "/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr",
        level_hpa: float = 500.0,
        batch_size: int = 4,
        num_workers: int = 0,
        device: str = "cuda:0",
        base_channels_g: int = 32,
        max_test_samples: Optional[int] = None,
        seed: int = 42,
        threshold_percentile: float = 75.0,
        threshold_sigfig: int = 2,
    ) -> None:

        self.model_path = Path(model_path)
        self.save_dir = model_path.parent / 'evaluation'      # PosixPath('/home/yx/projects/common_data/sais2025/test1/logs/GAN_test7_modifiedADVLoss_5/evaluation')
        self.zarr_path = zarr_path
        self.level_hpa = level_hpa

        self.batch_size = batch_size
        self.num_workers = num_workers   # 搞这么多压根用不到的参数调整很烦
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.base_channels_g = base_channels_g
        self.max_test_samples = max_test_samples
        self.seed = seed

        self.threshold_percentile = threshold_percentile
        self.threshold_sigfig = threshold_sigfig

        self.save_dir.mkdir(parents=True, exist_ok=True)   # parents=True   父目录不在，可以自己创建。

        self.csv_path = self.save_dir / f"metrics_{int(level_hpa)}hPa.csv"   #  应该多来几个高度的？   1000，850，500，250
        self.threshold_path = self.save_dir / f"thresholds_{int(level_hpa)}hPa.json"

        # 运行时填充
        self.level_idx = None
        self.level_actual = None
        self.thresholds_raw = None
        self.mean_4 = None
        self.std_4 = None
        self.climatology_4 = None
        self.latitudes = None

    # --------------------------------------------------------
    # 1. 读取 zarr 元数据、阈值、标准化参数、气候态
    # --------------------------------------------------------
    def prepare_metadata(self):
        """    # 这里只需要，提取特定的mean/std，
        准备指定高度层的：
            1. level_idx
            2. 四个云变量的阈值
            3. mean/std，用于反标准化
            4. 训练集气候态，用于 ACC
            5. latitude，用于纬度权重
        """
        ds = xr.open_zarr(self.zarr_path, consolidated=True, chunks=None)

        pressure_levels = np.asarray(ds["pressure_level"].values, dtype=np.float32)
        self.level_idx = int(np.argmin(np.abs(pressure_levels - self.level_hpa)))   # 这个完全没必要
        self.level_actual = float(pressure_levels[self.level_idx])   # 
        self.latitudes = np.asarray(ds["latitude"].values, dtype=np.float32)   # 完全可以通过dataloaders获取

        var_names = [str(v) for v in ds["variable"].values]
        name_to_idx = {name: i for i, name in enumerate(var_names)}
        var_idx = [name_to_idx[v] for v in CLOUD_VARIABLES]

        thresholds = []
        means = []
        stds = []
        threshold_info = {}   # 这里想要获得的数据

        for idx, short_name, full_name in zip(var_idx, CLOUD_SHORT_NAMES, CLOUD_VARIABLES):
            arr = ds["data"].isel(
                variable=idx,
                pressure_level=self.level_idx,
            ).values.astype(np.float32)

            valid = np.isfinite(arr) & (arr > 0)
            nonzero_values = arr[valid]

            if nonzero_values.size == 0:
                raise ValueError(f"{short_name}_{int(self.level_actual)}hPa 没有非零值，无法计算阈值。")

            q = float(np.nanpercentile(nonzero_values, self.threshold_percentile))
            thr = ceil_to_sigfig(q, sigfig=self.threshold_sigfig)

            mean = float(ds["mean"].isel(variable=idx, pressure_level=self.level_idx).values)
            std = float(ds["std"].isel(variable=idx, pressure_level=self.level_idx).values)

            thresholds.append(thr)
            means.append(mean)
            stds.append(std)

            key = f"{short_name}_{int(self.level_actual)}"
            threshold_info[key] = {   # 最终想要获取的信息。
                "full_name": full_name,
                "q75_nonzero": q,
                "threshold_raw": thr,
                "mean": mean,
                "std": std,
                "threshold_norm": float((thr - mean) / max(std, 1e-12)),
                "nonzero_count": int(nonzero_values.size),
            }

            del arr, nonzero_values

        self.thresholds_raw = np.asarray(thresholds, dtype=np.float32)
        self.mean_4 = np.asarray(means, dtype=np.float32)
        self.std_4 = np.asarray(stds, dtype=np.float32)

        # ACC 使用训练集 2018-2020 的气候态，shape: [4, H, W]
        clim = ds["data"].isel(
            variable=var_idx,
            pressure_level=self.level_idx,
        ).sel(
            time=slice("2018-01-01", "2020-12-31T23:59:59")
        ).mean(dim="time").values.astype(np.float32)

        self.climatology_4 = clim

        with open(self.threshold_path, "w", encoding="utf-8") as f:   # 还保存了一个文件
            json.dump(threshold_info, f, indent=4, ensure_ascii=False)

        print(f"[INFO] level requested: {self.level_hpa} hPa")
        print(f"[INFO] nearest level:   {self.level_actual} hPa")
        print(f"[INFO] thresholds saved to: {self.threshold_path}")

    # --------------------------------------------------------
    # 2. 加载 Generator
    # --------------------------------------------------------
    def load_generator(self):
        """
        加载训练好的 Generator。
        只需要 Generator，不需要 Discriminator。
        """
        model = UNet3DGenerator(
            in_channels=18,
            out_channels=48,
            base_channels=self.base_channels_g,
        ).to(self.device)
        ckpt = torch.load(self.model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "generator" in ckpt:
            state_dict = ckpt["generator"]
        else:
            state_dict = ckpt
        # 兼容 DataParallel / DDP 保存的 module. 前缀
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"[INFO] Generator loaded from: {self.model_path}")
        return model

    # --------------------------------------------------------
    # 3. 构建 test loader
    # --------------------------------------------------------
    def build_test_loader(self):
        """
        复用项目中的 make_dataloaders，只使用 loaders.test。
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
    # 4. 核心评估函数
    # --------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        """
        遍历 test set，计算：
            CSI, POD, FAR, ACC, RMSE

        返回
        ----
        df : pandas.DataFrame
            每个 lead time 上 4 个云变量的评估结果。
        """
        self.prepare_metadata()   # 应该是存了一个文件，然后之后打开文件调用参数
        model = self.load_generator()
        test_loader = self.build_test_loader()    # 

        # 纬度权重，shape: [1, 1, 1, H, 1]
        lat = torch.as_tensor(self.latitudes, dtype=torch.float32, device=self.device)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-12)
        weights = weights.view(1, 1, 1, -1, 1)

        # threshold / mean / std / climatology
        thr = torch.as_tensor(self.thresholds_raw, dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)  # ? [B, 48, 13, H, W]
        mean = torch.as_tensor(self.mean_4, dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)
        std = torch.as_tensor(self.std_4, dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)

        clim = torch.as_tensor(self.climatology_4, dtype=torch.float32, device=self.device).view(1, 1, 4, *self.climatology_4.shape[-2:])
        # ?
        # 分类指标累积量
        tp = torch.zeros(12, 4, dtype=torch.float64, device=self.device)
        fp = torch.zeros(12, 4, dtype=torch.float64, device=self.device)
        fn = torch.zeros(12, 4, dtype=torch.float64, device=self.device)

        # RMSE 累积量
        rmse_sse = torch.zeros(12, 4, dtype=torch.float64, device=self.device)
        rmse_den = torch.zeros(12, 4, dtype=torch.float64, device=self.device)

        # ACC 累积量
        acc_num = torch.zeros(12, 4, dtype=torch.float64, device=self.device)
        acc_pred_den = torch.zeros(12, 4, dtype=torch.float64, device=self.device)
        acc_true_den = torch.zeros(12, 4, dtype=torch.float64, device=self.device)

        print("[INFO] Start test-set evaluation...")

        for batch_id, batch in enumerate(test_loader):
            x, y, _ = batch

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            pred = model(x)

            B, C, D, H, W = pred.shape    # 模型输出的数据维度： [B, 48, 13, H, W]

            # [B, 48, 13, H, W] -> [B, 12, 4, 13, H, W]
            pred = pred.view(B, 12, 4, D, H, W)
            y = y.view(B, 12, 4, D, H, W)

            # 只取指定高度层，得到 [B, 12, 4, H, W]
            pred_l = pred[:, :, :, self.level_idx, :, :]
            y_l = y[:, :, :, self.level_idx, :, :]

            # 反标准化回原始物理量 kg kg^-1
            pred_raw = pred_l * std + mean
            y_raw = y_l * std + mean

            # 云变量物理上非负
            pred_raw = torch.clamp(pred_raw, min=0.0)
            y_raw = torch.clamp(y_raw, min=0.0)

            # -------------------------
            # CSI / POD / FAR
            # -------------------------
            pred_event = pred_raw >= thr
            obs_event = y_raw >= thr

            tp += (pred_event & obs_event).sum(dim=(0, 3, 4)).double()
            fp += (pred_event & (~obs_event)).sum(dim=(0, 3, 4)).double()
            fn += ((~pred_event) & obs_event).sum(dim=(0, 3, 4)).double()

            # -------------------------
            # RMSE，纬度加权
            # -------------------------
            err2 = (pred_raw - y_raw) ** 2
            rmse_sse += (err2 * weights).sum(dim=(0, 3, 4)).double()

            batch_den = float(B * W * weights.view(-1).sum().detach().cpu().item())
            rmse_den += batch_den

            # -------------------------
            # ACC，使用训练集气候态距平
            # -------------------------
            pred_anom = pred_raw - clim
            true_anom = y_raw - clim

            acc_num += (pred_anom * true_anom * weights).sum(dim=(0, 3, 4)).double()
            acc_pred_den += ((pred_anom ** 2) * weights).sum(dim=(0, 3, 4)).double()
            acc_true_den += ((true_anom ** 2) * weights).sum(dim=(0, 3, 4)).double()

            if (batch_id + 1) % 50 == 0:
                print(f"[INFO] evaluated batches: {batch_id + 1}")

        # -------------------------
        # 汇总指标
        # -------------------------
        csi = tp / (tp + fp + fn + 1e-12)
        pod = tp / (tp + fn + 1e-12)
        far = fp / (tp + fp + 1e-12)

        rmse = torch.sqrt(rmse_sse / (rmse_den + 1e-12))

        acc = acc_num / (
            torch.sqrt(acc_pred_den) * torch.sqrt(acc_true_den) + 1e-12
        )

        df = self._make_dataframe(
            csi.detach().cpu().numpy(),
            pod.detach().cpu().numpy(),
            far.detach().cpu().numpy(),
            acc.detach().cpu().numpy(),
            rmse.detach().cpu().numpy(),
        )

        df.to_csv(self.csv_path, index=False)
        print(f"[INFO] metrics csv saved to: {self.csv_path}")

        return df

    # --------------------------------------------------------
    # 5. 将结果整理成 DataFrame
    # --------------------------------------------------------
    def _make_dataframe(self, csi, pod, far, acc, rmse):
        """
        将 [12, 4] 的指标矩阵整理成长表。
        """
        rows = []

        for lead_idx in range(12):
            row = {
                "lead_step": lead_idx + 1,
                "lead_hour": (lead_idx + 1) * 6,
                "level_hpa": self.level_actual,
            }

            for var_i, name in enumerate(CLOUD_SHORT_NAMES):
                row[f"CSI_{name}"] = float(csi[lead_idx, var_i])
                row[f"POD_{name}"] = float(pod[lead_idx, var_i])
                row[f"FAR_{name}"] = float(far[lead_idx, var_i])
                row[f"ACC_{name}"] = float(acc[lead_idx, var_i])
                row[f"RMSE_{name}"] = float(rmse[lead_idx, var_i])

            rows.append(row)

        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # 6. 绘制单个指标随 lead time 的曲线
    # --------------------------------------------------------
    def plot_metric(self, df: pd.DataFrame, metric: str):
        """
        绘制一个指标随 lead time 变化的曲线。

        每张图：
            x 轴：lead hour
            y 轴：metric
            4 条曲线：CIWC, CLWC, CRWC, CSWC
        """
        metric = metric.upper()

        colors = {
            "CIWC": "#5B6CFF",
            "CLWC": "#2CA25F",
            "CRWC": "#F28E2B",
            "CSWC": "#4ECDC4",
        }

        plt.figure(figsize=(8.5, 5.8))

        for name in CLOUD_SHORT_NAMES:
            plt.plot(
                df["lead_hour"],
                df[f"{metric}_{name}"],
                marker="o",
                linewidth=2.0,
                color=colors[name],
                label=name,
            )

        plt.xlabel("Forecast lead time (hour)", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(
            f"{metric} vs Forecast Lead Time\n"
            f"Level: {int(self.level_actual)} hPa | Test set",
            fontsize=14,
            fontweight="bold",
        )

        plt.grid(True, linestyle="--", alpha=0.35)
        plt.xticks(df["lead_hour"].values)
        plt.legend()

        if metric in ["CSI", "POD", "FAR"]:
            plt.ylim(0, 1)
        elif metric == "ACC":
            plt.ylim(-0.2, 1.0)

        plt.tight_layout()

        fig_path = self.save_dir / f"{metric}_{int(self.level_actual)}hPa_lead_curve.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] {metric} figure saved to: {fig_path}")

    # --------------------------------------------------------
    # 7. 绘制全部指标
    # --------------------------------------------------------
    def plot_all_metrics(self, df: Optional[pd.DataFrame] = None):
        """
        绘制 CSI, POD, FAR, ACC, RMSE 五张图。
        如果 df=None，则从 csv 中读取。
        """
        if df is None:
            df = pd.read_csv(self.csv_path)

        for metric in ["CSI", "POD", "FAR", "ACC", "RMSE"]:
            self.plot_metric(df, metric)

    # --------------------------------------------------------
    # 8. 一键运行：评估 + 绘图
    # --------------------------------------------------------
    def run(self):
        """
        一键完成：
            evaluate()
            plot_all_metrics()
        """
        df = self.evaluate()   #?? GANForecastEvaluator这个类，的函数，也是self的属性？？？
        # self，它代表类的实例，即当前对象的地址。
        self.plot_all_metrics(df)
        return df




