import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Iterable, Optional,Union   # 这两个库，真要认真看看。
import numpy as np
from dataclasses import dataclass
import xarray as xr

import sys
sys.path.append('/home/yx/projects/common_data/sais2025/era5_cloud_gan')
from src.data.fixed_info import INPUT_VARIABLES,CLOUD_VARIABLES,CLOUD_SHORT_NAMES,\
                                THRESHOLDS,THRESHOLDS_normalized,LEVELS,SPLIT_RANGES


'''
data_loader_v2采用全新的，归一化方案。   ZTQUV继续采用一般的，mean 和std的zscore标准化， 云变量由于数值特殊性，采用更合理的，z = log10(1+q/q0)的归一化方法。 q0对所有阈值取为1e-6。

'''
# 每个 lead：4 个云要素在前，温度在最后。不要用 list.append()，它返回 None。
TARGET_VARIABLES = CLOUD_VARIABLES + ["temperature"]


class ERA5CloudWindowDataset(Dataset):
    """Windowed dataset built directly from era5_china_2018_2022.zarr.

    It does not materialize all samples as tensors. Each __getitem__ slices one
    continuous time window from the Zarr-backed xarray Dataset.

    Returns
    -------
    x: torch.FloatTensor
        If flatten_time_variable: [input_steps * n_input_vars, 13, 41, 71].
        Otherwise: [input_steps, n_input_vars, 13, 41, 71].
        Anchor time i uses input [i-input_steps+1, ..., i].  # NOTE
    y: torch.FloatTensor
        If flatten_time_variable: [forecast_steps * n_target_vars, 13, 41, 71].
        Otherwise: [forecast_steps, n_target_vars, 13, 41, 71].
        Target order is cloud variables, then temperature.
    meta: dict
        Includes anchor_time, input_times, target_times.
    """
    def __init__(
        self,
        zarr_path: Union[str,Path],    # | Path,    # python版本不适配这种写法。  str | Path → (改成)   Union[str, Path]
        split: str,
        input_steps: int = 2,
        forecast_steps: int = 12,
        input_variables: Iterable[str] = INPUT_VARIABLES,
        target_variables: Iterable[str] = TARGET_VARIABLES,
        normalize: bool = True,    # 默认用DataLoader加载数据的时候就完成了标准化。
        flatten_time_variable: bool = True,
        max_samples: Optional[int] = None,  #这个参数可以是 int，也可以是 None。
        sample_seed: int = 42,    # 随机抽取样本的随机种子。
        consolidated: bool = True,
    ) -> None:     # 括号内是可输入的参数，None？  -> None 是返回值类型注解,  表示不返回实际值。
        super().__init__()
        if split not in SPLIT_RANGES:
            raise ValueError(f"split must be one of {list(SPLIT_RANGES)}, got {split!r}")
        if input_steps < 1:
            raise ValueError(f"input_steps must be >= 1, got {input_steps}")

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

        ds = xr.open_zarr(self.zarr_path, consolidated=self.consolidated)   # 这里报错。 
        self.times = ds["time"].values.astype("datetime64[ns]")     # 数据集所有时刻的列表
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
        self.std_all = np.where(self.std_all < 1e-12, 1.0, self.std_all).astype("float32")   # 为什么把小于1e-12的换成1.

        self.indices = self._build_valid_anchor_indices()   # 这个返回，所有样本的，anchor time列表
        if max_samples is not None and max_samples > 0 and len(self.indices) > max_samples:   # 如果样本总数要大于要求的样本数，那就随机选取要求的样本数
            rng = np.random.default_rng(sample_seed)   # 创建随机数生成器并设置种子； 为什么要来个随机数生成器？
            self.indices = np.sort(rng.choice(self.indices, size=max_samples, replace=False))  # 从验证集中随机抽取固定样本??
        # 测试集样本是从测试集年份中随机选取的样本
        ds.close()
    
    @staticmethod   # ？ 什么作用和意思？   staticmethod 表示这个函数不需要访问 self，也不需要访问类本身 cls
    def _names_to_indices(names: list[str], all_names: list[str]) -> list[int]:  # 将变量的文件路径名转化为要素名？  --应该是返回，要素名的序号。
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        missing = [name for name in names if name not in name_to_idx]  # 空数组
        if missing:
            raise KeyError(
                "Missing variables in Zarr variable coordinate: "
                f"{missing}. Available variables: {all_names}"
            )
        return [name_to_idx[name] for name in names]   # 

    def _build_valid_anchor_indices(self) -> np.ndarray:   # 干嘛的？
        start_s, end_s = SPLIT_RANGES[self.split]
        start = np.datetime64(start_s, "ns")   # 'ns'到底是什么？
        end = np.datetime64(end_s, "ns")
        n = len(self.times)   # 7304   # 5年总时间步数？
        valid = []
        # anchor i uses input [i-input_steps+1, ..., i] and target [i+1, ..., i+forecast_steps]
        first_offset = self.input_steps - 1
        for i in range(first_offset, n - self.forecast_steps):
            first_time = self.times[i - first_offset]
            last_time = self.times[i + self.forecast_steps]
            if first_time >= start and last_time <= end:
                valid.append(i)
        if not valid:
            raise RuntimeError(f"No valid window found for split={self.split}")
        return np.asarray(valid, dtype=np.int64)  # np.asarray()??

    def _ensure_open(self):
        if self._ds is None:   # 怎么还需要读取一遍？
            self._ds = xr.open_zarr(self.zarr_path, consolidated=self.consolidated)
        return self._ds

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_ds"] = None  # xarray objects are opened lazily inside each worker.
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):   # 正式读取样本。。。用这个类的时候，这个idx也并没输入什么？ 这就是pytroch的Dataset类的用法吗？
        ds = self._ensure_open()
        i = int(self.indices[idx])

        # x_raw: [input_steps, n_input_vars, 13, 41, 71]
        x = ds["data"].isel(
            time=slice(i - self.input_steps + 1, i + 1),
            variable=self.input_var_idx,
        ).values.astype("float32")

        # y_raw: [forecast_steps, n_cloud_vars, 13, 41, 71]
        y = ds["data"].isel(
            time=slice(i + 1, i + 1 + self.forecast_steps),
            variable=self.target_var_idx,
        ).values.astype("float32")

        if self.normalize:
            cloud_names = set(CLOUD_VARIABLES)
            # 按云变量名划分。温度同时出现在 x 和 y 里，不能因为在 target 里就被 log 变换。
            basic_local = [
                j for j, name in enumerate(self.input_variables) if name not in cloud_names
            ]
            cloud_local = [
                j for j, name in enumerate(self.input_variables) if name in cloud_names
            ]
            basic_zarr_idx = [self.input_var_idx[j] for j in basic_local]

            x_basic = x[:, basic_local]
            x_mean = self.mean_all[basic_zarr_idx][None, :, :, None, None]
            x_std = self.std_all[basic_zarr_idx][None, :, :, None, None]
            x_basic = (x_basic - x_mean) / x_std

            x_cloud = x[:, cloud_local]
            q0 = 1e-8
            x_cloud = np.log10(1 + x_cloud / q0)
            x = np.concatenate((x_basic, x_cloud), axis=1)

            y_cloud_local = [
                j for j, name in enumerate(self.target_variables) if name in cloud_names
            ]
            y_temp_local = [
                j for j, name in enumerate(self.target_variables) if name not in cloud_names
            ]
            y_cloud = np.log10(1 + y[:, y_cloud_local] / q0)
            if y_temp_local:
                temp_zarr_idx = [self.target_var_idx[j] for j in y_temp_local]
                y_temp = y[:, y_temp_local]
                temp_mean = self.mean_all[temp_zarr_idx][None, :, :, None, None]
                temp_std = self.std_all[temp_zarr_idx][None, :, :, None, None]
                y_temp = (y_temp - temp_mean) / temp_std
                y = np.concatenate((y_cloud, y_temp), axis=1)
            else:
                y = y_cloud

        if self.flatten_time_variable:
            # [T, V, D, H, W] -> [T*V, D, H, W]
            x = np.ascontiguousarray(
                x.reshape(self.input_steps * len(self.input_variables), *x.shape[2:])
            )
            y = np.ascontiguousarray(
                y.reshape(self.forecast_steps * len(self.target_variables), *y.shape[2:])
            )
        meta = {
            "anchor_time": str(self.times[i]),
            "input_times": [str(t) for t in self.times[i - self.input_steps + 1:i + 1]],
            "target_start": str(self.times[i + 1]),
            "target_end": str(self.times[i + self.forecast_steps]),
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta     # 返回tensor数组




@dataclass   # 什么含义和作用？ 自动为类生成特殊方法如__init__、__repr__、__eq__等，简化了类的创建过程。
class DataLoaders:   # 什么意思？ 
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: ERA5CloudWindowDataset
    val_dataset: ERA5CloudWindowDataset
    test_dataset: ERA5CloudWindowDataset

def make_dataloaders(
    zarr_path: Union[str,Path],   # 
    batch_size: int = 4,
    num_workers: int = 0,  # 加载数据的子进程
    max_val_samples: int = 2048,  # 为什么是2048？ 2048怎么来的？   随便设置一个数字，意思是默认用2048个样本来检验模型。
    max_test_samples: Optional[int] = None,
    seed: int = 42,
    forecast_steps: int = 12,
    input_steps: int = 2,
    normalize: bool= True,
    target_variables: Iterable[str] = TARGET_VARIABLES,
) -> DataLoaders:
    train_ds = ERA5CloudWindowDataset(zarr_path, split="train", normalize=normalize, max_samples=None, input_steps=input_steps, forecast_steps=forecast_steps, sample_seed=seed, target_variables=target_variables)
    val_ds = ERA5CloudWindowDataset(zarr_path, split="val", normalize=normalize, max_samples=max_val_samples, input_steps=input_steps, forecast_steps=forecast_steps, sample_seed=seed, target_variables=target_variables)
    test_ds = ERA5CloudWindowDataset(zarr_path, split="test", normalize=normalize, max_samples=max_test_samples, input_steps=input_steps, forecast_steps=forecast_steps, sample_seed=seed, target_variables=target_variables)
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













