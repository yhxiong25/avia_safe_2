


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Iterable, Optional,Union   # 这两个库，真要认真看看。
import numpy as np
from dataclasses import dataclass
import xarray as xr

'''
NOTE  顺序都是按照 CIWC,CLWC,CRWC,CSWC来的吗？

1，需要把这些代码的逻辑逐行吃透！！
'''



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

CLOUD_VARIABLES = [    # 顺序都是按照 CIWC,CLWC,CRWC,CSWC来的？
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


'''
1,默认在Dataloaer里面就完成了数据标准化。
'''


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
        zarr_path: Union[str,Path],    # | Path,    # python版本不适配这种写法。  str | Path → (改成)   Union[str, Path]
        split: str,
        input_steps: int = 2,
        forecast_steps: int = 12,
        input_variables: Iterable[str] = INPUT_VARIABLES,
        target_variables: Iterable[str] = CLOUD_VARIABLES,
        normalize: bool = True,    # 默认用DataLoader加载数据的时候就完成了标准化。
        flatten_time_variable: bool = True,
        max_samples: Optional[int] = None,  #这个参数可以是 int，也可以是 None。
        sample_seed: int = 42,    # 随机抽取样本的随机种子。
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

        ds = xr.open_zarr(self.zarr_path, consolidated=self.consolidated)   # 这里报错。 
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
        n = len(self.times)   # 7304
        valid = []
        # anchor i uses input [i-1, i] and target [i+1, ..., i+forecast_steps]
        for i in range(1, n - self.forecast_steps):   # 把所有的时间的idx提取出来
            first_time = self.times[i - 1]
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

        # x_raw: [2, 9, 13, 41, 71]
        x = ds["data"].isel(
            time=slice(i - 1, i + 1),   # slice不包括最后一项，所以这里索引的是 i-1和i
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

        meta = {    # ？ 包含某个样本的anchor时刻，起始时刻的信息
            "anchor_time": str(self.times[i]),
            "input_times": [str(t) for t in self.times[i - 1:i + 1]],
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


'''dataLoader涉及到程序就会变得非常慢。  查看程序发现卡在多线程，也没有时间进度：
torch_learn) root@ubuntu:/home/yx/projects/common_data/sais2025/test1# ps -ef | grep gan
root      656487  347887  0 21:35 pts/64   00:00:08 python3 -u train_GAN_test1.py --zarr_path /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr --save_dir ./logs/GAN_test1/ --gan_epochs 50 --batch_size 4
root      656591  656487  0 21:35 pts/64   00:00:00 python3 -u train_GAN_test1.py --zarr_path /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr --save_dir ./logs/GAN_test1/ --gan_epochs 50 --batch_size 4
root      656592  656487  0 21:35 pts/64   00:00:00 python3 -u train_GAN_test1.py --zarr_path /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr --save_dir ./logs/GAN_test1/ --gan_epochs 50 --batch_size 4
root      656594  656487  0 21:35 pts/64   00:00:00 python3 -u train_GAN_test1.py --zarr_path /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr --save_dir ./logs/GAN_test1/ --gan_epochs 50 --batch_size 4
root      656597  656487  0 21:35 pts/64   00:00:00 python3 -u train_GAN_test1.py --zarr_path /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr --save_dir ./logs/GAN_test1/ --gan_epochs 50 --batch_size 4
root      696200  347887  0 22:58 pts/64   00:00:00 grep --color=auto gan


'''


def make_dataloaders(
    zarr_path: Union[str,Path],   # 
    batch_size: int = 4,
    num_workers: int = 4,  # 加载数据的子进程
    max_val_samples: int = 2048,  # 为什么是2048？ 2048怎么来的？   随便设置一个数字，意思是默认用2048个样本来检验模型。
    max_test_samples: Optional[int] = None,
    seed: int = 42,
) -> DataLoaders:
    train_ds = ERA5CloudWindowDataset(zarr_path, split="train", max_samples=None, sample_seed=seed)
    val_ds = ERA5CloudWindowDataset(zarr_path, split="val", max_samples=max_val_samples, sample_seed=seed)
    test_ds = ERA5CloudWindowDataset(zarr_path, split="test", max_samples=max_test_samples, sample_seed=seed)    # 这又返回什么？ 为什么还要用DataLoader包装一下？  为什么shuffle要在Dataloader里面指定参数？

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



