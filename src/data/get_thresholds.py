#GPT写的提取阈值函数：

'''一般选择75%的数值的小数点第二位向上取整的值作为thresholds。
4个云变量的顺序会是怎样的呢
'''
import numpy as np
import xarray as xr
import torch


CLOUD_VARIABLES = [
    "specific_cloud_ice_water_content",     # CIWC
    "specific_cloud_liquid_water_content",  # CLWC
    "specific_rain_water_content",          # CRWC
    "specific_snow_water_content",          # CSWC
]

CLOUD_SHORT_NAMES = ["CIWC", "CLWC", "CRWC", "CSWC"]


def ceil_to_sigfig(x: float, sigfig: int = 2) -> float:
    """
    将数值 x 按有效数字向上取整。
    例如：
        8.83e-06 -> 8.9e-06
        3.12e-05 -> 3.2e-05
        2.01e-07 -> 2.1e-07
    参数
    ----
    x : float
        原始数值。
    sigfig : int
        保留的有效数字位数，默认 2。
    返回
    ----
    float
        向上取整后的数值。
    """
    if x == 0 or not np.isfinite(x):
        return x
    sign = 1.0 if x > 0 else -1.0
    x_abs = abs(x)
    # 例如 x=8.83e-06，则 exponent=-6
    exponent = np.floor(np.log10(x_abs))
    # scale 用来把 x 移动到需要取整的位置
    scale = 10 ** (sigfig - 1 - exponent)
    # 对正数向上取整
    x_ceil = np.ceil(x_abs * scale) / scale
    return sign * x_ceil


def get_thresholds(
    level_hpa: float,
    zarr_path: str = "/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr",
    percentile: float = 75.0,
    sigfig: int = 2,
    cloud_variables=CLOUD_VARIABLES,
    cloud_short_names=CLOUD_SHORT_NAMES,
    consolidated: bool = True,
):
    """
    根据指定高度层，计算四个云变量的原始阈值和标准化阈值。
    阈值定义：
        对每个云变量，在指定高度层上提取所有非零值；
        计算这些非零值的 percentile 分位数；
        再按 sigfig 位有效数字向上取整。
    标准化阈值：
        alpha_norm = (alpha_raw - mean) / std
    注意：
        因为 DataLoader 中的 y 已经标准化，
        所以后续 MRdiff / FARdiff loss 中必须使用 alpha_norm，
        不能直接使用 alpha_raw。
    参数
    ----
    level_hpa : float
        目标气压层，例如 500, 200, 850。
    zarr_path : str
        era5_china_2018_2022.zarr 路径。
    percentile : float
        分位数，默认 75。
    sigfig : int
        有效数字位数，默认 2。
    cloud_variables : list[str]
        四个云变量在 zarr 中的完整变量名。
    cloud_short_names : list[str]
        四个云变量简称，顺序必须与 cloud_variables 一致。
    consolidated : bool
        是否使用 zarr consolidated metadata。
    返回
    ----
    result : dict
        包含：
            level_requested
            level_actual
            level_idx
            raw_thresholds_dict
            norm_thresholds_dict
            raw_thresholds_list
            norm_thresholds_list
            norm_thresholds_tensor
            details
    """
    ds = xr.open_zarr(
        zarr_path,
        consolidated=consolidated,
        chunks=None,
    )
    pressure_levels = np.asarray(ds["pressure_level"].values, dtype=np.float32)
    level_idx = int(np.argmin(np.abs(pressure_levels - level_hpa)))
    level_actual = float(pressure_levels[level_idx])
    var_names = [str(v) for v in ds["variable"].values]
    name_to_idx = {name: i for i, name in enumerate(var_names)}
    raw_thresholds_dict = {}
    norm_thresholds_dict = {}
    raw_thresholds_list = []
    norm_thresholds_list = []
    details = {}
    for full_name, short_name in zip(cloud_variables, cloud_short_names):
        if full_name not in name_to_idx:
            raise ValueError(f"变量 {full_name} 不在 zarr 的 variable 坐标中。")
        var_idx = name_to_idx[full_name]
        # ----------------------------------------------------
        # 1. 读取该变量、该高度层的所有中国区域数据
        #    arr shape: [time, latitude, longitude]
        # ----------------------------------------------------
        arr = ds["data"].isel(
            variable=var_idx,
            pressure_level=level_idx,
        ).values.astype(np.float32)
        total_count = arr.size
        finite_mask = np.isfinite(arr)
        finite_count = int(finite_mask.sum())
        # 只取非零正值
        positive_values = arr[finite_mask & (arr > 0)]
        positive_count = int(positive_values.size)
        positive_ratio = positive_count / finite_count if finite_count > 0 else np.nan
        if positive_count == 0:
            raise ValueError(f"{short_name} at {level_actual} hPa 没有非零正值，无法计算阈值。")
        # ----------------------------------------------------
        # 2. 计算非零值的 75% 分位数
        # ----------------------------------------------------
        q_raw = float(np.nanpercentile(positive_values, percentile))
        # ----------------------------------------------------
        # 3. 按 2 位有效数字向上取整
        # ----------------------------------------------------
        threshold_raw = float(ceil_to_sigfig(q_raw, sigfig=sigfig))
        # ----------------------------------------------------
        # 4. 读取该变量、该高度层的 mean/std
        # ----------------------------------------------------
        mean = float(ds["mean"].isel(
            variable=var_idx,
            pressure_level=level_idx,
        ).values)
        std = float(ds["std"].isel(
            variable=var_idx,
            pressure_level=level_idx,
        ).values)
        std_safe = max(std, 1e-12)
        # ----------------------------------------------------
        # 5. 标准化阈值
        #    DataLoader 中 y_norm = (y_raw - mean) / std
        #    所以 alpha_norm = (alpha_raw - mean) / std
        # ----------------------------------------------------
        threshold_norm = float((threshold_raw - mean) / std_safe)
        key = f"{short_name}_{int(level_actual)}"
        raw_thresholds_dict[key] = threshold_raw
        norm_thresholds_dict[key] = threshold_norm
        raw_thresholds_list.append(threshold_raw)
        norm_thresholds_list.append(threshold_norm)
        details[key] = {
            "full_variable_name": full_name,
            "variable_index": int(var_idx),
            "level_hpa": level_actual,
            "level_index": int(level_idx),
            "total_count": int(total_count),
            "finite_count": int(finite_count),
            "positive_count": int(positive_count),
            "positive_ratio": float(positive_ratio),
            f"raw_p{percentile:g}_nonzero": q_raw,
            "raw_threshold_ceil_sigfig": threshold_raw,
            "mean": mean,
            "std": std,
            "norm_threshold": threshold_norm,
        }
        print(
            f"[INFO] {key}: "
            f"p{percentile:g}={q_raw:.8e}, "
            f"threshold_raw={threshold_raw:.8e}, "
            f"mean={mean:.8e}, std={std:.8e}, "
            f"threshold_norm={threshold_norm:.6f}, "
            f"positive_ratio={positive_ratio*100:.2f}%"
        )
        # 释放当前变量数组
        del arr, positive_values
    result = {
        "level_requested": float(level_hpa),
        "level_actual": level_actual,
        "level_idx": int(level_idx),
        # 字典形式，便于打印和检查
        "raw_thresholds_dict": raw_thresholds_dict,
        "norm_thresholds_dict": norm_thresholds_dict,
        # list 顺序固定为 [CIWC, CLWC, CRWC, CSWC]
        "raw_thresholds_list": raw_thresholds_list,
        "norm_thresholds_list": norm_thresholds_list,
        # tensor 顺序固定为 [CIWC, CLWC, CRWC, CSWC]
        "norm_thresholds_tensor": torch.as_tensor(norm_thresholds_list, dtype=torch.float32),
        # 详细统计信息
        "details": details,
    }
    return result

'''
thresholds_500 = get_thresholds(
    level_hpa=500,
    zarr_path="/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr",
    percentile=75.0,
    sigfig=2,
)

检验：THRESHOLDS = {"CIWC_500":8.9e-06,"CLWC_500":3.2e-05,"CRWC_500":3.0e-07,"CSWC_500":2.2e-05}   
[INFO] CIWC_500: p75=8.82148743e-06, threshold_raw=8.90000000e-06, mean=3.43666193e-06, std=1.20734212e-05, threshold_norm=0.452510, positive_ratio=37.82%
[INFO] CLWC_500: p75=3.14712524e-05, threshold_raw=3.20000000e-05, mean=8.19737306e-06, std=3.12327138e-05, threshold_norm=0.762106, positive_ratio=28.14%
^[[A[INFO] CRWC_500: p75=2.90572643e-07, threshold_raw=3.00000000e-07, mean=1.66341607e-08, std=1.37515269e-06, threshold_norm=0.206061, positive_ratio=1.03%
[INFO] CSWC_500: p75=2.11000443e-05, threshold_raw=2.20000000e-05, mean=1.51820313e-05, std=8.67405470e-05, threshold_norm=0.078602, positive_ratio=40.50%
检验通过✅

'''





