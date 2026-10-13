

from __future__ import annotations   # 没什么用？

import json
import shutil
from dataclasses import dataclass, field   # 这个库具体
from pathlib import Path
from typing import Sequence, Optional    # typing这个库，是专门用于方便写class代码用的吗？typing 不是专门给 class 用的，而是给 类型注解 用的。

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar   #使用的是 Dask 的进度条


# era5文件路径： /mnt/sdd/era5/sais2025/pressure_level
# era5_to_china_zarr.py
# 中国区域的范围  longitude: 70°E–140°E，latitude: 15°N–55°N


'''不用dataclass，直接写这个，我觉得还更加清晰一些。
def __init__(
    self,
    input_root,
    output_zarr,
    years=[2018, 2019, 2020, 2021, 2022],
    variables=[...],
    pressure_levels=[...],
    lat_range=(15.0, 55.0),
    lon_range=(70.0, 140.0),
    raw_time_dim="valid_time",
    time_dim="time",
    time_chunk=16,
    sort_latitude_ascending=True,
    engine=None,
    overwrite=True,
):
    self.input_root = input_root
    self.output_zarr = output_zarr
    ...
'''


@dataclass    # 这个装饰器是干嘛的？ 有什么作用？ 它是 Python 3.7 引入的语法糖。它可以自动帮你生成 __init__, __repr__, __eq__ 等方法。
class ERA5ChinaZarrBuilder:      # 这里为什么没有一个括号，来输入要用到的参数？  默认继承 Python 最基础的 object 类。  这个类没有显式继承其他父类
    """
    将 ERA5 pressure-level 日文件处理成中国区域连续时间序列 Zarr。

    输出核心变量：
        data[time, variable, pressure_level, latitude, longitude]

    其中：
        time: 所有年份所有天的 4 个时次，连续 6 小时
        variable: 9 个气象要素
        pressure_level: 13 个气压层
        latitude: 15N–55N，共约 41 个格点
        longitude: 70E–140E，共约 71 个格点
    """

    input_root: str | Path
    output_zarr: str | Path    # 这里默认就是self的属性吗？ 这是self属性定义的简洁形式？

    years: Sequence[int] = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022])   # field具体是什么？ 什么用法？   前面的Sequence[int]就是一个修饰性作用？ 可以直接删掉？ 
    '''field的用法：默认值不是直接用这个列表，而是每次调用这个 lambda 生成一个新列表。
    相当于：def make_default_years():
    return [2018, 2019, 2020, 2021, 2022]
    '''

    variables: Sequence[str] = field(default_factory=lambda: [   # 这应该是typing库中定义self属性的简便写法？？   这返回的是一个列表类型吗？  lambda？ 
        "geopotential",
        "temperature",
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_cloud_ice_water_content",
        "specific_cloud_liquid_water_content",
        "specific_rain_water_content",
        "specific_snow_water_content",
    ])

    pressure_levels: Sequence[int] = field(default_factory=lambda: [
        1000, 925, 850, 700, 600, 500, 400,
        300, 250, 200, 150, 100, 50
    ])

    lat_range: tuple[float, float] = (15.0, 55.0)
    lon_range: tuple[float, float] = (70.0, 140.0)

    raw_time_dim: str = "valid_time"   # ？
    time_dim: str = "time"

    time_chunk: int = 16    #将最后的数组，按照时间维度，分为16个小块吗？
    sort_latitude_ascending: bool = True

    engine: Optional[str] = None   # 
    overwrite: bool = True

    def __post_init__(self):   # ？
        self.input_root = Path(self.input_root)   # 将字符串路径转化为了Path对象
        self.output_zarr = Path(self.output_zarr)

        if not self.input_root.exists():
            raise FileNotFoundError(f"Input root does not exist: {self.input_root}")
        self.output_zarr.parent.mkdir(parents=True, exist_ok=True)

    def build(self):
        """
        主入口：
        1. 逐变量读取所有年份 daily nc
        2. 裁剪中国区域
        3. 拼接成 data[time, variable, pressure_level, latitude, longitude]
        4. 写入 Zarr
        5. 计算并追加 mean/std 标准化统计量
        """
        if self.output_zarr.exists():
            if self.overwrite:
                shutil.rmtree(self.output_zarr)    # 删除已存在的Zarr文件    删除整个文件夹  
            else:
                raise FileExistsError(f"Output already exists: {self.output_zarr}")

        variable_arrays = []   # 收集不同变量的多个dataset组成列表，最终通过concat组成一个更大的dataset。
        units_map = {}   # 存储每个变量的单位。
        source_name_map = {}   # 存储每个变量的真正气象变量名。 而不是self.variables中的路径的变量名。

        for var_name in self.variables:   # self.variables  ，  包含9个要素名称的列表
            '''遍历每个变量，然后进行处理。
            '''
            print(f"\n[INFO] Processing variable: {var_name}")

            files = self._collect_files(var_name)   # 收集所有类似的nc文件。？？  整理文件名为列表？
            if len(files) == 0:
                raise FileNotFoundError(f"No nc files found for variable: {var_name}")

            source_var = self._detect_data_var(files[0])   # 检测nc文件中的真正气象变量名。  # 返回第一个nc文件中的真正气象变量名。
            source_name_map[var_name] = source_var

            units_map[var_name] = self._read_units(files[0], source_var)   # 读取第一个nc文件中的真正气象变量名。

            print(f"[INFO] Found {len(files)} files")   # 打印找到的文件数量。
            print(f"[INFO] Source variable in nc: {source_var}")   # 打印真正气象变量名。
            print('开始加载数据')
            ds = xr.open_mfdataset(     # 打开多个类似的nc文件。   # 打开第一个nc文件。
                files,           # xr.open_mfdataset可以读取不同文件夹的多个nc文件吗？
                combine="by_coords",
                preprocess=self._make_preprocess_func(source_var),   # 这是在干嘛？  调用class的一个函数，返回一个前处理函数，
                parallel= False,     #True,   改为false，不在出现segment false。
                engine=self.engine,   # enging 是none ,  为什么我的ide里面，指定engine = 'None'报错：unrecognized engine None must be one of: ['netcdf4', 'store', 'zarr']？？？， 用engine = None，这里的意思是：让 xarray 自动选择合适的读取引擎。
                chunks={     # ?    不懂chunks 的参数具体什么含义？什么作用？？ 分块？？
                    self.raw_time_dim: 4,      # self.raw_time_dim: "valid time"，调用的是era5原来数据的时间维度名称。 这个具体什么作用？  
                    "pressure_level": len(self.pressure_levels),   # 气压层维度的切块的大小，是气压层的层数。     所以最终每4*6  =24个时次切一次，在每个气压层也切一次，这样来区分一个chunk。
                },
            )
            '''    这里存在问题，可能是无法一次性读取并加载完所有的nc文件。。  
            Segmentation fault (core dumped)
            这不是你的 Python 逻辑写错了（Python 语法错误通常报 ValueError、KeyError 等），而是底层的 C/C++ 库（如 NetCDF4、HDF5）在读写内存时崩溃了。
              
            '''
            if source_var not in ds.data_vars:
                raise KeyError(f"{source_var} not found after open_mfdataset for {var_name}")

            da = ds[source_var]    # da就是该变量的所有时次，高度的数组了。

            # 统一变量名为文件夹名，方便后续 variable 坐标索引
            da = da.rename(var_name)

            # 统一维度顺序
            da = da.transpose(   # 更换维度顺序， Time,level, lat,lon
                self.time_dim,
                "pressure_level",
                "latitude",
                "longitude",
            )

            # 转 float32，节省空间
            da = da.astype("float32")

            # 添加 variable 维
            da = da.expand_dims(variable=[var_name])   #之后在利用variable维concat。

            variable_arrays.append(da)

        print("\n[INFO] Concatenating all variables...")

        data = xr.concat(
            variable_arrays,
            dim="variable",
            join="exact",
        )

        data = data.rename("data")

        # 最终维度顺序：time, variable, pressure_level, latitude, longitude
        data = data.transpose(
            self.time_dim,
            "variable",
            "pressure_level",
            "latitude",
            "longitude",
        )

        data = data.chunk({    # 将数据分块，便于后续计算。   dataset的chunk属性？？
            self.time_dim: self.time_chunk,   # self.time_chunk : 16   ， 每16个时次放进一个chunk。
            "variable": len(self.variables),
            "pressure_level": len(self.pressure_levels),  #13 层整体放进一个 chunk，不切开
            "latitude": -1,   # 表示这一维全部放进一个 chunk
            "longitude": -1,
        })     
        '''#？这里各个维度的设置具体什么含义？
        时间维 (time)：每 16 个时次切一刀。

变量维 (variable)：不切（所有变量打包在一起）。

气压层 (pressure_level)：不切（所有高度层打包在一起）。

纬度 (latitude)：不切。

经度 (longitude)：不切。
        '''

        self._check_time_continuity(data[self.time_dim].values)

        ds_out = xr.Dataset({"data": data})

        ds_out.attrs["description"] = (
            "ERA5 pressure-level variables cropped to China region "
            "and merged into a continuous time-series Zarr dataset."
        )
        ds_out.attrs["region"] = json.dumps({   # json.dumps() 会把 Python 对象（object）转成字符串：，转成 JSON 字符串最稳 
            "longitude": list(self.lon_range),
            "latitude": list(self.lat_range),
        })
        ds_out.attrs["years"] = json.dumps(list(self.years))
        ds_out.attrs["dimension_order"] = "time, variable, pressure_level, latitude, longitude"
        ds_out.attrs["variable_units"] = json.dumps(units_map, ensure_ascii=False)
        ds_out.attrs["source_variable_names"] = json.dumps(source_name_map, ensure_ascii=False)

        channel_names = [
            f"{var}_{lev}hPa"
            for var in self.variables
            for lev in self.pressure_levels
        ]
        ds_out.attrs["channel_names_variable_level_order"] = json.dumps(   # ?
            channel_names,
            ensure_ascii=False,
        )

        print("\n[INFO] Writing main data to Zarr...")
        print(f"[INFO] Output path: {self.output_zarr}")

        with ProgressBar():   # 显示进度条？
            ds_out.to_zarr(
                self.output_zarr,
                mode="w",
                consolidated=True,   # 是否合并小文件？ ？ 什么意思？consolidated=True 会把所有小文件的元数据（Metadata）统合打包成一个单独的 .zmetadata 文件。以后读取时，只需秒读这个文件就能掌握全局，速度提升几十倍。
            )

        print("\n[INFO] Main Zarr written successfully.")

        print("\n[INFO] Computing normalization statistics...")
        self.compute_and_append_stats()

        print("\n[DONE] ERA5 China Zarr build finished.")
        print(f"[DONE] Output: {self.output_zarr}")

    def _collect_files(self, var_name: str) -> list[str]:   # 函数名前用_是为了表示不要再类之外用这个函数吗？ 私有函数？  这习惯上表面，这是类内部使用的辅助函数，不建议外部直接调用。
        '''具体在干嘛？ 收集所有类似的nc文件。？？  整理文件名为列表？
        就是为了在某一个变量，遍历所有年份，收集所有类似的nc文件。
        然后返回所有nc文件的列表。？？
        '''
        files = []   # 收集所有类似的nc文件。？？  整理文件名为列表？   收集某个要素，所有年份的nc文件。
        for year in self.years:
            year_dir = self.input_root / var_name / str(year)   # 年份文件夹路径...    为什么这里我在ide里面报错unsupported operand type(s) for /，  而这里却可以直接合并为路径字符串？   # 最终形式：'/mnt/sdd/era5/sais2025/pressure_level/geopotential/2018'
            # self.input_root 是 Path 对象！！PosixPath('/mnt/sdd/era5/sais2025/pressure_level')
            if not year_dir.exists():   # 年份文件夹不存在？
                raise FileNotFoundError(f"Year directory not found: {year_dir}")

            year_files = sorted(year_dir.glob("*.nc"))   # 年份文件夹中的所有nc文件。 是所有nc文件夹的路径！   year_dir具体是什么？  为什么我在ide里面year_dir是str，且没有glob属性？
            '''  .glob是Path的属性吗？   最终的year_files : 
            '), PosixPath('/mnt/sdd/era5/sais2025/pressure_level/geopotential/2018/20181226.nc'), PosixPath('/mnt/sdd/era5/sais2025/pressure_level/geopotential/2018/20181227.nc'), PosixPath('/mnt/sdd/era5/sais2025/pressure_level/geopotential/2018/20181228.nc'), PosixPath('/mnt/sdd/era5/sais2025/pressure_level/geopotential/2018/20181229.nc'), PosixPath('/mnt/sdd/era5/sais2025/pressure_level/geopotential/2018/20181230.nc'), PosixPath('/mnt/sdd/era5/sais2025/pressure_level/geopotential/2018/20181231.nc')]
            '''
            if len(year_files) == 0:
                raise FileNotFoundError(f"No nc files found in: {year_dir}")   # 

            files.extend(year_files)   # 将年份文件夹中的所有nc文件添加到列表中。

        return [str(f) for f in files]   # 返回所有nc文件的列表。

    def _detect_data_var(self, file_path: str) -> str:
        """  挑选第一个文件，读取其中的变量名
        自动识别 nc 文件里的真正气象变量名。
        例如文件夹叫 temperature，但 nc 内部变量可能叫 t。
        """
        with xr.open_dataset(file_path, engine=self.engine) as ds:
            candidates = []
            for name, da in ds.data_vars.items():    # 返回维度名称和数组。?   这里应该有问题。   如果要检测一个变量的nc文件中，真正的各维度的变量名？
                '''ds.data_vars
Data variables:
    number   int64 ...
    expver   (valid_time) object ...
    u        (valid_time, pressure_level, latitude, longitude) float32 ...
    只有变量u地da.dims才会包含required_dims中的四个要素  ：  ('valid_time', 'pressure_level', 'latitude', 'longitude')

                '''
                dims = set(da.dims)
                required_dims = {  # 这是个集合。
                    self.raw_time_dim,   # ??
                    "pressure_level",
                    "latitude",
                    "longitude",
                }
                if required_dims.issubset(dims):  #  ? dims必须要是required_dims的子集？  检验required_dims 是否是 dims 的子集？
                    candidates.append(name)   # name : u

            if len(candidates) == 0:
                raise ValueError(f"No valid 4D data variable found in {file_path}")

            if len(candidates) > 1:
                print(f"[WARN] Multiple candidate variables in {file_path}: {candidates}")
                print(f"[WARN] Use first candidate: {candidates[0]}")

            return candidates[0]

    def _read_units(self, file_path: str, source_var: str) -> str:
        with xr.open_dataset(file_path, engine=self.engine) as ds:
            return str(ds[source_var].attrs.get("units", "unknown"))

    def _make_preprocess_func(self, source_var: str):
        """
        给 open_mfdataset 使用的 preprocess：
        1. 只保留目标变量
        2. valid_time 改名为 time
        3. 裁剪 pressure level
        4. 裁剪中国区域
        5. 可选：纬度升序排列
        """
        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            # 只保留目标变量，减少无关数据。   就是裁剪一下经纬度。
            ds = ds[[source_var]]

            # valid_time -> time
            if self.raw_time_dim in ds.dims or self.raw_time_dim in ds.coords:
                ds = ds.rename({self.raw_time_dim: self.time_dim})

            # 选择 13 个气压层
            ds = ds.sel(pressure_level=list(self.pressure_levels))

            # 经度裁剪
            lon_min, lon_max = self.lon_range
            lon_values = ds["longitude"].values

            if lon_values[0] <= lon_values[-1]:   # 先判断一下，在进行裁剪。  只能是这一行的情况，else什么情况应该不会出现。
                ds = ds.sel(longitude=slice(lon_min, lon_max))
            else:
                ds = ds.sel(longitude=slice(lon_max, lon_min))

            # 纬度裁剪
            lat_min, lat_max = self.lat_range
            lat_values = ds["latitude"].values

            if lat_values[0] <= lat_values[-1]:  # 纬度数组是升序。
                ds = ds.sel(latitude=slice(lat_min, lat_max))   # 因为slice的第一个数值必须要小于第二个数值
            else:
                ds = ds.sel(latitude=slice(lat_max, lat_min))

            if self.sort_latitude_ascending:
                ds = ds.sortby("latitude")

            return ds

        return preprocess

    def _check_time_continuity(self, times: np.ndarray):
        """
        检查时间是否为严格 6 小时间隔。
        """
        if len(times) < 2:
            raise ValueError("Time coordinate has fewer than 2 records.")

        times_h = times.astype("datetime64[h]")
        diffs = np.diff(times_h)

        expected = np.timedelta64(6, "h")
        bad = np.where(diffs != expected)[0]

        print(f"\n[INFO] Total time steps: {len(times)}")
        print(f"[INFO] Start time: {times[0]}")
        print(f"[INFO] End time:   {times[-1]}")

        if len(bad) > 0:
            first_bad = bad[0]
            raise ValueError(
                "Time coordinate is not continuous at 6-hour interval.\n"
                f"First bad index: {first_bad}\n"
                f"time[{first_bad}] = {times[first_bad]}\n"
                f"time[{first_bad + 1}] = {times[first_bad + 1]}\n"
                f"diff = {diffs[first_bad]}"
            )

        print("[INFO] Time continuity check passed: all intervals are 6 hours.")

    def compute_and_append_stats(self):
        """
        计算标准化统计量：
            mean[variable, pressure_level]   # 9个要素的13个层各自计算，平均值和标准差
            std[variable, pressure_level]

        统计维度：
            time, latitude, longitude

        注意：
            这里不改变原始 data，只把 mean/std 存进同一个 Zarr。
        """
        ds = xr.open_zarr(
            self.output_zarr,
            consolidated=True,   # 为什么读取zarr也需要用到consolidated = True??  请优先读取 consolidated metadata，也就是 .zmetadata。如果写入时有 consolidated，读取时用 consolidated=True 是推荐做法。
            chunks={self.time_dim: self.time_chunk},  # self.time_dim : "time",  self.time_chunk : 16
        )      # 这里直接读取并修改这个zarr文件以添加元数据
        '''>>> ds = xr.open_zarr(
...     output_zarr,
...     consolidated=True,   # 为什么读取zarr也需要用到consolidated = True??
...     chunks={"time": time_chunk},  # self.time_dim : "time",  self.time_chunk : 16
... )
>>> ds
<xarray.Dataset>
Dimensions:         (time: 7304, variable: 9, pressure_level: 13, latitude: 41,
                     longitude: 71)
Coordinates:
  * latitude        (latitude) float64 15.0 16.0 17.0 18.0 ... 53.0 54.0 55.0
  * longitude       (longitude) float64 70.0 71.0 72.0 ... 138.0 139.0 140.0
  * pressure_level  (pressure_level) float64 1e+03 925.0 850.0 ... 100.0 50.0
  * time            (time) datetime64[ns] 2018-01-01 ... 2022-12-31T18:00:00
  * variable        (variable) object 'geopotential' ... 'specific_snow_water...
Data variables:
    data            (time, variable, pressure_level, latitude, longitude) float32 dask.array<chunksize=(16, 9, 13, 41, 71), meta=np.ndarray>
    mean            (variable, pressure_level) float32 dask.array<chunksize=(9, 13), meta=np.ndarray>
    std             (variable, pressure_level) float32 dask.array<chunksize=(9, 13), meta=np.ndarray>

        '''

        data = ds["data"]

        mean = data.mean(
            dim=(self.time_dim, "latitude", "longitude"),
            skipna=True,
        ).astype("float32").rename("mean")

        std = data.std(
            dim=(self.time_dim, "latitude", "longitude"),
            skipna=True,
        ).astype("float32").rename("std")

        # 防止某些变量/层 std 极小导致除零   ？
        std = xr.where(std < 1e-12, 1.0, std).astype("float32")   # 将小于10的-12次方的值换成1吗？✅
        std = std.rename("std")

        stats_ds = xr.Dataset({     #？
            "mean": mean,
            "std": std,
        })

        stats_ds["mean"].attrs["description"] = (
            "Mean for normalization, computed over time, latitude, longitude."
        )
        stats_ds["std"].attrs["description"] = (
            "Standard deviation for normalization, computed over time, latitude, longitude."
        )

        with ProgressBar():
            stats_ds.to_zarr(
                self.output_zarr,
                mode="a",   # 这里直接在原数据的基础上，dataset添加dataarray？？！！
                consolidated=True,
            )

        print("[INFO] mean/std appended to Zarr.")


if __name__ == "__main__":
    builder = ERA5ChinaZarrBuilder(
        input_root="/mnt/sdd/era5/sais2025/pressure_level",
        output_zarr="/mnt/sdf/common_data/sais2025/dataset/era5_china_2018_2022.zarr",
        years=[2018, 2019, 2020, 2021, 2022],
        lat_range=(15.0, 55.0),
        lon_range=(70.0, 140.0),
        time_chunk=16,
        overwrite=True,
    )

    builder.build()   #？  这个就是调用了ERA5ChinaZarrBuilder这个实例的build函数了。






'''运行记录
(torch_learn) root@ubuntu:/home/yx/projects/common_data/sais2025/test1# python3 dataset_generate_1.py 

[INFO] Processing variable: geopotential
[INFO] Found 1826 files
[INFO] Source variable in nc: z
开始加载数据



[INFO] Processing variable: temperature
[INFO] Found 1826 files
[INFO] Source variable in nc: t
开始加载数据



[INFO] Processing variable: specific_humidity
[INFO] Found 1826 files
[INFO] Source variable in nc: q
开始加载数据

[INFO] Processing variable: u_component_of_wind
[INFO] Found 1826 files
[INFO] Source variable in nc: u
开始加载数据



[INFO] Processing variable: v_component_of_wind
[INFO] Found 1826 files
[INFO] Source variable in nc: v
开始加载数据

[INFO] Processing variable: specific_cloud_ice_water_content
[INFO] Found 1826 files
[INFO] Source variable in nc: ciwc
开始加载数据

[INFO] Processing variable: specific_cloud_liquid_water_content
[INFO] Found 1826 files
[INFO] Source variable in nc: clwc
开始加载数据

[INFO] Processing variable: specific_rain_water_content
[INFO] Found 1826 files
[INFO] Source variable in nc: crwc
开始加载数据

[INFO] Processing variable: specific_snow_water_content
[INFO] Found 1826 files
[INFO] Source variable in nc: cswc
开始加载数据

[INFO] Concatenating all variables...

[INFO] Total time steps: 7304
[INFO] Start time: 2018-01-01T00:00:00.000000000
[INFO] End time:   2022-12-31T18:00:00.000000000
[INFO] Time continuity check passed: all intervals are 6 hours.

[INFO] Writing main data to Zarr...
[INFO] Output path: /mnt/sdf/common_data/sais2025/dataset/era5_china_2018_2022.zarr
[########################################] | 100% Completed | 20m 47s

[INFO] Main Zarr written successfully.

[INFO] Computing normalization statistics...
[########################################] | 100% Completed | 12.56 s
[INFO] mean/std appended to Zarr.

[DONE] ERA5 China Zarr build finished.
[DONE] Output: /mnt/sdf/common_data/sais2025/dataset/era5_china_2018_2022.zarr

'''



'''
ide变量输入
input_root = "/mnt/sdd/era5/sais2025/pressure_level"
output_zarr = "/mnt/sdf/common_data/sais2025/dataset/era5_china_2018_2022.zarr"
years=[2018, 2019, 2020, 2021, 2022]
lat_range=(15.0, 55.0)
lon_range=(70.0, 140.0)
time_chunk=16
>>> variables = [        "geopotential",
...         "temperature",
...         "specific_humidity",
...         "u_component_of_wind",
...         "v_component_of_wind",
...         "specific_cloud_ice_water_content",
...         "specific_cloud_liquid_water_content",
...         "specific_rain_water_content",
...         "specific_snow_water_content",
... ]


'''
