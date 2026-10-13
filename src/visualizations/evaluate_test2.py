


'''
决定自己重写评价模型效果的源代码
不用一个大类，先用函数，来挨个尝试
'''

import os
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

import torch
import matplotlib
matplotlib.use("Agg")   # 服务器环境下建议使用无界面后端
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm    # ？
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.append("/mnt/sdf/common_data/sais2025/era5_cloud_gan")
from src.data.data_loader_v2 import make_dataloaders
from src.models.generator import UNet3DGenerator
from src.models.generator_convGRU import ConvGRUUNet3DGenerator 
from src.metrics.metrics import csi_score    # 除了csi score，还有其它score吗？ 

from src.data.fixed_info import INPUT_VARIABLES,CLOUD_VARIABLES,CLOUD_SHORT_NAMES,\
                                THRESHOLDS,THRESHOLDS_normalized,LEVELS



'''
模型评估任务的待办：TODO
1，/mnt/sdf/common_data/sais2025/test1/src/visualizations/evaluate_test2.py:76: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
这个报错future warning什么含义？ 怎么解决？ 
2，

'''






class NormallyGANmodelEvaluation:
    '''
    '''
    def __init__(self,
        model_path:Union[str, Path],   # 必须要指定模型路径
        zarr_path: str = "/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr",
        levels: list = [850,500,250],    # 各种指标评定的高度层。一般这三个高度就够了
        device: str = "cuda:0",
        max_test_samples: Optional[int] = None,   # 默认使用全部测试集
        seed: int=42,
        lead_steps: int = 12,
        generator_version: str='unet'):
        self.model_path = Path(model_path)
        self.eval_type = self.model_path.name.split('.')[0]   # 'best_gan'   # 评价模型的名称。 
        self.save_dir = self.model_path.parent / (self.eval_type+'_evaluation')    # 要写成    # PosixPath('/home/yx/projects/common_data/sais2025/test1/logs/GAN_test7_modifiedADVLoss_5/evaluation')
        self.zarr_path = zarr_path
        self.ds = xr.open_zarr(zarr_path, consolidated=True, chunks=None)
        self.times = pd.to_datetime(self.ds['time'].values)
        self.batch_size = 4  # 用于dataloader
        self.levels = levels
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_test_samples = max_test_samples
        self.data = self.ds["data"]
        self.seed = seed
        ## self.save_dir.mkdir(parents=True, exist_ok=True)   # 这里可能回改变save_dir的属性   # parents=True   父目录不在，可以自己创建。
        self.split = 'test'   # 也可能会用这个类，来检验，验证集啥的
        # 存放，各个高度，各个要素，各个指标的csv文件
        self.lead_steps = lead_steps   # 所用的时间步长可能不一样
        self.generator_version = generator_version


    def load_generator(self):  # 我记得这个self不用写的
        if self.generator_version == 'unet':
            model = UNet3DGenerator(
                in_channels=18,
                out_channels=4*self.lead_steps,
                base_channels=32,    # 
            ).to(self.device)
        elif self.generator_version == 'convgru':
            model = ConvGRUUNet3DGenerator(in_channels=18,out_channels=4*self.lead_steps,base_channels=32,input_steps=2,input_vars=9,output_vars=4,forecast_steps=self.lead_steps).to(self.device)
        ckpt = torch.load(self.model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "generator" in ckpt:   # 一般就这种情况
            state_dict = ckpt["generator"]
        else:
            state_dict = ckpt
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print(f"[INFO] Generator loaded from: {self.model_path}")
        return model

    def get_test_dataloader(self):
        loaders = make_dataloaders(
            self.zarr_path,
            batch_size=self.batch_size,
            max_test_samples=self.max_test_samples,   # None应该就是全部test样本
            seed=self.seed,
            num_workers = 0,  # 不知道为什么必须设为0才行了？
            forecast_steps = self.lead_steps,
            target_variables=CLOUD_VARIABLES,
        )
        # print(f"[INFO] test samples: {len(loaders.test_dataset)}")
        return loaders.test



    def get_meanALL_df(self):
        '''
        返回一种这样的df：
        >>> df_mean
                                             1000          925           850           700   ...          200           150           100           50  
        geopotential                         1.208358e+03  7.654873e+03  1.457803e+04  3.014869e+04  ...  1.185192e+05  1.364204e+05  1.607488e+05  2.019949e+05
        temperature                          2.909865e+02  2.869489e+02  2.830936e+02  2.746290e+02  ...  2.201068e+02  2.133398e+02  2.056925e+02  2.118178e+02
        specific_humidity                    8.273488e-03  7.304224e-03  6.170044e-03  3.850811e-03  ...  3.230363e-05  7.596306e-06  2.946524e-06  2.860751e-06
        u_component_of_wind                 -3.840731e-03  4.638453e-01  1.513583e+00  4.443351e+00  ...  2.229187e+01  2.085185e+01  1.406110e+01  4.178991e+00
        v_component_of_wind                 -2.008481e-01 -3.703504e-02 -4.622418e-02 -3.487780e-01  ... -8.333771e-02 -2.590651e-01 -7.192123e-01  3.889771e-02
        specific_cloud_ice_water_content     1.257143e-07  2.129657e-07  4.325276e-07  9.575366e-07  ...  4.081753e-06  2.123216e-06  1.806552e-07  1.562104e-13
        specific_cloud_liquid_water_content  5.440921e-06  1.005669e-05  1.512688e-05  1.388681e-05  ...  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00
        specific_rain_water_content          3.131947e-06  3.275106e-06  3.291871e-06  2.882707e-06  ...  1.255791e-15  0.000000e+00  0.000000e+00  0.000000e+00
        specific_snow_water_content          2.006183e-06  2.273033e-06  2.775140e-06  4.676890e-06  ...  1.480716e-06  3.546491e-07  1.464212e-08  1.242061e-14
        '''
        data_loader = self.get_test_dataloader()
        df_mean = pd.DataFrame(data_loader.dataset.mean_all, columns=LEVELS,index=INPUT_VARIABLES)   # 各个要素在各个高度的平均值。 9×13的dataFrame
        df_std = pd.DataFrame(data_loader.dataset.std_all, columns=LEVELS,index=INPUT_VARIABLES)    # 检查到很多个要素，在高层数值都很接近0了，我认为dataloader里面，当数据足够小的时候，整个区域的数值都小于10e-6的时候，就应该直接归0.  或者是，小于10e-6的数值，都要直接归0.   还是不清楚这些极小数值，会有什么负面作用？
        latitudes = data_loader.dataset.latitudes
        longitudes = data_loader.dataset.longitudes
        return (latitudes,longitudes,df_mean,df_std)

    @torch.no_grad()
    def get_metrics_df(self,level):
        '''遍历test set，计算 CSI, POD, FAR, ACC, RMSE
        计算某个高度的，4个要素的各个指标
        不应该每一个epoch一个指标吗？     那这样df就不好存放。  
        '''
        model = self.load_generator()   # 为什么这里又不需要self参数输入了？  Python将第一个传入的参数自动绑定为 self，即实例本身
        test_loader = self.get_test_dataloader()
        self.level_actual = level
        # 准备level_idx提取指定高度的数据
        level_idx = int(np.argmin(np.abs(np.array(LEVELS)-level)))   # 

        # 
        lat,_,df_mean,df_std = self.get_meanALL_df()
        lat = torch.as_tensor(lat,dtype=torch.float32, device=self.device)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-12)
        weights = weights.view(1, 1, 1, -1, 1)   # 一个序列，展开为5维度数组。
        #
        key_needs = [i+'_'+str(level) for i in CLOUD_SHORT_NAMES]
        thresholds_raw = {k:THRESHOLDS[k] for k in key_needs}
        thresholds_raw = list(thresholds_raw.values())
        thr = torch.as_tensor(thresholds_raw, dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)  # ? 为了对齐[B, 48, 13, H, W]   本身维度torch.Size([1, 1, 4, 1, 1])
        mean = torch.as_tensor(df_mean.loc[CLOUD_VARIABLES,level].values, dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)  # ? 为了对齐[B, 48, 13, H, W]   本身维度torch.Size([1, 1, 4, 1, 1])
        std = torch.as_tensor(df_std.loc[CLOUD_VARIABLES,level].values,dtype=torch.float32, device=self.device).view(1, 1, 4, 1, 1)  # ? 为了对齐[B, 48, 13, H, W]   本身维度torch.Size([1, 1, 4, 1, 1])

        #
        tp = torch.zeros(self.lead_steps, 4, dtype=torch.float64, device=self.device)
        fp = torch.zeros(self.lead_steps, 4, dtype=torch.float64, device=self.device)
        fn = torch.zeros(self.lead_steps, 4, dtype=torch.float64, device=self.device)
        
        # RMSE 累积量
        rmse_sse = torch.zeros(self.lead_steps, 4, dtype=torch.float64, device=self.device)
        rmse_den = torch.zeros(self.lead_steps, 4, dtype=torch.float64, device=self.device)    # ?

        # ACC 累积量。。   云指数的气候平均，使用训练集3年的数据的平均。
        acc_num = torch.zeros(self.lead_steps, 4, dtype=torch.float64, device=self.device)
        acc_pred_den = torch.zeros(self.lead_steps, 4, dtype=torch.float64, device=self.device)
        acc_true_den = torch.zeros(self.lead_steps, 4, dtype=torch.float64, device=self.device)
        # get clim
        ds = xr.open_zarr(self.zarr_path, consolidated=True, chunks=None)
        clim = ds["data"].isel(   # level_idx : 5； var_idx : [5, 6, 7, 8]
            variable=[5,6,7,8],
            pressure_level=level_idx,   # 
        ).sel(
            time=slice("2018-01-01", "2020-12-31T23:59:59")
        ).mean(dim="time").values.astype(np.float32)   #clim:  (4, 41, 71)
        clim = torch.as_tensor(clim, dtype=torch.float32, device=self.device).view(1, 1, 4, *clim.shape[-2:])
        for batch_id,batch in enumerate(test_loader):   # 
            x,y,_= batch     # 每个batch中的y值，是标准化后的吗？  ✅
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            # 因为这里是loader加载的y值，这是标准化后的。 所以y_true也需要反标准化。
            pred = model(x)  
            B, C, D, H, W = pred.shape    # 模型输出的数据维度： [B, 48, 13, H, W]
            # [B, 48, 13, H, W] -> [B, 12, 4, 13, H, W]
            pred = pred.view(B, self.lead_steps, 4, D, H, W)
            y = y.view(B, self.lead_steps, 4, D, H, W)
            # 只取指定高度层，得到 [B, 12, 4, H, W]
            pred_l = pred[:, :, :, level_idx, :, :]   # 返回B,12,4,H,W维度数组。。   
            y_l = y[:, :, :, level_idx, :, :]
            #
            # 反标准化回原始物理量 kg kg^-1
            q0 = 1e-8
            pred_raw = (10**(pred_l)-1)*q0
            y_raw =  (10**(y_l)-1)*q0
            # pred_raw = pred_l * std + mean
            # y_raw = y_l * std + mean
            # 
            
            # -------------------------
            # CSI / POD / FAR
            # -------------------------
            pred_event = pred_raw >= thr   # 预报发生的时间， 用True/False网格表示
            obs_event = y_raw >= thr   # 真实发生的事件，用True/False网格表示
            tp += (pred_event & obs_event).sum(dim=(0, 3, 4)).double()  # ? .double ? 
            fp += (pred_event & (~obs_event)).sum(dim=(0, 3, 4)).double()
            fn += ((~pred_event) & obs_event).sum(dim=(0, 3, 4)).double()   # fp的个数
            # 最终变成了，(12,4维度的？ )
            # -------------------------
            # RMSE，纬度加权
            # -------------------------
            err2 = (pred_raw - y_raw) ** 2
            rmse_sse += (err2 * weights).sum(dim=(0, 3, 4)).double()   # 纬度加权之后，在经纬度格点上求和，并且每个batch也求和。 最终留下12，4维度的数据。
            batch_den = float(B * W * weights.view(-1).sum().detach().cpu().item())    # ？ 这是在干嘛？ 
            rmse_den += batch_den   # 这个可以理解为N。。   这里算的可能有问题
            # -------------------------
            # ACC，使用训练集气候态距平
            # -------------------------
            pred_anom = pred_raw - clim   # 各个变量什么维度？ 
            true_anom = y_raw - clim

            acc_num += (pred_anom * true_anom * weights).sum(dim=(0, 3, 4)).double()
            acc_pred_den += ((pred_anom ** 2) * weights).sum(dim=(0, 3, 4)).double()
            acc_true_den += ((true_anom ** 2) * weights).sum(dim=(0, 3, 4)).double()


        # -------------------------
        # 汇总指标
        # -------------------------
        csi = tp / (tp + fp + fn + 1e-12)   # 12，4维度的
        pod = tp / (tp + fn + 1e-12)
        far = fp / (tp + fp + 1e-12)    # 一个层在某一个时刻，往往只有一个这样的指数

        rmse = torch.sqrt(rmse_sse / (rmse_den + 1e-12))

        acc = acc_num / (
            torch.sqrt(acc_pred_den) * torch.sqrt(acc_true_den) + 1e-12
        )   # acc的公式
        df = self._make_dataframe(
            csi.detach().cpu().numpy(),
            pod.detach().cpu().numpy(),
            far.detach().cpu().numpy(),
            acc.detach().cpu().numpy(),
            rmse.detach().cpu().numpy(),
        )
        self.csv_path = self.save_dir / f"metrics_{level}_.csv"   #  应该多来几个高度的？   1000，850，500，250
        df.to_csv(self.csv_path, index=False)
        print(f"[INFO] metrics csv saved to: {self.csv_path}")

        return df

    # --------------------------------------------------------
    # 5. 将结果整理成 DataFrame
    # --------------------------------------------------------
    def _make_dataframe(self, csi, pod, far, acc, rmse):
        """
        将 [12, 4] 的指标矩阵整理成长表。    就是一个一列的series
        """
        rows = []

        for lead_idx in range(self.lead_steps):
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
    def plot_metric(self, metric: str, level:Union[float,int]):
        """
        绘制一个指标随 lead time 变化的曲线。
        # 输入metric形式为： "CSI", "POD", "FAR", "ACC", "RMSE"
        每张图：     需要指定level，每一张图只能绘制，某个高度，4种云要素，对某种指标，随lead time的变化。 只能有4条线。 每张图都要指定所绘制的高度
            x 轴：lead hour
            y 轴：metric
            4 条曲线：CIWC, CLWC, CRWC, CSWC
        """
        if (self.save_dir / f"metrics_{level}_.csv").exists():   # 如果该高度的，参数csv文件存在，就用csv文件
            path = self.save_dir / f"metrics_{level}_.csv"
            df = pd.read_csv(path,)   # index_col='lead_step'
        elif not (self.save_dir / f"metrics_{level}_.csv").exists():    # 不存在csv文件就false
            df = self.get_metrics_df(level = level)
        self.level_actual = level
        metric = metric.upper()    # 输入metric形式为： "CSI", "POD", "FAR", "ACC", "RMSE"

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




    def get_one_case_dict(self,anchor_time,split='test',input_steps=2,forecast_steps=12):
        '''读取单个样本，返回这样本，各个方面的值
                case_dict : dict
            {
                "x_model":    [2, 9, 13, H, W]        标准化后的模型输入
                "x_cloud":    [2, 4, 13, H, W]      历史2时刻的4个云变量原始值
                "y_true":     [12, 4, 13, H, W]     未来12时刻的4个云变量原始值
                "times":      dict                  时间信息
            }

        '''
        anchor_time = pd.Timestamp(anchor_time)
        # x_raw = self.data.sel(time = slice(anchor_time-pd.Timedelta(hours=6), anchor_time),  \
        #     variable = INPUT_VARIABLES).values.astype(np.float32)
        # 
        x_raw_basic = self.data.sel(time = slice(anchor_time-pd.Timedelta(hours=6), anchor_time),  \
            variable = INPUT_VARIABLES[:5]).values.astype(np.float32)
        x_raw_cloud = self.data.sel(time = slice(anchor_time-pd.Timedelta(hours=6), anchor_time),  \
            variable = CLOUD_VARIABLES).values.astype(np.float32)

        y_true_raw = self.data.sel(time=slice(anchor_time+pd.Timedelta(hours=6),anchor_time+pd.Timedelta(hours=6)*self.lead_steps),
            variable = CLOUD_VARIABLES).values.astype(np.float32)
        x_cloud_raw = self.data.sel(time=slice(anchor_time-pd.Timedelta(hours=6), anchor_time),
            variable = CLOUD_VARIABLES).values.astype(np.float32)

        _,_,df_mean,df_std = self.get_meanALL_df()

        # 
        x_mean = df_mean.loc[INPUT_VARIABLES[:5],LEVELS].values
        x_mean = x_mean[None,:,:,None,None]  # [1, 9, 13, 1, 1]  广播
        x_std = df_std.loc[INPUT_VARIABLES[:5],LEVELS].values
        x_std = x_std[None,:,:,None,None]   

        # x_norm = (x_raw-x_mean) / x_std   # 输入的x的数据标准化
        # 采用最新的标准化方法
        x_norm_basic = (x_raw_basic-x_mean)/x_std
        q0 = 1e-8
        x_norm_cloud = np.log10(1+x_raw_cloud/q0)
        x_norm = np.concatenate((x_norm_basic, x_norm_cloud), axis=1)

        # 将 [2, 9, 13, H, W] -> [18, 13, H, W]
        x_model = x_norm.reshape(
            x_norm.shape[0] * x_norm.shape[1],  # 2 * 9 = 18
            x_norm.shape[2],
            x_norm.shape[3],
            x_norm.shape[4]
        ).astype(np.float32)
        time_info = {"anchor_time": anchor_time}
        return {
            "x_model": x_model,   # 输入模型的x的数据。
            "x_cloud": x_cloud_raw,   # 输入模型的数据。正则化之后的，
            "y_true": y_true_raw,   #
            "times": time_info}  # 只包含anchor time

    @torch.no_grad()    # 不知道有没有用？ 
    def forecast_one_case(self,anchor_time):
        '''对单个样本做前向推理，并将输出反标准化为原始物理量
        返回
        ----
        y_pred_raw : np.ndarray
            [12, 4, 13, H, W]，未来12时刻4个云变量的预测原始值
        '''
        # 需要根据anchor_time提取单个数据样本，需要对输入的前两个时刻的要素值进行标准化，输入到模型中，得到标准的预测值，在对其反标准化，得到最终的预测值。
        model = self.load_generator()
        data_loader = self.get_test_dataloader()
        latitudes,_,df_mean,df_std = self.get_meanALL_df()

        case_dict = self.get_one_case_dict(anchor_time=anchor_time)
        x_model = case_dict['x_model']
        x_tensor = torch.from_numpy(x_model).unsqueeze(0).to(self.device)    # unsqueeze ? 
        # 模型预测
        y_pred_norm = model(x_tensor).cpu().numpy()[0] 
        # [48, 13, H, W] -> [12, 4, 13, H, W]
        y_pred_norm = y_pred_norm.reshape(self.lead_steps, 4, y_pred_norm.shape[1], y_pred_norm.shape[2], y_pred_norm.shape[3])
        # 反标准化
        # y_mean = df_mean.loc[CLOUD_VARIABLES].values
        # y_mean = y_mean[None,:,:,None,None]
        # y_std = df_std.loc[CLOUD_VARIABLES].values
        # y_std = y_std[None,:,:,None,None]
        # y_pred_raw = y_pred_norm*y_std+y_mean  # 
        #更新标准化方案   z = log10(1+q/q0)
        q0 = 1e-8
        y_pred_raw = (10**(y_pred_norm) -1)*q0
        # 云变量物理上应为非负，作一个下限裁剪   # 
        y_pred_raw = np.clip(y_pred_raw, 0.0, None)
        return y_pred_raw.astype(np.float32)


    def add_china_map_features(self,ax, extent=[70, 140, 15, 55], show_left_labels=False, show_bottom_labels=False):
        """
        在给定 cartopy 轴上添加中国区域底图要素。
        """
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.95")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        ax.add_feature(cfeature.LAKES.with_scale("50m"), edgecolor="gray", facecolor="none", linewidth=0.3)
        ax.add_feature(cfeature.RIVERS.with_scale("50m"), edgecolor="gray", linewidth=0.3)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.3,
            color="gray",
            alpha=0.5,
            linestyle="--"
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = show_left_labels
        gl.bottom_labels = show_bottom_labels

        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}


    def visualize_one_case(self,anchor_time,plot_level=500,cmap="YlGnBu",lead_step =1):   # 必选参数（无默认）必须写在前面. 有默认值的参数，必须写在后面
        '''  类中的函数，必须要以self作为第一个参数      给定anchor_time可以绘制出，这个样本的，模型在给定高度的，预报效果
        plot_level = 500表示绘制500hPa的数据。
            lead_step : int
        未来第几个时刻，范围 1~12

        NOTE   这个类函数，需要保证在类中也可以随意调用。
        '''
        # 先处理数据
        case_dict = self.get_one_case_dict(anchor_time)
        time_info = case_dict['times']['anchor_time']   # 只包含一个anchor time
        anchor_time = pd.Timestamp(time_info) 
        level_idx = np.argmin(np.abs(np.array(LEVELS)-plot_level))    # 5 ✅    
        y_pred_raw = self.forecast_one_case(anchor_time)

        x_cloud_raw = case_dict['x_cloud'][:,:,level_idx,:,:]
        y_true_raw = case_dict['y_true'][:,:,level_idx,:,:]  # 具体什么维度的？
        y_pred_raw = y_pred_raw[:,:,level_idx,:,:]

        lats,lons,_,_ = self.get_meanALL_df()
        # 在画图
        fig, axes = plt.subplots(
            nrows=4,
            ncols=4,
            figsize=(22, 14),
            subplot_kw={"projection": ccrs.PlateCarree()})
        
        input_time_1 = anchor_time-pd.Timedelta(hours=6)
        input_time_2 = anchor_time
        target_time = anchor_time +pd.Timedelta(hours=6*lead_step)
        column_titles = [
            f"Input 1\n{input_time_1:%Y-%m-%d %H:%M}\n(True)",    # 了解这里的用法。  格式化字符串，%Y-%m-%d %H:%M 是日期时间格式。
            f"Input 2\n{input_time_2:%Y-%m-%d %H:%M}\n(True)",
            f"Lead +{lead_step*6} h\n{target_time:%Y-%m-%d %H:%M}\n(True)",
            f"Lead +{lead_step*6} h\n{target_time:%Y-%m-%d %H:%M}\n(Pred)"    #    ✅
            ]
        for j in range(4):
            axes[0, j].set_title(column_titles[j], fontsize=11, fontweight="bold")
        # 每一行分别绘图
        t_idx = lead_step-1   # 
        for row in range(4):
            # 本行画四幅图
            # print(f"y_pred_raw.shape:{y_pred_raw.shape}")
            row_fields = [
                x_cloud_raw[0, row],     # 输入时刻1
                x_cloud_raw[1, row],     # 输入时刻2
                y_true_raw[t_idx, row],  # 未来时刻真值   # 本来[12, 4, H, W] 。 
                y_pred_raw[t_idx, row],  # 未来时刻预测值   # y_pred_shape竟然是y_pred_raw.shape:(4, 13, 41, 71)？？？！！
                ]
            csi = csi_score(torch.tensor(y_pred_raw[t_idx, row]),torch.tensor(y_true_raw[t_idx, row]),THRESHOLDS[CLOUD_SHORT_NAMES[row]+f"_{plot_level}"])
            # print(row_fields[0].shape,row_fields[1].shape,row_fields[2].shape,row_fields[3].shape)
            # 为了让一行内颜色可比较，统一本行色标范围
            stacked = np.stack(row_fields, axis=0)  # 将 row_fields 沿着 axis=0 堆叠起来。形成一维数组。
            positive_vals = stacked[np.isfinite(stacked)]   # np.isfinite(stacked) 返回一个布尔数组，表示 stacked 中哪些元素是有限的（不是 NaN 或 Inf）。

            # 云变量通常是非负的，因此令 vmin=0
            vmin = 0.0
            if positive_vals.size > 0:   # positive_vals.size > 0 表示 positive_vals 不为空。 ??也就是有True值的元素。
                vmax = np.nanpercentile(positive_vals, 99.0)  # 计算 positive_vals 的 99% 分位数。？？    取99%分位数的数值为最大值！！？？
            else:   # 一般positive_vals为空，则vmax=1.0..  vmax是色标范围的最大值。
                vmax = 1.0   # 

            # 防止 vmax 太小或为 0
            if (not np.isfinite(vmax)) or (vmax <= 0):
                vmax = 1.0
            # 使用 PowerNorm，让弱信号更容易看见，且支持 0 值
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)   # PowerNorm 是 matplotlib.colors 模块中的一个类，用于创建幂次归一化（power normalization）对象。
            mappable = None
            for col in range(4):
                ax = axes[row,col]
                self.add_china_map_features(ax,
                    show_left_labels=(col == 0),
                    show_bottom_labels=(row == 3))
                field2d = row_fields[col]
                # 
                mappable = ax.pcolormesh(
                    lons,
                    lats,
                    field2d,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    norm=norm,
                    shading="auto")
                if col == 0:
                    ax.set_ylabel(     #  这里没画出来
                        CLOUD_SHORT_NAMES[row],
                        fontsize=12,
                        fontweight="bold",
                        rotation=90,
                        labelpad=12)
                elif col == 3:
                    ax.text(
                    0.95, 0.05,                  # X=0.95, Y=0.05，即右下角位置
                    f"CSI: {csi:.4f}\nThresh: {THRESHOLDS[CLOUD_SHORT_NAMES[row]+'_'+str(plot_level)]}", 
                    transform=ax.transAxes,      # 使用相对坐标系 (0~1)
                    ha='right',                  # 水平右对齐
                    va='bottom',                 # 垂直底部对齐
                    color='k',               # 字体颜色
                    fontsize=8, 
                    fontweight='bold',
                    # 添加一个半透明的黑色背景框，防止气象图底色干扰文字阅读
                    bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.3')
                )
            cbar = fig.colorbar(
                mappable,
                ax=axes[row, -1],    # 把 ： 改成-1了。colorbar就不重叠了
                orientation="vertical",
                fraction=0.02,
                pad=0.02)
            cbar.ax.tick_params(labelsize=9)
            cbar.set_label(f"{CLOUD_SHORT_NAMES[row]}", fontsize=10)
        #
        fig.suptitle(
            f"Forecast visualization at {plot_level} hPa\n"
            f"Anchor time: {pd.Timestamp(anchor_time):%Y-%m-%d %H:%M} | "
            f"Lead step: {lead_step} ({lead_step*6} h)",
            fontsize=14,
            fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        anchor_str = pd.Timestamp(anchor_time).strftime("%Y%m%d_%H%M")
        save_name = f"{self.split}_anchor_{anchor_str}_lead{lead_step:02d}_{int(plot_level)}hPa_new2.png"
        save_path = self.save_dir / save_name
        self.save_dir.mkdir(parents=True, exist_ok=True)   # parents=True   父目录不在，可以自己创建。
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] figure saved to: {save_path}")

 


























