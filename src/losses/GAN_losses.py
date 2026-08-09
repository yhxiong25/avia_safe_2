import torch.nn as nn
import torch
import numpy as np
from typing import Iterable, Optional,Union,Sequence   # 这两个库，真要认真看看。
import logging
from src.data.fixed_info import THRESHOLDS,THRESHOLDS_normalized


'''
NOTE：疑虑困惑
1，registerbuffer？
2，顺序都是按照 CIWC,CLWC,CRWC,CSWC来的？
3，

'''




class LatitudeWeightedCharbonnierLoss(nn.Module):
    """Latitude-weighted Charbonnier L1 loss.     就是加了纬度权重的RMSE

    Expected tensor shape: [B, C, D, H, W], where H is latitude.
    """

    def __init__(self, latitudes: Union[np.ndarray , torch.Tensor], eps: float = 1e-3) -> None:
        # 这里必须要输入一个latitudes数组
        super().__init__()
        lat = torch.as_tensor(latitudes, dtype=torch.float32)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-12)
        self.register_buffer("weights", weights.view(1, 1, 1, -1, 1))   # register_buffer ?? 干嘛的？
        #这里还变换了weights的维度顺序   #避免被优化器更新参数。   没有这个会怎样？
        #Register_buffer 注册的张量会自动随模型一起移动到指定设备
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)  # torch.sqrt返回一个tensor
        return (loss * self.weights).mean()



class LatitudeWeightedSoftMRLoss_500(nn.Module):
    """
    Latitude-weighted differentiable Missing Rate loss.

    需要明确为什么，经过logistic换算后，才能作为损失函数被求导！！

    作用：
        将离散漏报率 MR = FN / (TP + FN)
        转换为可导版本 MR_diff，用于神经网络训练。

    Expected tensor shape:
        pred, target: [B, C, D, H, W]

    其中：
        C = forecast_steps * num_cloud_vars = 12 * 4 = 48
        D = pressure_level = 13
        H = latitude = 41
        W = longitude = 71

    当前默认只计算 500 hPa 层。
    """

    def __init__(
        self,
        latitudes: Union[np.ndarray, torch.Tensor],
        alpha: Union[float, Sequence[float], torch.Tensor],   # ？  alpha这里指变量的阈值。
        pressure_levels: Sequence[float],
        level_hpa: float = 500.0,
        forecast_steps: int = 12,
        num_cloud_vars: int = 4,    
        k: float = 2.0,   #？    k要从5设置为2？？   这个先不管
        eps: float = 1e-8,   
    ) -> None:
        super().__init__()

        # -----------------------------
        # 1. 纬度权重
        # -----------------------------
        lat = torch.as_tensor(latitudes, dtype=torch.float32)

        # cos(lat) 面积权重；低纬格点面积更大，高纬格点面积更小
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)

        # 归一化到平均值为 1，避免 loss 数值尺度变化太大
        weights = weights / weights.mean().clamp_min(1e-12)

        # pred_500 的形状是 [B, T, V, H, W]
        # 所以权重 reshape 成 [1, 1, 1, H, 1]，方便自动广播
        self.register_buffer("weights", weights.view(1, 1, 1, -1, 1))

        # -----------------------------
        # 2. 阈值 alpha
        # -----------------------------
        # alpha 可以是：
        #   一个标量：4 个云变量共用同一个阈值
        #   长度为 4 的列表：每个云变量一个阈值
        alpha = torch.as_tensor(alpha, dtype=torch.float32)

        if alpha.ndim == 0:
            alpha = alpha.repeat(num_cloud_vars)   #?

        if alpha.numel() != num_cloud_vars:    # .numel()返回元素个数   ?   
            raise ValueError(
                f"alpha 应该是标量或长度为 {num_cloud_vars} 的序列，"
                f"但当前 alpha.numel() = {alpha.numel()}"
            )

        # alpha shape: [4] -> [1, 1, 4, 1, 1]
        # 对应 [B, T, V, H, W] 中的 V 维
        self.register_buffer("alpha", alpha.view(1, 1, num_cloud_vars, 1, 1))   # register_buffer ?? 干嘛的？  把alpha注册为模块的缓冲区，方便在forward中使用。   
        # register_buffer 是 PyTorch 中用于在模型中注册 不需要梯度更新 的张量的方法。这些张量会被包含在模型的 state_dict 中，方便保存和加载，同时会随着模型移动到 GPU 或 CPU，但不会被优化器更新。
        # -----------------------------
        # 3. 找到目标高度层索引
        # -----------------------------
        pressure_levels = np.asarray(pressure_levels, dtype=np.float32)
        self.level_idx = int(np.argmin(np.abs(pressure_levels - level_hpa)))   # np.argmin返回最小值的索引
        self.level_hpa_actual = float(pressure_levels[self.level_idx])

        self.forecast_steps = forecast_steps
        self.num_cloud_vars = num_cloud_vars
        self.k = k
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        返回：
            标量 MR_diff loss，越大表示漏报越严重。
        """
        B, C, D, H, W = pred.shape   # [B, 48, 13, H, W]    D是高度。
        expected_C = self.forecast_steps * self.num_cloud_vars   # 12*4

        if C != expected_C:
            raise ValueError(f"Expected C={expected_C}, but got C={C}")

        # [B, 48, 13, H, W] -> [B, 12, 4, 13, H, W]
        pred = pred.view(B, self.forecast_steps, self.num_cloud_vars, D, H, W)
        target = target.view(B, self.forecast_steps, self.num_cloud_vars, D, H, W)

        # 只取 500 hPa 或指定 level_hpa 对应的层
        pred_l = pred[:, :, :, self.level_idx, :, :]      # [B, 12, 4, H, W]
        target_l = target[:, :, :, self.level_idx, :, :]  # [B, 12, 4, H, W]

        # -----------------------------
        # 观测事件 mask:
        # target > alpha 为真实发生事件
        # 这是 0/1 mask，不需要对 target 求导
        # -----------------------------
        obs_event = (target_l > self.alpha).float()

        # -----------------------------
        # 预报未超过阈值的 soft probability:
        # sigmoid(k * (alpha - pred))
        #
        # 如果 pred << alpha，则接近 1，表示强漏报；
        # 如果 pred >> alpha，则接近 0，表示没有漏报。
        # -----------------------------
        soft_miss = torch.sigmoid(self.k * (self.alpha - pred_l))

        # -----------------------------
        # 预报超过阈值的 soft probability:
        # sigmoid(k * (pred - alpha))
        #
        # 用于计算 soft TP。
        # -----------------------------
        soft_hit = torch.sigmoid(self.k * (pred_l - self.alpha))

        # soft TP 和 soft FN
        tp_field = obs_event * soft_hit
        fn_field = obs_event * soft_miss

        # 纬度加权求和
        tp = (tp_field * self.weights).sum()
        fn = (fn_field * self.weights).sum()

        mr = fn / (tp + fn + self.eps)

        return mr




class LatitudeWeightedSoftFARLoss_500(nn.Module):
    """
    Latitude-weighted differentiable False Alarm Rate loss.

    作用：
        将离散空报率 FAR = FP / (TP + FP)
        转换为可导版本 FAR_diff，用于神经网络训练。

    Expected tensor shape:
        pred, target: [B, C, D, H, W]

    当前默认只计算 500 hPa 层。
    """

    def __init__(
        self,
        latitudes: Union[np.ndarray, torch.Tensor],
        alpha: Union[float, Sequence[float], torch.Tensor],
        pressure_levels: Sequence[float],
        level_hpa: float = 500.0,
        forecast_steps: int = 12,
        num_cloud_vars: int = 4,
        k: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        # -----------------------------
        # 1. 纬度权重
        # -----------------------------
        lat = torch.as_tensor(latitudes, dtype=torch.float32)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-12)

        # pred_500 shape: [B, T, V, H, W]
        self.register_buffer("weights", weights.view(1, 1, 1, -1, 1))   #?

        # -----------------------------
        # 2. 阈值 alpha
        # -----------------------------
        alpha = torch.as_tensor(alpha, dtype=torch.float32)

        if alpha.ndim == 0:
            alpha = alpha.repeat(num_cloud_vars)

        if alpha.numel() != num_cloud_vars:
            raise ValueError(
                f"alpha 应该是标量或长度为 {num_cloud_vars} 的序列，"
                f"但当前 alpha.numel() = {alpha.numel()}"
            )

        self.register_buffer("alpha", alpha.view(1, 1, num_cloud_vars, 1, 1))

        # -----------------------------
        # 3. 找高度层索引
        # -----------------------------
        pressure_levels = np.asarray(pressure_levels, dtype=np.float32)
        self.level_idx = int(np.argmin(np.abs(pressure_levels - level_hpa)))
        self.level_hpa_actual = float(pressure_levels[self.level_idx])

        self.forecast_steps = forecast_steps
        self.num_cloud_vars = num_cloud_vars
        self.k = k
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        返回：
            标量 FAR_diff loss，越大表示空报越严重。
        """
        B, C, D, H, W = pred.shape
        expected_C = self.forecast_steps * self.num_cloud_vars

        if C != expected_C:
            raise ValueError(f"Expected C={expected_C}, but got C={C}")

        # [B, 48, 13, H, W] -> [B, 12, 4, 13, H, W]
        pred = pred.view(B, self.forecast_steps, self.num_cloud_vars, D, H, W)
        target = target.view(B, self.forecast_steps, self.num_cloud_vars, D, H, W)

        # 只取指定高度层，例如 500 hPa
        pred_l = pred[:, :, :, self.level_idx, :, :]      # [B, 12, 4, H, W]
        target_l = target[:, :, :, self.level_idx, :, :]  # [B, 12, 4, H, W]

        # -----------------------------
        # 观测非事件 mask:
        # target < alpha 表示真实没有发生事件
        # -----------------------------
        obs_nonevent = (target_l < self.alpha).float()

        # 观测事件 mask，用于 soft TP
        obs_event = (target_l > self.alpha).float()

        # 预报超过阈值的 soft probability
        soft_forecast_event = torch.sigmoid(self.k * (pred_l - self.alpha))

        # soft TP 和 soft FP
        tp_field = obs_event * soft_forecast_event
        fp_field = obs_nonevent * soft_forecast_event

        # 纬度加权求和
        tp = (tp_field * self.weights).sum()
        fp = (fp_field * self.weights).sum()

        far = fp / (tp + fp + self.eps)

        return far











class CompositeGeneratorLoss(nn.Module):
    def __init__(self, recon_fn, mr_fn, far_fn, lambda_mr=1.0, lambda_far=1.0):
        super().__init__()
        # 将子损失函数作为模块的属性，PyTorch会自动管理它们所在的设备(Device)
        self.recon_fn = recon_fn
        self.mr_fn = mr_fn
        self.far_fn = far_fn
        self.lambda_mr = lambda_mr
        self.lambda_far = lambda_far
    def compute_components(self, preds, targets):
        """
        计算总损失和各个子损失。

        参数
        ----
        preds : torch.Tensor
            模型预测值。
        targets : torch.Tensor
            真实值。
        返回
        ----
        loss_dict : dict
            返回包含总损失和子损失的字典。

            注意：
            这里返回的 loss tensor 都保留计算图，
            可以直接用于 backward。
        """
        loss_recon = self.recon_fn(preds, targets)
        loss_mr = self.mr_fn(preds, targets)
        loss_far = self.far_fn(preds, targets)
        loss_total = (
            loss_recon
            + self.lambda_mr * loss_mr
            + self.lambda_far * loss_far
        )
        return {
            "total": loss_total,
            "recon": loss_recon,
            "mr": loss_mr,
            "far": loss_far,
            "weighted_mr": self.lambda_mr * loss_mr,
            "weighted_far": self.lambda_far * loss_far,
        }
    def forward(self, preds, targets):
        # 只有在真正传入数据时，才进行前向计算得到 Loss Tensor
        loss_recon = self.recon_fn(preds, targets)
        loss_mr = self.mr_fn(preds, targets)
        loss_far = self.far_fn(preds, targets)
        # logging.info(f"loss_recon:{loss_recon};loss_mr:{loss_mr};loss_far:{loss_far}")
        # 将张量进行加权求和并返回
        return loss_recon + self.lambda_mr * loss_mr + self.lambda_far * loss_far






# 第二个损失函数，直接创建一个类包含上述所有类型的loss计算
class CompositeGeneratorLoss_v2(nn.Module):
    """
    Composite Generator Loss v2.
    功能：
        1. 计算全场纬度加权 Charbonnier 重构损失；
        2. 在多个指定高度层上计算 soft MR；
        3. 在多个指定高度层上计算 soft FAR；
        4. 对多个高度层的 MR/FAR 求平均；
        5. 最终返回 total loss。
    输入张量形状：
        preds, targets: [B, C, D, H, W]
    其中：
        C = forecast_steps * num_cloud_vars = 12 * 4 = 48
        D = pressure levels = 13
        H = latitude
        W = longitude
    注意：
        thresholds_normalized 必须是标准化后的阈值字典。
        因为 DataLoader 已经把 y 标准化了，所以 loss 中的 alpha 也必须是标准化空间。
    """
    def __init__(
        self,
        latitudes: Union[np.ndarray, torch.Tensor],
        pressure_levels: Sequence[float],
        thresholds_normalized: dict,
        levels_hpa: Sequence[float] = (1000, 925, 850, 700, 600, 500, 400, 300, 250),
        cloud_short_names: Sequence[str] = ("CIWC", "CLWC", "CRWC", "CSWC"),
        forecast_steps: int = 12,
        num_cloud_vars: int = 4,
        lambda_mr: float = 1.0,
        lambda_far: float = 1.0,
        k: float = 2.0,
        eps: float = 1e-8,
        recon_eps: float = 1e-3,
        average_mode: str = "var_level",
    ) -> None:
        """
        Parameters
        ----------
        latitudes:
            纬度数组，用于 cos(lat) 面积权重。

        pressure_levels:
            zarr / dataset 中的 pressure_level 坐标数组。

        thresholds_normalized:
            标准化后的阈值字典。
            key 格式必须类似：
                "CIWC_500", "CLWC_500", "CRWC_500", "CSWC_500"

        levels_hpa:
            需要纳入 MR/FAR 损失的高度层。
            默认是 250 hPa 及以下的 9 层：
                1000, 925, 850, 700, 600, 500, 400, 300, 250

        cloud_short_names:
            云变量顺序，必须和模型输出 reshape 后的变量维一致。
            默认：
                [CIWC, CLWC, CRWC, CSWC]

        k:
            sigmoid 陡峭度。
            k 越大，越接近硬阈值；k 越小，梯度更平滑。

        average_mode:
            "level":
                先对每个高度层计算一个 MR/FAR，再对高度层平均。
                这更符合“9 个高度的损失函数值求出后再平均”。

            "var_level":
                对每个变量、每个高度层分别计算 MR/FAR，再对所有有效变量-高度组合平均。
        """
        super().__init__()

        if average_mode not in ("level", "var_level"):
            raise ValueError("average_mode must be 'level' or 'var_level'.")

        self.forecast_steps = forecast_steps
        self.num_cloud_vars = num_cloud_vars
        self.lambda_mr = lambda_mr
        self.lambda_far = lambda_far
        self.k = k
        self.eps = eps
        self.average_mode = average_mode     # 

        # ----------------------------------------------------
        # 1. 重构损失：全高度、全变量、全 lead time 的纬度加权 Charbonnier
        # ----------------------------------------------------
        self.recon_fn = LatitudeWeightedCharbonnierLoss(   # 这里似乎直接引用外部calss了
            latitudes=latitudes,
            eps=recon_eps,
        )

        # ----------------------------------------------------
        # 2. 纬度权重
        #    后续 MR/FAR 张量形状为 [B, T, V, L, H, W]
        #    所以权重形状设为 [1, 1, 1, 1, H, 1]
        # ----------------------------------------------------
        lat = torch.as_tensor(latitudes, dtype=torch.float32)
        weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
        weights = weights / weights.mean().clamp_min(1e-12)
        self.register_buffer("weights", weights.view(1, 1, 1, 1, -1, 1))   # tensor的广播机制不熟悉
        # 通过 .view(1, 1, 1, 1, -1, 1)，它被塑造成了一个 $1 \times 1 \times 1 \times 1 \times N \times 1$ 的 6 维张量。
        # ----------------------------------------------------
        # 3. 找到指定高度层在 D 维中的索引
        # ----------------------------------------------------
        pressure_levels_np = np.asarray(pressure_levels, dtype=np.float32)
        level_indices = []
        level_actuals = []

        for lev in levels_hpa:
            idx = int(np.argmin(np.abs(pressure_levels_np - float(lev))))
            level_indices.append(idx)
            level_actuals.append(float(pressure_levels_np[idx]))

        self.level_hpas_requested = [float(v) for v in levels_hpa]
        self.level_hpas_actual = level_actuals
        self.num_levels = len(level_indices)

        self.register_buffer(
            "level_indices",
            torch.as_tensor(level_indices, dtype=torch.long),  # TypeError: as_tensor(): argument 'dtype' must be torch.dtype, not builtin_function_or_method
            persistent=False,     # ? 
        )
        # ----------------------------------------------------
        # 4. 构建 alpha 阈值矩阵
        #    alpha_matrix shape: [V, L]
        #    valid_matrix shape: [V, L]
        #
        #    顺序固定为：
        #       V = [CIWC, CLWC, CRWC, CSWC]
        #       L = levels_hpa
        # ----------------------------------------------------
        alpha_matrix = np.zeros((num_cloud_vars, self.num_levels), dtype=np.float32)
        valid_matrix = np.zeros((num_cloud_vars, self.num_levels), dtype=np.float32)
        missing_keys = []
        for v_idx, var_name in enumerate(cloud_short_names):
            for l_idx, lev in enumerate(levels_hpa):   # 各个云变量各个高度循环，得到阈值的矩阵
                key = f"{var_name}_{int(lev)}"
                value = thresholds_normalized.get(key, None)
                # None 表示这个变量-高度组合不参与 MR/FAR 计算
                if value is None:
                    alpha_matrix[v_idx, l_idx] = 0.0
                    valid_matrix[v_idx, l_idx] = 0.0
                    missing_keys.append(key)
                else:
                    alpha_matrix[v_idx, l_idx] = float(value)
                    valid_matrix[v_idx, l_idx] = 1.0   #？

        if valid_matrix.sum() == 0:   # valid_matrix的作用：检验。
            raise ValueError("没有任何有效阈值，请检查 thresholds_normalized 和 levels_hpa。")

        # alpha: [1, 1, V, L, 1, 1]
        self.register_buffer(
            "alpha",
            torch.as_tensor(alpha_matrix, dtype=torch.float32).view(
                1, 1, num_cloud_vars, self.num_levels, 1, 1
            ),
        )
        # valid_mask: [1, 1, V, L, 1, 1]
        self.register_buffer(
            "valid_mask",
            torch.as_tensor(valid_matrix, dtype=torch.float32).view(
                1, 1, num_cloud_vars, self.num_levels, 1, 1
            ),
        )
        # 用于按 level 平均时判断哪些高度层至少有一个有效变量
        valid_level = valid_matrix.sum(axis=0) > 0   #
        self.register_buffer(
            "valid_level",
            torch.as_tensor(valid_level, dtype=torch.bool),
            persistent=False,
        )

        # 用于记录和检查
        self.cloud_short_names = list(cloud_short_names)
        self.missing_keys = missing_keys

    def _reshape_and_select_levels(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        将 [B, 48, 13, H, W] reshape 为 [B, 12, 4, 13, H, W]，
        再选出多个高度层，得到 [B, 12, 4, L, H, W]。
        """
        B, C, D, H, W = preds.shape
        expected_C = self.forecast_steps * self.num_cloud_vars

        if C != expected_C:
            raise ValueError(f"Expected C={expected_C}, but got C={C}")

        preds = preds.view(B, self.forecast_steps, self.num_cloud_vars, D, H, W)  #转换维度
        targets = targets.view(B, self.forecast_steps, self.num_cloud_vars, D, H, W)

        # 选出多个高度层
        preds_l = torch.index_select(preds, dim=3, index=self.level_indices)
        targets_l = torch.index_select(targets, dim=3, index=self.level_indices)

        # shape: [B, T, V, L, H, W]
        return preds_l, targets_l
    def _compute_soft_mr_far(self, preds: torch.Tensor, targets: torch.Tensor):
        """ 这个应该要拆开，函数前不应该有_ 
        计算多个高度层上的 soft MR 和 soft FAR。

        返回：
            loss_mr_mean, loss_far_mean
        """
        pred_l, target_l = self._reshape_and_select_levels(preds, targets)

        # ----------------------------------------------------
        # 1. 观测事件 / 非事件 mask
        # ----------------------------------------------------
        obs_event = (target_l > self.alpha).float() * self.valid_mask   # valid_mask : torch.Size([1, 1, 4, 9, 1, 1])
        obs_nonevent = (target_l < self.alpha).float() * self.valid_mask

        # ----------------------------------------------------
        # 2. soft forecast event / miss
        # ----------------------------------------------------
        soft_hit = torch.sigmoid(self.k * (pred_l - self.alpha)) * self.valid_mask
        soft_miss = torch.sigmoid(self.k * (self.alpha - pred_l)) * self.valid_mask

        # ----------------------------------------------------
        # 3. soft TP / FN / FP fields
        # ----------------------------------------------------
        tp_field = obs_event * soft_hit
        fn_field = obs_event * soft_miss
        fp_field = obs_nonevent * soft_hit

        # ----------------------------------------------------
        # 4. 汇总方式一：每个高度层一个 MR/FAR，然后对高度层平均
        # ----------------------------------------------------
        if self.average_mode == "level":
            # 对 B,T,V,H,W 求和，只保留 L
            # 输出 shape: [L]
            sum_dims = (0, 1, 2, 4, 5)

            tp_l = (tp_field * self.weights).sum(dim=sum_dims)
            fn_l = (fn_field * self.weights).sum(dim=sum_dims)
            fp_l = (fp_field * self.weights).sum(dim=sum_dims)

            mr_l = fn_l / (tp_l + fn_l + self.eps)
            far_l = fp_l / (tp_l + fp_l + self.eps)

            # 如果某一层没有任何有效变量，则跳过
            valid_l = self.valid_level
            loss_mr = mr_l[valid_l].mean()
            loss_far = far_l[valid_l].mean()
            return loss_mr, loss_far
        # ----------------------------------------------------
        # 5. 汇总方式二：每个变量-高度层一个 MR/FAR，然后平均
        # ----------------------------------------------------
        if self.average_mode == "var_level":   # 需要用这个
            # 对 B,T,H,W 求和，保留 V,L    # 原来的数据有6个维度： B,T,4,9,41,71  # 4个云变量，9个高度， 
            # 输出 shape: [V, L]
            sum_dims = (0, 1, 4, 5)
            tp_vl = (tp_field * self.weights).sum(dim=sum_dims)
            fn_vl = (fn_field * self.weights).sum(dim=sum_dims)
            fp_vl = (fp_field * self.weights).sum(dim=sum_dims)
            mr_vl = fn_vl / (tp_vl + fn_vl + self.eps)
            far_vl = fp_vl / (tp_vl + fp_vl + self.eps)
            valid_vl = self.valid_mask.view(self.num_cloud_vars, self.num_levels).bool()
            loss_mr = mr_vl[valid_vl].mean()
            loss_far = far_vl[valid_vl].mean()
            return loss_mr, loss_far
    def compute_components(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        计算 total loss 以及各个子损失项。

        返回的字典可以直接用于训练日志记录：
            total
            recon
            mr
            far
            weighted_recon
            weighted_mr
            weighted_far
        """
        loss_recon = self.recon_fn(preds, targets)
        loss_mr, loss_far = self._compute_soft_mr_far(preds, targets)
        weighted_recon = loss_recon
        weighted_mr = self.lambda_mr * loss_mr
        weighted_far = self.lambda_far * loss_far
        loss_total = weighted_recon + weighted_mr + weighted_far
        return {
            "total": loss_total,
            "recon": loss_recon,
            "mr": loss_mr,
            "far": loss_far,
            "weighted_recon": weighted_recon,
            "weighted_mr": weighted_mr,
            "weighted_far": weighted_far,
        }
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        使 criterion(preds, targets) 直接返回 total loss。
        """
        return self.compute_components(preds, targets)["total"]







