import torch
import numpy as np
from typing import Iterable, Optional,Union   # 这两个库，真要认真看看。



@torch.no_grad()   # 是 PyTorch 中用于临时禁用梯度计算的上下文管理器或装饰器，常用于推理阶段以减少内存占用并加快计算速度。 所有张量运算的结果都会自动设置 requires_grad=False，即使输入张量原本需要梯度   #  针对的是，无需反向传播的任务
def latitude_weighted_rmse(pred: torch.Tensor, target: torch.Tensor, latitudes: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    lat = torch.as_tensor(latitudes, dtype=pred.dtype, device=pred.device)
    weights = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    weights = weights / weights.mean().clamp_min(1e-12)
    weights = weights.view(1, 1, 1, -1, 1)   # 
    mse = ((pred - target) ** 2 * weights).mean()
    return torch.sqrt(mse)


@torch.no_grad()
def csi_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Critical Success Index in the current tensor space.   # CSI必须要自定义一个threshold，自定义一个有云的阈值。
       csi定义：    hits/(hits+misses+F.alrams)   就是模型输出的有的全部区域中，正确的比例是多少？ 
    If pred/target are normalized, threshold should also be in normalized space.
    For physical CSI, denormalize first and use a physical threshold.
    """
    pred_event = pred > threshold   # 这个到底返回什么呢？返回一堆false/true吗？
    '''
>>> aa = torch.tensor([1,2,3])
>>> aa
tensor([1, 2, 3])
>>> bb = torch.tensor([2,2,2])
>>> aa>bb
tensor([False, False,  True])
    '''
    target_event = target > threshold  
    hits = (pred_event & target_event).sum().float()  #tensor(1.)  这种单个数字的tensor
    misses = (~pred_event & target_event).sum().float()  # 就是misses的格点数。
    false_alarms = (pred_event & ~target_event).sum().float()  # 就是F.alarms的点数
    denom = hits + misses + false_alarms   # tensor(2.) 这种浮点整数的tensor。
    return hits / denom.clamp_min(1.0)  # 这里本来就是一个整数，并不需要这个clamp   #.clamp_min function in PyTorch is used to set a minimum threshold for tensor elements. Any value in the tensor below the specified minimum is replaced with the minimum value, while other values remain unchanged.
    # 所以denom总是会是大于1的，。   这里的clamp_min可有可无





