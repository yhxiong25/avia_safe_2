'''
专门为mask_predictor设计，处理所有气象数据，并生成[B, 128, 11, 18]特征图
'''


import torch
import torch.nn as nn


def get_group_num(channels: int, max_groups: int = 8) -> int:
    """
    为 GroupNorm 自动选择合适的 group 数。
    要求 channels 能被 group 数整除。
    """
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1



class ConvGNAct(nn.Module):
    """
    基础卷积模块：
        Conv2d -> GroupNorm -> SiLU

    说明：
        这里使用 GroupNorm 而不是 BatchNorm。
        原因是你的 batch size 可能较小，尤其 DDP 多卡训练时，
        每张 GPU 上的 batch 更小，GroupNorm 通常更稳定。
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,),
            nn.GroupNorm(
                num_groups=get_group_num(out_channels),
                num_channels=out_channels,),
            nn.SiLU(inplace=True),)    # SiLU激活函数？ 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)



class ResidualBlock2D(nn.Module):
    """
    简单残差块。?
    输入:
        x: [B, C, H, W]
    输出:
        out: [B, C, H, W]
    作用:
        在不改变空间分辨率的情况下，增强局地特征提取能力。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,),
            nn.GroupNorm(
                num_groups=get_group_num(channels),
                num_channels=channels,),)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        # 残差连接：保留原始信息，同时学习修正量     NOTE : 不好理解
        out = out + residual
        out = self.act(out)
        return out



class DownsampleBlock2D(nn.Module):
    """
    下采样模块。
    输入:
        [B, C_in, H, W]
    输出:
        [B, C_out, ceil(H/2), ceil(W/2)] 近似

    这里通过 stride=2 的 Conv2d 完成下采样。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = ConvGNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,)
        self.res = ResidualBlock2D(out_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res(x)
        return x







class AtmosphericEncoder(nn.Module):
    """
    Atmospheric Encoder.
    作用：
        输入两个历史时刻的所有气象变量和高度层数据，
        提取用于未来 cloud mask prediction 的大气背景特征。
    输入:
        x_all_flat: [B, 234, 41, 71]
    输出:
        F_atm: [B, out_channels, 11, 18]
    默认参数:
        in_channels = 234
            2 个时刻 × 9 个变量 × 13 个高度层
        base_channels = 64
            第一层特征通道数
        out_channels = 128
            最终输出特征通道数
    """
    def __init__(
        self,
        in_channels: int = 234,
        base_channels: int = 64,
        out_channels: int = 128,):
        super().__init__()
        # Step 1: Conv stem
        # [B, 234, 41, 71] -> [B, 64, 41, 71]
        self.stem = nn.Sequential(
            ConvGNAct(in_channels, base_channels, stride=1),
            ResidualBlock2D(base_channels),)
        # Step 2: 第一次下采样
        # [B, 64, 41, 71] -> [B, 96, 21, 36]
        self.down1 = DownsampleBlock2D(
            in_channels=base_channels,
            out_channels=base_channels + base_channels // 2,)
        # Step 3: 第二次下采样
        # [B, 96, 21, 36] -> [B, 128, 11, 18]
        self.down2 = DownsampleBlock2D(
            in_channels=base_channels + base_channels // 2,
            out_channels=out_channels,)
        # Step 4: 输出前再做一次残差特征提取
        # [B, 128, 11, 18] -> [B, 128, 11, 18]
        self.out_block = ResidualBlock2D(out_channels)
    def forward(self, x_all_flat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_all_flat:
            shape: [B, 234, 41, 71]
        Returns
        -------
        F_atm:
            shape: [B, 128, 11, 18]
        """
        x = self.stem(x_all_flat)
        x = self.down1(x)
        x = self.down2(x)
        F_atm = self.out_block(x)
        return F_atm

