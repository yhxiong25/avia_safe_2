import torch
import torch.nn as nn


def get_group_num(channels: int, max_groups: int = 8) -> int:
    """
    为 GroupNorm 自动选择合适的 group 数。
    GroupNorm 要求：
        channels % num_groups == 0
    例如：
        channels = 64  -> num_groups = 8
        channels = 128 -> num_groups = 8
    """
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1




class ConvGNAct(nn.Module):
    """
    基础卷积模块：
        Conv2d -> GroupNorm -> SiLU
    输入:
        x: [B, C_in, H, W]
    输出:
        x: [B, C_out, H_out, W_out]
    说明:
        这里使用 GroupNorm，而不是 BatchNorm。
        因为你的 batch size 可能较小，GroupNorm 更稳定。
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,),
            nn.GroupNorm(
                num_groups=get_group_num(out_channels),
                num_channels=out_channels,),
            nn.SiLU(inplace=True),)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)




class ResidualBlock2D(nn.Module):
    """
    简单 2D 残差块。
    作用:
        在不改变空间分辨率的情况下，进一步提取局地空间特征。
    输入:
        x: [B, C, H, W]
    输出:
        out: [B, C, H, W]
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
        # 残差连接：学习对原始特征的修正量
        out = out + residual
        out = self.act(out)
        return out



class DownsampleBlock2D(nn.Module):
    """
    下采样模块。
    使用 stride=2 的 Conv2d 实现空间降采样。
    输入:
        [B, C_in, H, W]
    输出:
        [B, C_out, H/2, W/2] 近似
    对你的网格:
        41 x 71 -> 21 x 36
        21 x 36 -> 11 x 18
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


class PhysicalEncoder(nn.Module):
    """
    Physical Encoder / E_Mask.
    作用:
        输入由 mask_cloud 和 mask_ic 拼接得到的物理先验信息，
        提取未来有云区域预测所需要的 mask guidance feature。
    输入:
        E_mask: [B, 65, 41, 71]
    其中:
        65 = 52 + 13
        52 = 4 个云变量 × 13 个高度层的 cloud mask
        13 = 13 个高度层的 IC potential mask
    输出:
        F_mask: [B, 128, 11, 18]
    这个输出后续可以和 Atmospheric Encoder 的 F_atm 拼接。
    """
    def __init__(
        self,
        in_channels: int = 65,
        base_channels: int = 64,
        out_channels: int = 128,):
        super().__init__()
        # --------------------------------------------------
        # Step 1: Conv stem
        # [B, 65, 41, 71] -> [B, 64, 41, 71]
        # --------------------------------------------------
        self.stem = nn.Sequential(
            ConvGNAct(
                in_channels=in_channels,
                out_channels=base_channels,
                stride=1,),
            ResidualBlock2D(base_channels),)
        # --------------------------------------------------
        # Step 2: 第一次下采样
        # [B, 64, 41, 71] -> [B, 96, 21, 36]
        # --------------------------------------------------
        mid_channels = base_channels + base_channels // 2
        self.down1 = DownsampleBlock2D(
            in_channels=base_channels,
            out_channels=mid_channels,)
        # --------------------------------------------------
        # Step 3: 第二次下采样
        # [B, 96, 21, 36] -> [B, 128, 11, 18]
        # --------------------------------------------------
        self.down2 = DownsampleBlock2D(
            in_channels=mid_channels,
            out_channels=out_channels,)
        # --------------------------------------------------
        # Step 4: 输出前再提取一次特征
        # [B, 128, 11, 18] -> [B, 128, 11, 18]
        # --------------------------------------------------
        self.out_block = ResidualBlock2D(out_channels)
    def forward(self, E_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        E_mask:
            mask_cloud 和 mask_ic 拼接后的输入。
            shape: [B, 65, 41, 71]
        Returns
        -------
        F_mask:
            物理 mask 特征图。
            shape: [B, 128, 11, 18]
        """
        x = self.stem(E_mask)
        x = self.down1(x)
        x = self.down2(x)
        F_mask = self.out_block(x)
        return F_mask





