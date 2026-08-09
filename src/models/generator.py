import torch.nn as nn
import torch
import torch.nn.functional as F



def _group_count(channels: int, max_groups: int = 8) -> int:   # ？  计算出来能整除batch个数的group组数。
    for g in reversed(range(1, max_groups + 1)):    # reversed？ 
    # reversed， 从大到小遍历整数，找到最大能够整除channels的整数就退出 
        if channels % g == 0:
            return g   # 返回能整除channels的最大整数，但是这个整数不能超过max_groups。。。因为base_channels = 32，所以就是8
    return 1     #这个函数无论怎样都返回1？   # 


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(    # nn.Sequential is a class in PyTorch that allows you to build neural network models by stacking layers in a sequential manner.
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),   # groupNorm具体是怎么回事？  GroupNorm：按通道分组归一化，比BatchNorm更适合小batch和3D数据  y = gamma * (x - mean) / sqrt(var + eps) + beta
            nn.SiLU(inplace=True),   #? 不是Relu吗？   Sigmoid Linear Unit (SiLU) ； silu(x) = x * sigmoid(x)
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # 这里的convBlock3D数据输入in_ch维度，输出out_ch维度


class DownBlock3D(nn.Module):
    """Downsample only latitude-longitude; keep pressure-level depth unchanged."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    """Upsample latitude-longitude to the skip feature size and fuse."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)   #？ 三线性插值
        x = self.reduce(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3DGenerator(nn.Module):
    """4-level anisotropic 3D U-Net generator.

    Input : [B, 18, 13, 41, 71]
    Output: [B, 48, 13, 41, 71]
    """

    def __init__(self, in_channels: int = 18, out_channels: int = 48, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock3D(in_channels, c)
        self.enc2 = DownBlock3D(c, c * 2)
        self.enc3 = DownBlock3D(c * 2, c * 4)
        self.enc4 = DownBlock3D(c * 4, c * 8)
        self.bottleneck = DownBlock3D(c * 8, c * 16)

        self.up4 = UpBlock3D(c * 16, c * 8, c * 8)
        self.up3 = UpBlock3D(c * 8, c * 4, c * 4)
        self.up2 = UpBlock3D(c * 4, c * 2, c * 2)
        self.up1 = UpBlock3D(c * 2, c, c)
        self.out_conv = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out_conv(x)

