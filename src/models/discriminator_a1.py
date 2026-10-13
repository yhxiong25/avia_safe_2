import torch
import torch.nn as nn


def _group_count(channels: int, max_groups: int = 8) -> int:   # ？
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:   # 求除了之后的余数。 如果base channel d取得很小，比如4，那么g返回的就是4
            return g
    return 1


def _sn(module: nn.Module, use_spectral_norm: bool) -> nn.Module:    # use_spectral_norm  ? 什么东西？ 谱归一化？ 
    '''用于在 神经网络层的权重参数 上施加谱归一化约束，使其满足 1-Lipschitz 连续性，常用于 GAN 判别器 以稳定训练。
    什么是谱归一化？    谱归一化会约束权重矩阵的最大奇异值，简单理解是限制判别器不要变化得太剧烈。
      这里的discriminator具体是什么逻辑啊？
    '''
    return nn.utils.spectral_norm(module) if use_spectral_norm else module



class VoxelLocalDiscriminator(nn.Module):
    """四层 stride-1 的条件判别器，空间尺寸保持 13×41×71。

    输入是 concat([x, y_cloud], dim=1)，不包含温度：
        x: [B, 18, 13, 41, 71]
        y_cloud: [B, 4*lead, 13, 41, 71]
    输出 logits: [B, 4*lead, 13, 41, 71]，每个时效、云要素、高度和格点一个分数。
    通道顺序与 y_cloud 一致，每个时效内为 CIWC, CLWC, CRWC, CSWC。
    """

    def __init__(
        self,
        in_channels: int = 66,
        out_channels: int = 48,
        base_channels: int = 8,
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        c = base_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            _sn(nn.Conv3d(in_channels, c, kernel_size=3, stride=1, padding=1), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=False), spectral_norm),
            nn.GroupNorm(_group_count(c * 2), c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=False), spectral_norm),
            nn.GroupNorm(_group_count(c * 4), c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c * 4, out_channels, kernel_size=3, stride=1, padding=1), spectral_norm),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))
