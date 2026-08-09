import torch.nn as nn
import torch
import torch.nn.functional as F

# discriminator需要专门一行行运行理解一下

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



class PatchGAN3DDiscriminator(nn.Module):
    """Conditional 3D PatchGAN discriminator.

    Input is concat([x, y], dim=1):    # 为什么要合并？
        x: [B, 18, 13, 41, 71]
        y: [B, 48, 13, 41, 71]
        concat: [B, 66, 13, 41, 71]
    Output: patch logits [B, 1, D', H', W'].
    """

    def __init__(self, in_channels: int = 66, base_channels: int = 32, spectral_norm: bool = True) -> None:  # 默认要用谱标准化。
        super().__init__()
        c = base_channels   # 就是卷积的核的数量
        layers = [
            _sn(nn.Conv3d(in_channels, c, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c, c * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False), spectral_norm),
            nn.GroupNorm(_group_count(c * 2), c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c * 2, c * 4, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False), spectral_norm),
            nn.GroupNorm(_group_count(c * 4), c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=False), spectral_norm),
            nn.GroupNorm(_group_count(c * 8), c * 8),    # 分组进行标准化？具体什么含义？
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv3d(c * 8, 1, kernel_size=3, stride=1, padding=1), spectral_norm),
        ]
        self.net = nn.Sequential(*layers)   # 为什么要加个*  ？  *layers 把列表“拆开”

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))   # 就是把这个x，y的tensor合并，然后在经过上面定义的模型架构，   这里的x，和y具体是什么？

