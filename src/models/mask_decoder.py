'''
生成mask probability logits

'''


# mask_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_num(channels: int, max_groups: int = 8) -> int:
    """
    为 GroupNorm 自动选择合适的 group 数。
    GroupNorm 要求 channels 能被 num_groups 整除。
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
        [B, C_in, H, W]
    输出:
        [B, C_out, H, W]
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,),
            nn.GroupNorm(
                num_groups=get_group_num(out_channels),
                num_channels=out_channels,),
            nn.SiLU(inplace=True),)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleConvBlock(nn.Module):
    """
    上采样模块。
    使用 interpolate + Conv2d，而不是 ConvTranspose2d。
    原因:
        interpolate + Conv2d 更稳定，
        不容易产生 ConvTranspose2d 常见的棋盘格伪影。
    输入:
        [B, C_in, H, W]
    输出:
        [B, C_out, target_H, target_W]
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvGNAct(in_channels, out_channels),
            ConvGNAct(out_channels, out_channels),)
    def forward(self, x: torch.Tensor, size):
        """
        Parameters
        ----------
        x:
            输入特征图。
        size:
            目标空间尺寸，例如 (21, 36) 或 (41, 71)。
        """
        # 先用双线性插值恢复空间分辨率
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,)
        # 再用卷积细化上采样后的特征
        x = self.conv(x)
        return x


class MaskUpsampleDecoder(nn.Module):
    """
    Mask Predictor 的上采样解码器。
    作用:
        将 Swin Block 输出的低分辨率特征 F_context
        恢复到原始中国区域网格大小，并输出 mask logits。
    输入:
        F_context: [B, 128, 11, 18]
    输出:
        mask_logits: [B, lead_steps * num_cloud_vars * num_levels, 41, 71]
    默认:
        lead_steps = 4
        num_cloud_vars = 4
        num_levels = 13
    因此:
        out_channels = 4 * 4 * 13 = 208
        mask_logits: [B, 208, 41, 71]
    """
    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 96,
        hidden_channels: int = 64,
        lead_steps: int = 4,
        num_cloud_vars: int = 4,
        num_levels: int = 13,
        output_size=(41, 71),):
        super().__init__()
        self.lead_steps = lead_steps
        self.num_cloud_vars = num_cloud_vars
        self.num_levels = num_levels
        self.output_size = output_size
        out_channels = lead_steps * num_cloud_vars * num_levels
        # --------------------------------------------------
        # 11 x 18 -> 21 x 36
        # --------------------------------------------------
        self.up1 = UpsampleConvBlock(
            in_channels=in_channels,
            out_channels=mid_channels,)
        # --------------------------------------------------
        # 21 x 36 -> 41 x 71
        # --------------------------------------------------
        self.up2 = UpsampleConvBlock(
            in_channels=mid_channels,
            out_channels=hidden_channels,)
        # --------------------------------------------------
        # 输出层:
        # [B, hidden_channels, 41, 71]
        #       -> [B, lead_steps*4*13, 41, 71]
        #
        # 这里不加 sigmoid。
        # 因为训练时 FocalLoss/BCEWithLogitsLoss 通常直接吃 logits。
        # --------------------------------------------------
        self.out_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,)
    def forward(self, F_context: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        F_context:
            shape: [B, 128, 11, 18]
        Returns
        -------
        mask_logits:
            shape: [B, lead_steps*4*13, 41, 71]
        """
        # 第一次上采样
        x = self.up1(F_context, size=(21, 36))
        # 第二次上采样，恢复到原始网格
        x = self.up2(x, size=self.output_size)
        # 输出 logits，不做 sigmoid
        mask_logits = self.out_conv(x)
        return mask_logits
    def logits_to_prob(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """
        将 logits 转换为概率图。
        输出:
            mask_prob: [B, lead_steps, 4, 13, 41, 71]
        """
        B, C, H, W = mask_logits.shape
        mask_prob = torch.sigmoid(mask_logits)
        mask_prob = mask_prob.view(
            B,
            self.lead_steps,
            self.num_cloud_vars,
            self.num_levels,
            H,
            W,)
        return mask_prob
    def logits_to_binary_mask(
        self,
        mask_logits: torch.Tensor,
        threshold: float = 0.5,) -> torch.Tensor:
        """
        将 logits 转换为 binary cloud mask。
        注意:
            这个函数只建议用于验证、绘图和推理。
            训练时不要用 hard binary mask，因为 threshold 操作不可导。
        输出:
            mask_binary: [B, lead_steps, 4, 13, 41, 71]
        """
        mask_prob = self.logits_to_prob(mask_logits)
        mask_binary = (mask_prob >= threshold).float()
        return mask_binary    # 之后需要，更改这些模型的流程。   在src的源代码中定义单个模块，然后细节处理，维度变换，多个模块接续处理这些，在主程序中进行就好。 不要全部打包到模型代码文件中。 流程打包到模块代码本身就很怪。 所以之后要改。

