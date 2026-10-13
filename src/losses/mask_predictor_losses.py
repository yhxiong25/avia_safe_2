

'''

'''
# mask_losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaBalancedBCELoss(nn.Module):
    """
    α-balanced BCE Loss.
    作用：
        处理有云/无云样本不平衡问题。
    普通 BCE:
        BCE = - y log(p) - (1-y) log(1-p)
    α-balanced BCE:
        BCE = - α * y log(p) - (1-α) * (1-y) log(1-p)
        BCE = - ⍺t * log(pt)    
    其中：
        y = 1 表示有云
        y = 0 表示无云
    参数：
        alpha_pos:
            正样本，也就是有云格点的权重。
            如果有云区域较少，可以设大一些，例如 0.75。
    """
    def __init__(self, alpha_pos: float = 0.75):
        super().__init__()
        if not 0.0 < alpha_pos < 1.0:
            raise ValueError("alpha_pos should be between 0 and 1.")
        self.alpha_pos = alpha_pos
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits:
            模型输出的 logits。
            shape: [B, C, H, W]
        targets:
            binary mask target。
            shape: [B, C, H, W]
            数值应为 0 或 1。
        Returns
        -------
        loss:
            一个标量 loss。
        """
        targets = targets.float()
        # BCEWithLogitsLoss = sigmoid + BCE
        # reduction="none" 表示先保留每个格点的 loss
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",)
        # 正样本权重 alpha_pos，负样本权重 1-alpha_pos
        weights = (
            self.alpha_pos * targets
            + (1.0 - self.alpha_pos) * (1.0 - targets))
        loss = bce * weights
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    作用：
        降低大量容易分类样本的权重，
        让模型更关注难分类的云区、边界区、稀有云区。
    公式思想：
        FL = alpha_t * (1 - p_t)^gamma * BCE
    其中：
        p_t 表示模型对真实类别的预测概率。
    默认参数：
        gamma = 1.5
        alpha = 0.25
    这是你希望采用的 AviaSafe 风格默认设置。
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 1.5,
    ):
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha should be between 0 and 1.")
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits:
            模型输出的 logits。
            shape: [B, C, H, W]
        targets:
            binary mask target。
            shape: [B, C, H, W]
        Returns
        -------
        loss:
            一个标量 focal loss。
        """
        targets = targets.float()
        # 每个格点的 BCE
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",)
        # pt = p       if target = 1
        # pt = 1 - p   if target = 0
        #
        # 由于 bce = -log(pt)
        # 所以 pt = exp(-bce)
        pt = torch.exp(-bce)
        # alpha_t:
        # target=1 时为 alpha
        # target=0 时为 1-alpha
        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets))
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


class MaskPredictorLoss(nn.Module):
    """
    Mask Predictor 的组合损失函数。
    总损失：
        total_loss = lambda_bce * alpha_balanced_bce
                   + lambda_focal * focal_loss
    你的设定：
        lambda_bce = 1
        lambda_focal = 1
    输入：
        mask_logits:
            [B, lead_steps*4*13, H, W]
        mask_target:
            可以是：
            [B, lead_steps*4*13, H, W]
            或
            [B, lead_steps, 4, 13, H, W]
    """

    def __init__(
        self,
        alpha_bce_pos: float = 0.75,
        focal_alpha: float = 0.25,
        focal_gamma: float = 1.5,
        lambda_bce: float = 1.0,
        lambda_focal: float = 1.0,
    ):
        super().__init__()
        self.bce_loss = AlphaBalancedBCELoss(
            alpha_pos=alpha_bce_pos,)
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,)
        self.lambda_bce = lambda_bce
        self.lambda_focal = lambda_focal
    def _reshape_target_if_needed(
        self,
        mask_logits: torch.Tensor,
        mask_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        如果 mask_target 是 [B, lead_steps, 4, 13, H, W]，
        自动 reshape 成 [B, lead_steps*4*13, H, W]。
        """
        if mask_target.ndim == 6:
            B, T, V, D, H, W = mask_target.shape
            mask_target = mask_target.view(B, T * V * D, H, W)
        if mask_logits.shape != mask_target.shape:
            raise ValueError(
                f"mask_logits shape {tuple(mask_logits.shape)} "
                f"does not match mask_target shape {tuple(mask_target.shape)}")
        return mask_target
    def compute_components(
        self,
        mask_logits: torch.Tensor,
        mask_target: torch.Tensor,):
        """
        返回总损失和两个子损失，方便日志记录和画图。
        """
        mask_target = self._reshape_target_if_needed(
            mask_logits,
            mask_target,)
        loss_bce = self.bce_loss(mask_logits, mask_target)
        loss_focal = self.focal_loss(mask_logits, mask_target)
        loss_total = (
            self.lambda_bce * loss_bce
            + self.lambda_focal * loss_focal)
        return {
            "total": loss_total,
            "alpha_balanced_bce": loss_bce,
            "focal": loss_focal,
            "weighted_bce": self.lambda_bce * loss_bce,
            "weighted_focal": self.lambda_focal * loss_focal,}
    def forward(
        self,
        mask_logits: torch.Tensor,
        mask_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        直接返回总损失。
        """
        loss_dict = self.compute_components(mask_logits, mask_target)
        return loss_dict["total"]



