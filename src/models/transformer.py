# swin_context_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP projection
class MLPProjection2D(nn.Module):
    """用MLP维度下采样
    MLP Projection 模块。
    在图像/网格特征中，常用 1x1 Conv 来实现 MLP projection。
    因为 1x1 Conv 本质上是在每个空间格点上，对 channel 维做线性变换。
    输入:
        F: [B, 256, 11, 18]
    输出:
        F_proj: [B, 128, 11, 18]
    """
    def __init__(self, in_channels=256, embed_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),   # kernel size=1的卷积？说明一个点提取一种特征，共提取128种特征
            nn.GroupNorm(num_groups=8, num_channels=embed_dim),   # GN，BN，LN，反正就是在不同维度上做标准化，防止梯度爆炸这些。
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=embed_dim),
            nn.SiLU(inplace=True),)
    def forward(self, x):
        return self.proj(x)




def window_partition(x, window_size):
    """
    将 feature map 切成多个 window。
    输入:
        x: [B, H, W, C]
    输出:
        windows: [num_windows * B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将 window 重新拼回 feature map。
    输入:
        windows: [num_windows * B, window_size, window_size, C]
    输出:
        x: [B, H, W, C]
    """
    num_windows_per_sample = (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows_per_sample
    C = windows.shape[-1]
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        C,)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    return x


# ============================================================
# 3. Window Self-Attention
# ============================================================

class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention.
    作用:
        在每个局地 window 内部做 self-attention。
    输入:
        x: [num_windows * B, N, C]
    其中:
        N = window_size * window_size
        C = embed_dim
    输出:
        out: [num_windows * B, N, C]
    """
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.scale = self.head_dim ** -0.5
        # 一次性生成 Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        # attention 后的输出映射
        self.proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, attn_mask=None):
        B_windows, N, C = x.shape
        # qkv: [B_windows, N, 3*C]
        qkv = self.qkv(x)   # 直接一个全链接层映射就是QKV？ 
        # [B_windows, N, 3, num_heads, head_dim]
        qkv = qkv.view(
            B_windows,
            N,
            3,
            self.num_heads,
            self.head_dim,)
        # [3, B_windows, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # attention score:
        # [B_windows, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # shifted window 需要 attention mask，防止不同区域错误混合
        if attn_mask is not None:
            num_windows = attn_mask.shape[0]
            B = B_windows // num_windows
            attn = attn.view(
                B,
                num_windows,
                self.num_heads,
                N,
                N,)
            attn = attn + attn_mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = torch.softmax(attn, dim=-1)
        # [B_windows, num_heads, N, head_dim]
        out = attn @ v
        # [B_windows, N, C]
        out = out.transpose(1, 2).contiguous().view(B_windows, N, C)
        out = self.proj(out)
        return out


# ============================================================
# 4. Swin Transformer Block
# ============================================================

class SwinBlock2D(nn.Module):
    """
    简化版 Swin Transformer Block。
    输入:
        x: [B, C, H, W]   # Swin Transformer的输入输出竟然维度不变？？
    输出:
        x: [B, C, H, W]
    主要步骤:
        1. LayerNorm
        2. Window Attention
        3. 残差连接
        4. LayerNorm
        5. MLP
        6. 残差连接
    shift_size = 0:
        普通 window attention
    shift_size = window_size // 2:
        shifted window attention
    """
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        window_size=6,
        shift_size=0,
        mlp_ratio=4.0,):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),)
    def _create_attn_mask(self, Hp, Wp, device):
        """
        为 shifted window attention 创建 mask。
        作用:
            shifted window 后，不同原始窗口的 token 可能被放到同一个 window 中。
            attention mask 用来阻止这些不该互相注意的位置。
        """
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),)
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask
    def forward(self, x):
        """
        x:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        # Swin Block 更习惯使用 [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()    # .permute就是更改数组的维度顺序。
        # ----------------------------------------------------
        # 1. padding 到 window_size 的整数倍
        # ----------------------------------------------------
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        # ----------------------------------------------------
        # 2. Window Attention
        # ----------------------------------------------------
        shortcut = x
        x_norm = self.norm1(x)
        # shifted window
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x_norm,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),)
            attn_mask = self._create_attn_mask(Hp, Wp, x.device)
        else:
            shifted_x = x_norm
            attn_mask = None
        # 切 window
        x_windows = window_partition(shifted_x, self.window_size)
        # [num_windows*B, window_size*window_size, C]
        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            C,)
        # window attention
        attn_windows = self.attn(x_windows, attn_mask=attn_mask)
        # 还原 window 形状
        attn_windows = attn_windows.view(
            -1,
            self.window_size,
            self.window_size,
            C,)
        # 拼回 feature map
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            Hp,
            Wp,)
        # reverse shift
        if self.shift_size > 0:
            x_attn = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),)
        else:
            x_attn = shifted_x
        # 残差连接
        x = shortcut + x_attn
        # ----------------------------------------------------
        # 3. MLP
        # ----------------------------------------------------
        shortcut = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = shortcut + x_mlp
        # ----------------------------------------------------
        # 4. 去掉 padding，恢复原始 H, W
        # ----------------------------------------------------
        x = x[:, :H, :W, :].contiguous()
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# ============================================================
# 5. MLP Projection + 2 Swin Blocks
# ============================================================

class SwinContextModule(nn.Module):
    """
    对融合特征 F 进行处理。
    输入:
        F: [B, 256, 11, 18]..  是指PhysicalEncoder和AtmosphericEncoder的输出合并
    输出:
        F_context: [B, 128, 11, 18]
    结构:
        MLP Projection
        ↓
        SwinBlock 1: window attention     # 这里已经有了两个Swin transformer的处理。  这个模块相当于把对F的MLP proj，两个Swin block处理，都打包了。
        ↓
        SwinBlock 2: shifted window attention
    """
    def __init__(
        self,
        in_channels=256,
        embed_dim=128,
        num_heads=4,
        window_size=6,
    ):
        super().__init__()
        # [B, 256, 11, 18] -> [B, 128, 11, 18]
        self.proj = MLPProjection2D(
            in_channels=in_channels,
            embed_dim=embed_dim,)
        # 第一个 Swin Block：普通 window attention
        self.swin_block1 = SwinBlock2D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,)
        # 第二个 Swin Block：shifted window attention
        self.swin_block2 = SwinBlock2D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,)
    def forward(self, F):
        """
        Parameters
        ----------
        F:
            融合后的特征图。
            shape: [B, 256, 11, 18]
        Returns
        -------
        F_context:
            经过 MLP projection 和两个 Swin Blocks 后的特征图。
            shape: [B, 128, 11, 18]
        """
        F = self.proj(F)
        F = self.swin_block1(F)   # Swin transformer处理后的数据维度，不变吗？
        F_context = self.swin_block2(F)
        return F_context