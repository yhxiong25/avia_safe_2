# ERA5 中国区域云预报 GAN 技术说明

本文按当前训练入口 `train_GAN_test8_a1.py`、判别器 `src/models/discriminator_a1.py`（`VoxelLocalDiscriminator`）和生成损失 `src/losses/GAN_losses.py` 整理。

---

## 1. 任务与整体角色

这是**条件 GAN**。生成器根据历史大气场预报未来云水和温度；判别器不单独判断“这张云图真不真”，而是判断**在给定历史条件 $x$ 下，未来云场 $y$ 是否像真值**。

| 角色 | 当前实现 |
| --- | --- |
| 生成器 $G$ | 默认 ConvGRU + 3D U-Net（`--generator_version convgru`），也可换纯 3D U-Net |
| 判别器 $D$ | `VoxelLocalDiscriminator`，条件输入 $\mathrm{concat}(x, y_{\mathrm{cloud}})$ |
| 对抗对象 | **只对 4 个云要素**。温度不进 $D$，只走重建损失 |
| 训练阶段 | 可先监督预训练 $G$，再 GAN；也可 `--pretrain_epochs 0` 直接对抗 |

数据网格：气压 13 层 × 纬度 41 × 经度 71。时间上默认 2 个历史时刻（间隔 6 h）预报 `lead_steps` 个未来时刻。

---

## 2. 张量约定（读后续公式时对照）

**输入 $x$**（归一化后 flatten）：

$$
x \in \mathbb{R}^{B \times 18 \times 13 \times 41 \times 71}
= B \times (2 \times 9) \times D \times H \times W
$$

每个历史时刻 9 个变量，顺序为位势高度、温度、比湿、$u$、$v$，再加 CIWC、CLWC、CRWC、CSWC。前 5 个为 z-score，4 个云量为 $z=\log_{10}(1+q/q_0)$，$q_0=10^{-8}$。

**生成器输出 / 标签 $y$**（每个 lead 内：4 云 + 1 温度）：

$$
y \in \mathbb{R}^{B \times 5L \times 13 \times 41 \times 71}
$$

其中 $L=$ `lead_steps`。云通道仍是 log 空间；温度是 z-score。

送入 $D$ 之前切开：

```text
y  --split_cloud_temp-->  y_cloud [B, 4L, 13, 41, 71]
                          y_temp  [B,  L, 13, 41, 71]   # 不进 D
```

因此 $D$ 的通道数为

$$
C_{\mathrm{in}} = 18 + 4L
$$

`lead_steps=1` 时为 22；`lead_steps=12` 时为 66。旧注释里写的 `[B,66]` 对应 12 步、4 云、无温度进 $D$ 的情形，与当前“温度不进判别器”一致。

---

## 3. 判别器架构

实现类：`VoxelLocalDiscriminator`。

### 3.1 为什么叫 Voxel-Local，而不是 PatchGAN

`PatchGAN3DDiscriminator` 在水平方向用 `stride=(1,2,2)` 下采样，输出是较粗的 patch logit。

`VoxelLocalDiscriminator` **全部卷积 `stride=1`、`padding=1`、`kernel=3`**，空间尺寸不变：

$$
\mathrm{logit} \in \mathbb{R}^{B \times 4L \times 13 \times 41 \times 71}
$$

通道顺序与 $y_{\mathrm{cloud}}$ 相同：每个 lead 内依次是 CIWC、CLWC、CRWC、CSWC。因此每个 lead、每个云要素、每个气压层、每个经纬格点各有一个 logit。这是“体素级、分要素判别”：在原始网格上打分，而不是先压成大块 patch，也不再让四个云要素共用一张分数图。

### 3.2 层结构（`base_channels_d = c`，近期实验常用 $c=8$）

| 顺序 | 模块 | 通道 | 备注 |
| --- | --- | --- | --- |
| 1 | SpectralNorm Conv3d $3\times3\times3$ | $C_{\mathrm{in}} \to c$ | 无 GroupNorm |
| 2 | LeakyReLU(0.2) | | |
| 3 | SpectralNorm Conv3d | $c \to 2c$ | `bias=False`，后接 GroupNorm |
| 4 | GroupNorm + LeakyReLU(0.2) | | |
| 5 | SpectralNorm Conv3d | $2c \to 4c$ | 同上 |
| 6 | GroupNorm + LeakyReLU(0.2) | | |
| 7 | SpectralNorm Conv3d | $4c \to 4L$ | 每个云通道一个 logit，无激活 |

感受野：四层 $3\times3\times3$、无下采样，每个输出体素大约看输入的 $9\times9\times9$ 邻域（气压 × 纬 × 经）。它看的是**局部一致性**，不是整张中国区域是否“像一张图”。

### 3.3 谱归一化与 GroupNorm

- **谱归一化**（默认开启）：约束每层权重最大奇异值，使 $D$ 更接近 Lipschitz，减轻 GAN 中 $D$ 过陡、梯度爆炸。
- **GroupNorm**：中间两层不用 BatchNorm。云场大量格点为 0、batch 往往不大（如 4），BN 统计不稳；GN 按通道分组归一化。
- 第一层和最后一层不加 GN，避免过早抹掉输入尺度、以及把各云通道的 logit 再标准化。

### 3.4 前向数据流

```text
x [B,18,D,H,W]     y_cloud [B,4L,D,H,W]
        \                /
         concat dim=1
              |
        [B, 18+4L, 13, 41, 71]
              |
     Conv3d + SN + LeakyReLU
              |
     Conv3d + SN + GN + LeakyReLU
              |
     Conv3d + SN + GN + LeakyReLU
              |
     Conv3d + SN  →  logit [B,4L,13,41,71]
              |
         sigmoid →  伪概率 ∈ (0,1)
```

代码就是：

```python
def forward(self, x, y):
    return self.net(torch.cat([x, y], dim=1))
```

条件信息完全靠通道拼接进入，没有单独的 embedding 或 FiLM。$D$ 必须同时看见“历史天气”和“未来云”，才能判断云是否与上游环流、水汽等条件匹配。若只输入 $y$，$D$ 容易学成“有云/无云纹理分类器”，而不是条件预报的真假判别器。

---

## 4. 对抗损失设计

对抗损失，真假样本都走 `masked_adv_loss`。

### 4.1 设计目标（相对标准 GAN）

标准 GAN 常用 $\mathrm{BCE}(\sigma(\ell), \{0,1\})$ 或 LSGAN 的 $\mathrm{MSE}(\ell, \{0,1\})$，并对整图求平均。这里针对**云量极度稀疏、极值少、希望 GAN 盯有云区**做了三处改动：

1. **LSGAN 风格的概率 MSE** + **Focal BCE**，各占一半；
2. **只用真值有云格点加权**（`beta_no_cloud=0`）；
3. **$D$ 过强时不更新 $D$**（门控）。

### 4.2 单个体素上的两项

记 $\ell$ 为 logit，$p=\sigma(\ell)$，标签 $t\in\{t_{\mathrm{real}}, t_{\mathrm{fake}}\}$（命令行默认 1 和 0）。

**（1）概率空间 MSE（LSGAN 变体）**

$$
\mathcal{L}_{\mathrm{mse}} = (p - t)^2
$$

注意：教材 LSGAN 常对 **logit** 做 MSE；这里对 **sigmoid 之后的概率**做 MSE。$p$ 被压在 $(0,1)$，比直接拟合 logit 更温和。

**（2）Focal loss（在 logit 上）**

先算逐点 BCE：

$$
\mathrm{BCE} = -\bigl[t\log p + (1-t)\log(1-p)\bigr]
$$

再按“当前预测有多对”降权：

$$
p_t = p\,t + (1-p)(1-t),\qquad
\mathcal{L}_{\mathrm{focal}} = (1-p_t)^{\gamma}\,\mathrm{BCE}
$$

$\gamma=1.5$。已经分得很对的体素（$p_t\to 1$）权接近 0；分错的体素权更大。意图是避免 $D$ 只靠大片晴空“轻松正确”就把梯度吃满。

**合成图**

$$
\mathcal{L}_{\mathrm{map}} = 0.5\,\mathcal{L}_{\mathrm{mse}} + 0.5\,\mathcal{L}_{\mathrm{focal}}
$$

### 4.3 有云掩膜：对抗项只作用在真值有云处

掩膜来自**真值云通道**（log 空间），不是来自预报：

$$
m = \mathbf{1}[y_{\mathrm{cloud}} > 0] \in \{0,1\}^{B\times 4L\times 13\times 41\times 71}
$$

权重（当前 `beta_no_cloud=0`）：

$$
w = m + \beta_{\mathrm{clear}}(1-m) = m
$$

标量损失是加权平均：

$$
\mathcal{L}_{\mathrm{adv}}(\ell, t, m)
= \frac{\sum (\mathcal{L}_{\mathrm{map}} \odot w)}{\sum w + \varepsilon}
$$

含义：

- 真值无云的格点**不进入对抗损失**（重建损失仍会管它们）；
- 真值有云的格点，$D$ 要能认出真云、揭穿假云；$G$ 要让这些位置的 $D$ 输出靠近“真”。

**实现上的维度对齐（需要向老师说明）：**  
$D$ 的 logit 与云掩膜形状相同，都是 $[B,4L,D,H,W]$，按通道逐点相乘，不再把一张分数图广播到所有云要素上。`masked_adv_loss` 会检查两者形状一致。某个格点上只有 CIWC 为真值有云时，只有 CIWC 这一通道进入对抗损失，CLWC/CRWC/CSWC 在该点的分数不计入。

### 4.4 判别器一步

每个 batch：

1. `fake_y = G(x)`，切断梯度，只取 `fake_cloud`；
2. `real_logits = D(x, y_cloud)`，标签 $t_{\mathrm{real}}=1$；
3. `fake_logits = D(x, fake_cloud)`，标签 $t_{\mathrm{fake}}=0$；
4. **两边都用同一张真值掩膜 $m$**；
5. $ \mathcal{L}_D = \frac12(\mathcal{L}_{\mathrm{adv}}^{\mathrm{real}} + \mathcal{L}_{\mathrm{adv}}^{\mathrm{fake}}) $。

### 4.5 $D$ 的更新门控

多卡时先对 $\mathcal{L}_D$ 做 `all_reduce` 平均，各 rank 用同一个布尔量：

$$
\text{update } D \iff \mathcal{L}_D > 0.15
$$

小于等于 0.15 时只算 $\mathcal{L}_D$、不反传、不 `optimizer.step`。目的是：$D$ 已经很容易分真假时，停止强化 $D$，避免 $G$ 的对抗梯度消失。阈值 0.15 是经验数，不是理论最优。

注意：即使本步不更新 $D$，日志里的 `D_loss` 仍会记录该 batch 的 $\mathcal{L}_D$。

### 4.6 生成器一步

再前向一次 $G$（带梯度），取 `fake_cloud`：

$$
\mathcal{L}_{\mathrm{adv}}^{G}
= \mathcal{L}_{\mathrm{adv}}\bigl(D(x, G_{\mathrm{cloud}}(x)),\; t_{\mathrm{real}},\; m\bigr)
$$

即：**骗 $D$，让假云在真值有云位置上被判成真**。

总损失：

$$
\mathcal{L}_G
= \underbrace{\mathcal{L}_{\mathrm{cloud}} + \lambda_T \mathcal{L}_{T}}_{\text{CompositeGeneratorLoss}}
+ \lambda_{\mathrm{adv}}\,\mathcal{L}_{\mathrm{adv}}^{G}
$$

- $\mathcal{L}_{\mathrm{cloud}}$：纬度加权 Charbonnier，并可对物理云量加极值权重 $w=1+\alpha(q/q_{\mathrm{ref}})^{\beta}$（实验常用 $\alpha=2,\beta=0.5,q_{\mathrm{ref}}=10^{-5},w_{\max}=100$）；
- $\mathcal{L}_{T}$：温度的纯纬度加权 Charbonnier（无极值权重），$\lambda_T=0.2$；
- $\lambda_{\mathrm{adv}}$：CLI 默认 $10^{-3}$，近期实验常用 **0.2**。

对抗项**不包含温度**。温度只通过 $\mathcal{L}_T$ 学习。

### 4.7 一个训练 step 的顺序

```text
x, y
  │
  ├─ y_cloud, m = (y_cloud > 0)
  │
  ├─ [更新 D]
  │     fake_y = G(x)          # no_grad
  │     ℓ_real = D(x, y_cloud)
  │     ℓ_fake = D(x, fake_cloud)
  │     L_D = 0.5 (adv(ℓ_real,1,m) + adv(ℓ_fake,0,m))
  │     if L_D > 0.15: 反传 D
  │
  └─ [更新 G]
        fake_y = G(x)
        L_adv_G = adv(D(x, fake_cloud), 1, m)
        L_rec  = cloud Charbonnier + 0.2 * temp Charbonnier
        L_G    = L_rec + λ_adv * L_adv_G
        反传 G
```

优化器：AdamW；$G$ 近期常用 $2\times10^{-5}$，$D$ 常用 $1\times10^{-5}$（$D$ 更慢，也是为了不压过 $G$）。可用 AMP。

---

## 5. 这套对抗设计实际在优化什么

用一句话：**在“真值本来有云”的体素上，让 $D$ 根据历史场判断云场真假，并逼 $G$ 在这些位置把云做得更像真值；晴空和大范围平均误差仍主要由 Charbonnier 负责。**

因此它**不是**在优化“整张图的真实感”，也**不是**在优化 CSI 本身。CSI 只在验证日志里作为诊断。

和指导老师讨论时，下面三点最容易引起误解，需要主动说清。

**（1）掩膜来自真值 $y$，不是来自 $G(x)$。**  
对抗梯度只出现在“未来真实有云”的位置。$G$ 若在真值晴空处虚报云，对抗项几乎不管，主要靠重建项压。反过来，$G$ 漏报真值有云区，会同时挨重建和对抗。这是有意把 GAN 用在稀有有云事件上，但 $D$ 的注意力图带有**未来有云位置**信息。训练时 $D$ 看见的是 $(x, y_{\mathrm{true}})$ 或 $(x, y_{\mathrm{fake}})$，掩膜却始终按 $y_{\mathrm{true}}$ 切。若担心 $D$ 变相“知道未来哪里有云”，根源在这张掩膜，而不在通道拼接本身。

**（2）$D$ 是条件的、局部的，并且按云通道分开打分。**  
它输出的是 $[B,4L,D,H,W]$ 的打分图，不是一个样本一个标量。每个 lead、每个云要素、每个气压层和每个格点各有一个 logit，再与同形状的真值有云掩膜逐点相乘。

**（3）温度与 GAN 解耦。**  
加温度目标后，$G$ 多一个输出通道，但 $D$ 的 `in_channels` 仍是 `18+4L`。温度 z-score 可为负，也不能用 `y>0` 当“有云掩膜”，所以温度必须排除在 $D$ 和 `y_cloud_mask` 之外。

---

## 6. 和常见 GAN 公式的对照

| 常见写法 | 本项目 |
| --- | --- |
| $\min_G\max_D \mathbb{E}[\log D(y)]+\mathbb{E}[\log(1-D(G(x)))]$ | 不用全局标量 $D$，用体素 logit |
| 非条件 $D(y)$ | 条件 $D(x,y_{\mathrm{cloud}})$ |
| 纯 BCE 或纯 LSGAN | 0.5 概率 MSE + 0.5 Focal |
| 全图平均 | 真值有云掩膜加权平均 |
| 每步都更新 $D$ | $\mathcal{L}_D\le 0.15$ 则跳过 $D$ |
| $G$ 只有对抗损失 | 重建为主，对抗为 $\lambda_{\mathrm{adv}}$ 加权项 |

---

## 7. 实现位置索引

| 内容 | 文件 |
| --- | --- |
| GAN 一步、掩膜、门控、$G$ 总损失 | `train_GAN_test8_a1.py` 中 `focal_loss_with_logits`、`masked_adv_loss`、`train_gan` |
| $D$ 结构与 `concat(x,y)` | `src/models/discriminator_a1.py` |
| 云/温度切开、Charbonnier、$\lambda_T$ | `src/losses/GAN_losses.py` |
| 旧版下采样 PatchGAN（未用于当前脚本） | `src/models/discriminator.py` |
| ConvGRU 生成器 | `src/models/generator_convGRU.py` |

近期一组典型命令（以仓库注释中的 GAN_test18 为例）：无预训练，80 个 GAN epoch，`lead_steps=12`，`lambda_adv=0.2`，`base_channels_d=8`，`base_channels_g=64`，极值权重 $\alpha=2,\beta=0.5$，`lambda_temp=0.2`。评估 5 通道 checkpoint 请用 `evaluation/General_eval_2.py`，不要用仍按 4 通道解释输出的旧 `General_eval.py`。
