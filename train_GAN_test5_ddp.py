"""
----
Input  : historical 2 time steps, 9 variables, 13 pressure levels, 41 x 71 grid
         x_raw shape = [2, 9, 13, 41, 71]
Target : future 12 time steps, 4 cloud variables, 13 pressure levels, 41 x 71 grid
         y_raw shape = [12, 4, 13, 41, 71]

For the 3D U-Net generator, time and variable are folded into channels while pressure level
is treated as the 3D depth dimension:
    x = [2*9, 13, 41, 71]   = [18, 13, 41, 71]
    y = [12*4, 13, 41, 71]  = [48, 13, 41, 71]

The discriminator is a conditional 3D PatchGAN:
    D(concat(x, y)) -> patch logits

Data split
----------
Train: windows fully within 2018-01-01 00UTC to 2020-12-31 18UTC
Val  : windows fully within 2021-01-01 00UTC to 2021-12-31 18UTC
Test : windows fully within 2022-01-01 00UTC to 2022-12-31 18UTC

Typical usage
-------------
python era5_cloud_gan.py \
  --zarr_path /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr \
  --save_dir ./checkpoints_cloud_gan \
  --batch_size 4 \
  --pretrain_epochs 30 \
  --gan_epochs 50 \
  --max_val_samples 2048       #？ 一年的最大样本数量？ 2048？    # 365 * 4 = 1460   真实一年大概 1447 个样本左右。   2048只是一个安全参数


TODO
1，DDP并行式训练好复杂， 一头雾水，应该多去读一些教程，把这个弄懂

"""

from __future__ import annotations

# 为什么没用到logging？
import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional,Union   # 这两个库，真要认真看看。
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


import xarray as xr  # 没必要使用try 和 except   。   


# 引入自定义的各个函数
from src.data.data_loader_v2 import DataLoaders,make_dataloaders
from src.losses.GAN_losses import LatitudeWeightedCharbonnierLoss,\
    LatitudeWeightedSoftMRLoss_500,\
    LatitudeWeightedSoftFARLoss_500,\
    CompositeGeneratorLoss_v2
from src.metrics.metrics import latitude_weighted_rmse,csi_score
from src.models.generator import UNet3DGenerator
from src.models.discriminator import PatchGAN3DDiscriminator
from src.utils.log import setuplogging,log_args
from src.utils.loss_epoch_plot import plot_loss_curves
from src.data.fixed_info import INPUT_VARIABLES,CLOUD_VARIABLES,CLOUD_SHORT_NAMES,THRESHOLDS,THRESHOLDS_normalized



def parse_args() -> TrainConfig:    # 这里的  ->什么含义？   
    '''
nohup python3 -u train_GAN_test5.py --save_dir './logs/GAN_test9_NewlyNormalized_debug/' --gan_epochs 50  --pretrain_epochs 20  --lr_g 2e-4  --lr_d 2e-5 --lambda_mr 0. --lambda_far 0.  --batch_size 4  --num_workers 0 --real_label_value 0.8  --fake_label_value 0.1  --no_amp> train4.log 2>&1 &    

DDP运行：
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup torchrun --standalone --nproc_per_node=gpu train_GAN_test5_ddp.py \
  --save_dir './logs/GAN_test_ddp/' \
  --gan_epochs 50 \
  --pretrain_epochs 20 \
  --lambda_adv 0.005 \
  --lr_d 2e-5 \
  --lambda_mr 0.3 \
  --lambda_far 0.3 \
  --batch_size 4 \
  --num_workers 0 \
  > train_ddp.log 2>&1 &
    ''' 
    p = argparse.ArgumentParser(description="Train a conditional GAN for ERA5 China cloud forecasting.")
    p.add_argument("--zarr_path", type=str,default='/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr' )  # 这里应该加一个default   required=True 表示命令行必须传入这个参数。
    p.add_argument("--save_dir", type=str, default="./logs/checkpoints_cloud_gan")   # 
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)    #不再有 4 个 worker 子进程，也就绕开了多进程读 Zarr/xarray 的锁和进程通信问题，所以可以正常取 batch。
    p.add_argument("--max_val_samples", type=int, default=2048)   # 验证集以后都是用全部。。 验证集是一年，最多大概有 1447个，一年有1447个样本
    p.add_argument("--max_test_samples", type=int, default=None)   # 测试集的样本数量，None表示不限制。那到底是多少个测试集样本数量？
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--gan_epochs", type=int, default=50)
    p.add_argument("--lambda_mr",type=float,default=1.0)  # miss rate这个损失函数的权重
    p.add_argument("--lambda_far",type=float,default=1.0)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--lambda_adv", type=float, default=1e-3)
    p.add_argument("--base_channels_g", type=int, default=32)
    p.add_argument("--base_channels_d", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")    # "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--no_amp", action="store_true")    # 采用no_amp     action='store_true':当命令行中出现该参数时，会将对应变量设为 True；未出现时则为 False
    p.add_argument("--real_label_value",type=float,default=1.0)   #
    p.add_argument("--fake_label_value",type=float,default=0.)  # 平滑标签，可以增大d_adv_loss ?  
    args = p.parse_args()
    return TrainConfig(
        zarr_path=args.zarr_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        pretrain_epochs=args.pretrain_epochs,
        gan_epochs=args.gan_epochs,
        lambda_mr = args.lambda_mr,
        lambda_far = args.lambda_far,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_adv=args.lambda_adv,
        base_channels_g=args.base_channels_g,    # 这也需要自定义？  都是默认32，不要去定义这个
        base_channels_d=args.base_channels_d,   # 就是定义卷积核的数量，第一次卷积操作的卷积核数量，之后卷积的数量在依次翻倍
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,   #？默认amp 为True的bool值
        real_label_value = args.real_label_value,
        fake_label_value = args.fake_label_value
    )



@dataclass     
class TrainConfig:   ## 单纯定义参数和参数类型？  不能在命令行中运行文件时，在通过args自定义一些参数？   
    '''是不是在这里定义各种超参数不用在linux命令行运行程序也可以？   那在命令行中运行这个程序，带上超参数，谁的优先级更高呢？   这里训练模式，有输出各种训练超参数的设置吗？
TrainConfig 是一个 dataclass，用来集中保存训练超参数。
    命令行参数优先。‘

    如果你不想每次在命令行写，也可以直接修改 TrainConfig 默认值或 parse_args() 的 default。
    '''
    zarr_path: str     #  /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr/
    save_dir: str = "./logs/checkpoints_cloud_gan"    # /home/yx/projects/common_data/sais2025/test1/logs/某个test文件夹/    需要在正式存储内容之前，检查一下文件夹名字是否存在，并创建一下。
    batch_size: int = 4
    num_workers: int = 0   # 什么意思？  子进程？
    max_val_samples: int = 2048   # >>> len(loaders.val.dataset)   1447..    
    max_test_samples: Optional[int] = None  # 这里表示这个变量可以是None，也可以是int吗？ 
    pretrain_epochs: int = 30  # 与训练的轮次
    lambda_mr: float = 1.0
    lambda_far: float = 1.0
    gan_epochs: int = 50   #正式训练的轮次
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5   # 
    beta2: float = 0.999 #   adam优化器的动量相关的参数
    lambda_adv: float = 1e-3
    base_channels_g: int = 32  # generator的卷积核数量
    base_channels_d: int = 32  # 
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"   # 先在一个GPU上训练，之后再看并行
    amp: bool = True  # 什么意思？ 自动混合精度训练（Automatic Mixed Precision） ？它会自动把部分运算用 float16 / bfloat16 执行，以节省显存并加速。
    real_label_value:float=1.
    fake_label_value:float = 0.
# `18个变量，之后的cfg也就这18个变量`






# =========================
# DDP helper functions
# =========================

def ddp_is_initialized() -> bool:
    """Return True if torch.distributed has been initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(cfg: TrainConfig):
    """Initialize DDP from torchrun environment variables.

    torchrun will provide RANK, LOCAL_RANK, WORLD_SIZE.
    In single-GPU/single-process runs these variables do not exist, so this
    function leaves the original cfg.device unchanged.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training here expects CUDA GPUs, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        cfg.device = f"cuda:{local_rank}"
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if ddp_is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if ddp_is_initialized():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when using DistributedDataParallel."""
    return model.module if isinstance(model, DDP) else model


def rebuild_train_loader_for_ddp(loaders: DataLoaders, cfg: TrainConfig) -> None:
    """Replace only the training loader with a DistributedSampler loader.

    Validation/test loaders are kept unchanged because only rank 0 evaluates.
    """
    if not ddp_is_initialized():
        return
    sampler = DistributedSampler(
        loaders.train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=cfg.seed,
        drop_last=True,
    )
    loaders.train = DataLoader(
        loaders.train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )

def save_checkpoint(path: Union[str, Path], generator: nn.Module, discriminator: Optional[nn.Module], extra: dict) -> None:
    '''discriminator: Optional[nn.Module] 表示discriminator可以是nn.Module类型， 也可以是其它任何类型。
    '''
    ckpt = {
        "generator": unwrap_model(generator).state_dict(),
        "extra": extra,
    }
    if discriminator is not None:   # 按照字典保存各个模型的   .state_dict()具体是怎样的呢？ 需要自己在ide里面试试
        ckpt["discriminator"] = unwrap_model(discriminator).state_dict()
    torch.save(ckpt, path)   # 保存模型。  torch保存字典？     把模型的参数保存到ckpt字典中，然后保存到path路径中。就是.pt文件？？


def _batch_to_device(batch, device: str):  # ？
    x, y, meta = batch    # meta是什么意思？   metadata？  元数据？  
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), meta


@torch.no_grad()
def validate_generator(
    generator: nn.Module,
    loader: DataLoader,
    criterion: CompositeGeneratorLoss,
    latitudes: np.ndarray,
    device: str,
) -> dict:
    '''具体是在干嘛呢？  返回某一个批次，generator的指标。。。    用于用验证集评价generator效果，以保存最佳模型。
    '''
    generator.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_csi = 0.0
    n = 0
    for batch in loader:   # 为了计算所有数样本的误差平均。  # 这里应该不能迭代2048次，样本数最多1700个
        '''这里值迭代了362次？  为什么只迭代了362次呢？   val总共有1447个样本，这里batch_size = 4。  '''
        x, y, _ = _batch_to_device(batch, device)
        pred = generator(x)
        loss = criterion(pred, y)
        rmse = latitude_weighted_rmse(pred, y, latitudes)  ## 
        csi = csi_score(pred, y, threshold=0.0)    #csi需要设置一个阈值。
        bs = x.shape[0]
        total_loss += loss.item() * bs    # ？
        total_rmse += rmse.item() * bs
        total_csi += csi.item() * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),   # 应该用这个综合损失函数，作为评判训练效果的指标
        "rmse": total_rmse / max(n, 1),    # 这里的rmse，没考虑纬度平均的。
        "csi_norm0": total_csi / max(n, 1),
    }



def pretrain_generator(
    generator: nn.Module,
    loaders: DataLoaders,
    criterion: LatitudeWeightedCharbonnierLoss,
    cfg: TrainConfig,
) -> float:
    """Pretrain generator with supervised composite loss.

    DDP note:
        Every rank trains on its own DistributedSampler shard.
        Only rank 0 runs validation, checkpoint saving and loss plotting.
    """
    device = cfg.device
    generator.to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_g, betas=(0.9, 0.95), weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    best_val = math.inf
    save_dir = Path(cfg.save_dir)
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info('q0改成1e-8次方试试')
    loss_log_ = {}

    for epoch in range(1, cfg.pretrain_epochs + 1):
        if hasattr(loaders.train, "sampler") and hasattr(loaders.train.sampler, "set_epoch"):
            loaders.train.sampler.set_epoch(epoch)

        t0 = time.time()
        generator.train()
        average_loss = {'total_loss': [], 'weighted_recon': [], 'weighted_far': [], 'weighted_mr': []}
        seen = 0

        for batch in loaders.train:
            x, y, _ = _batch_to_device(batch, device)
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                pred = generator(x)
                loss_log = criterion.compute_components(pred, y)
                loss = loss_log["total"]

            bs = x.shape[0]
            average_loss['total_loss'].append(loss.detach().item() * bs)
            average_loss['weighted_recon'].append(loss_log['recon'].detach().item() * bs)
            average_loss['weighted_mr'].append(loss_log['weighted_mr'].detach().item() * bs)
            average_loss['weighted_far'].append(loss_log['weighted_far'].detach().item() * bs)

            scaler.scale(loss).backward()
            scaler.step(opt_g)
            scaler.update()
            seen += bs

        if is_main_process():
            # 用 unwrap_model 避免 rank 0 单独验证 DDP 包装模型时触发分布式 buffer 同步。
            val_metrics = validate_generator(unwrap_model(generator), loaders.val, criterion, loaders.val_dataset.latitudes, device)
            average_loss = {key: sum(val) / max(seen, 1) for key, val in average_loss.items()}
            loss_log_[epoch] = {
                'total_loss_train': average_loss['total_loss'],
                'weighted_recon_train': average_loss['weighted_recon'],
                'weighted_mr_train': average_loss['weighted_mr'],
                'weighted_far_train': average_loss['weighted_far'],
                'total_loss_val': val_metrics['loss'],
            }
            t1 = time.time()
            logging.info(
                f"[Pretrain {epoch:03d}/{cfg.pretrain_epochs}] "
                f"{t1-t0:.2f} sec train_loss={loss_log_[epoch]['total_loss_train']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_rmse={val_metrics['rmse']:.6f} "
                f"val_csi_norm0={val_metrics['csi_norm0']:.4f}"
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(
                    save_dir / "best_pretrain_generator.pt",
                    generator,
                    None,
                    {"epoch": epoch, "val_metrics": val_metrics, "stage": "pretrain"},
                )
                logging.info(f"  -> saved best pretrain generator, val_csi_norm0={val_metrics['csi_norm0']:.4f}")

        # 确保非主进程等待 rank 0 完成验证与保存后再进入下一轮。
        barrier()

    if is_main_process():
        plot_loss_curves(loss_log_, 'pretrain', save_dir / "pretrain_loss_curves.png")

    barrier()
    return best_val


@torch.no_grad()
def evaluate_test(generator: nn.Module, loaders: DataLoaders, criterion: LatitudeWeightedCharbonnierLoss, cfg: TrainConfig) -> dict:
    '''
    '''
    generator.to(cfg.device)
    metrics = validate_generator(generator, loaders.test, criterion, loaders.test_dataset.latitudes, cfg.device)
    print("[Test]", json.dumps(metrics, indent=2))    # json.dumps(metrics, indent=2) 是什么意思？  把metrics转换为json格式，并缩进2个空格。
    return metrics


def load_generator_checkpoint(generator: nn.Module, path: Union[str, Path], device: str) -> dict:
    '''
    '''
    ckpt = torch.load(path, map_location=device, weights_only=True)  #
    state_dict = ckpt["generator"]
    # 兼容 DataParallel/DDP 保存的 module. 前缀；新保存的权重不会带这个前缀。
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    unwrap_model(generator).load_state_dict(state_dict)
    return ckpt.get("extra", {})


def set_seed(seed: int = 42) -> None:   # 设置随机种子
    np.random.seed(seed)    # 这三个随机种子，各自具体什么作用和含义？
    torch.manual_seed(seed)   # torch.manual_seed(seed) is used in PyTorch to set the seed for generating random numbers on all devices (CPU and GPU), ensuring reproducibility of experiments. It returns a torch.Generator object that can be reused for controlled randomness.  确保每次生成的随机数是一样的。
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子 为所有可用的 GPU 设置随机数生成器的种子，以确保多 GPU 环境下的结果可复现。
    torch.backends.cudnn.benchmark = True   # 用于启用 cuDNN 的自动优化功能。它会根据硬件和输入数据的大小，选择最优的卷积算法，从而提升模型的运行效率。；一般场景下，只要简单地在 PyTorch 程序开头将其值设置为 True，就可以大大提升卷积神经网络的运行速度。；适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
# 把所有随机性设置随机种子


###########  以上多个函数，是辅助train_gan用的



def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    loaders: DataLoaders,
    criterion: LatitudeWeightedCharbonnierLoss,
    cfg: TrainConfig,
) -> float:
    """Adversarial training stage.

    DDP note:
        Every rank updates G and D on a different shard of the training data.
        Only rank 0 validates, saves checkpoints and plots loss curves.
    """
    device = cfg.device
    generator.to(device)
    discriminator.to(device)
    opt_g = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2), weight_decay=1e-4)
    adv_criterion = nn.BCEWithLogitsLoss()
    scaler_g = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    best_val = math.inf
    save_dir = Path(cfg.save_dir)
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)
    loss_log_ = {}

    for epoch in range(1, cfg.gan_epochs + 1):
        if hasattr(loaders.train, "sampler") and hasattr(loaders.train.sampler, "set_epoch"):
            loaders.train.sampler.set_epoch(epoch)

        t0 = time.time()
        generator.train()
        discriminator.train()
        average_loss = {
            'total_loss': [], 'weighted_recon': [], 'weighted_far': [], 'weighted_mr': [],
            'discriminator_loss': [], 'adv_loss_g': []
        }
        seen = 0

        for batch in loaders.train:
            x, y, _ = _batch_to_device(batch, device)
            bs = x.shape[0]

            # -----------------------
            # 1. Update discriminator
            # -----------------------
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_y = generator(x)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                real_logits = discriminator(x, y)
                fake_logits = discriminator(x, fake_y.detach())
                real_labels = torch.ones_like(real_logits) * cfg.real_label_value
                fake_labels = torch.zeros_like(fake_logits) + cfg.fake_label_value
                d_loss_real = adv_criterion(real_logits, real_labels)
                d_loss_fake = adv_criterion(fake_logits, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
            average_loss['discriminator_loss'].append(d_loss.detach().item() * bs)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ------------------
            # 2. Update generator
            # ------------------
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                fake_y = generator(x)
                fake_logits = discriminator(x, fake_y)
                adv_loss_g = adv_criterion(fake_logits, torch.ones_like(fake_logits))
                loss_log = criterion.compute_components(fake_y, y)
                g_loss = loss_log['total'] + cfg.lambda_adv * adv_loss_g

            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            average_loss['total_loss'].append(g_loss.detach().item() * bs)
            average_loss['weighted_recon'].append(loss_log['recon'].detach().item() * bs)
            average_loss['weighted_mr'].append(loss_log['weighted_mr'].detach().item() * bs)
            average_loss['weighted_far'].append(loss_log['weighted_far'].detach().item() * bs)
            average_loss['adv_loss_g'].append(adv_loss_g.detach().item() * bs)
            seen += bs

        if is_main_process():
            val_metrics = validate_generator(unwrap_model(generator), loaders.val, criterion, loaders.val_dataset.latitudes, device)
            average_loss = {key: sum(val) / max(seen, 1) for key, val in average_loss.items()}
            loss_log_[epoch] = {
                'discriminator_loss': average_loss['discriminator_loss'],
                'total_loss_train': average_loss['total_loss'],
                'weighted_recon_train': average_loss['weighted_recon'],
                'weighted_mr_train': average_loss['weighted_mr'],
                'weighted_far_train': average_loss['weighted_far'],
                'total_loss_val': val_metrics['loss'],
                'adv_loss_g': average_loss['adv_loss_g'],
            }

            t1 = time.time()
            logging.info(
                f"[GAN {epoch:03d}/{cfg.gan_epochs}] "
                f"{t1-t0:.2f} sec "
                f"D={loss_log_[epoch]['discriminator_loss']:.6f} "
                f"G_adv_raw={loss_log_[epoch]['adv_loss_g']:.6f} "
                f"Recon={loss_log_[epoch]['weighted_recon_train']:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_csi_norm0={val_metrics['csi_norm0']:.4f}"
            )
            if val_metrics['loss'] < best_val:
                best_val = val_metrics['loss']
                save_checkpoint(
                    save_dir / "best_gan.pt",
                    generator,
                    discriminator,
                    {"epoch": epoch, "val_metrics": val_metrics, "stage": "gan"},
                )
                logging.info(f"  -> saved best GAN checkpoint, val_csi_norm0={val_metrics['csi_norm0']:.4f}")

        barrier()

    if is_main_process():
        logging.info('train_gan结束')
        np.save(os.path.join(cfg.save_dir, 'loss_log_.npy'), loss_log_)
        plot_loss_curves(loss_log_, "gan_train", save_dir / "gan_train_loss_curve.png")

    barrier()
    return best_val


##
def main() -> None:
    cfg = parse_args()
    rank, local_rank, world_size = setup_distributed(cfg)
    try:
        set_seed(cfg.seed)
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        if is_main_process():
            setuplogging(cfg.save_dir + '0.log')
            log_args(cfg)
            logging.info(f'DDP world_size={world_size}, local_rank={local_rank}, rank={rank}')
            logging.info(f'使用设备{cfg.device}')
            logging.info('加载data loaders')
        else:
            # 非主进程不写同一个日志文件，避免多进程争抢 0.log。
            logging.basicConfig(level=logging.ERROR)

        loaders = make_dataloaders(
            cfg.zarr_path,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            max_val_samples=cfg.max_val_samples,
            max_test_samples=cfg.max_test_samples,
            seed=cfg.seed,
            target_variables=CLOUD_VARIABLES,
        )
        rebuild_train_loader_for_ddp(loaders, cfg)

        if is_main_process():
            print(f"[INFO] train samples: {len(loaders.train_dataset)}")
            print(f"[INFO] val samples:   {len(loaders.val_dataset)}")
            print(f"[INFO] test samples:  {len(loaders.test_dataset)}")

            # One batch shape check only on rank 0.
            xb, yb, meta = next(iter(loaders.train))
            print(f"[INFO] x batch shape: {tuple(xb.shape)}  expected [B,18,13,41,71]")
            print(f"[INFO] y batch shape: {tuple(yb.shape)}  expected [B,48,13,41,71]")
            print(f"[INFO] first sample meta: {meta}")
            logging.info("本次训练的配置细节信息： 改进了损失函数，FARloss和MRloss，使用250hPa以下的所有高度(共9个)，修改了的权重超参数，尝试规避discriminator过强的GAN训练不均衡问题。")

        generator = UNet3DGenerator(in_channels=18, out_channels=48, base_channels=cfg.base_channels_g).to(cfg.device)
        discriminator = PatchGAN3DDiscriminator(in_channels=18 + 48, base_channels=cfg.base_channels_d).to(cfg.device)

        criterion = CompositeGeneratorLoss_v2(
            latitudes=loaders.train_dataset.latitudes,
            pressure_levels=loaders.train_dataset.pressure_levels,
            thresholds_normalized=THRESHOLDS_normalized,
            levels_hpa=(1000, 925, 850, 700, 600, 500, 400, 300, 250),
            lambda_mr=cfg.lambda_mr,
            lambda_far=cfg.lambda_far,
            k=2.0,
        ).to(cfg.device)

        # DDP wrapping must happen after .to(device) and before optimizer creation.
        if ddp_is_initialized():
            generator = DDP(generator, device_ids=[local_rank], output_device=local_rank)
            discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)

        if is_main_process():
            logging.info("Pretraining generator...")
        pretrain_generator(generator, loaders, criterion, cfg)

        pretrain_path = Path(cfg.save_dir) / "best_pretrain_generator.pt"
        barrier()     # 到底什么含义？
        if pretrain_path.exists():
            extra = load_generator_checkpoint(generator, pretrain_path, cfg.device)
            if is_main_process():
                logging.info(f"[INFO] Loaded best pretrained generator: {extra}")
        barrier()

        if is_main_process():
            logging.info("正式 Training GAN...")
        train_gan(generator, discriminator, loaders, criterion, cfg)

        best_gan_path = Path(cfg.save_dir) / "best_gan.pt"
        barrier()
        if is_main_process() and best_gan_path.exists():
            extra = load_generator_checkpoint(generator, best_gan_path, cfg.device)
            logging.info(f"Loaded best GAN generator: {extra}")
            metrics = evaluate_test(unwrap_model(generator), loaders, criterion, cfg)
            with open(Path(cfg.save_dir) / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
