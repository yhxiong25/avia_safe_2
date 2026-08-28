import sys
sys.path.append("/home/yx/projects/common_data/sais2025/test1/")  # 不能把src也加上
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import logging
import numpy as np
import argparse
import os
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Iterable, Optional,Union   # 这两个库，真要认真看看。
import math

from src.data.data_loader_v2 import DataLoaders,make_dataloaders
from src.data.fixed_info import INPUT_VARIABLES,CLOUD_VARIABLES,CLOUD_SHORT_NAMES,THRESHOLDS,THRESHOLDS_normalized
from src.utils.utilities import set_seed
from src.models.Atmospheric_Encoder import AtmosphericEncoder
from src.models.Physical_Encoder import PhysicalEncoder
from src.models.transformer import SwinContextModule,SwinBlock2D
from src.models.mask_decoder import MaskUpsampleDecoder
from src.losses.mask_predictor_losses import MaskPredictorLoss
from src.utils.log import setuplogging,log_args


'''
TODO: 
1,把ConvGRU那个代码弄清楚   ✅
2，把这里的残差，等网络结构的原理弄清楚
3，把Gelu，Silu这些激活函数的含义弄清楚。
4，把transformer用ide一行行运行弄清楚原理。
5，git上传代码到github上去。   git运用要更加熟悉熟练，融入到开发中去。
6，仔细研读代码，用ide一行行过数据流弄清楚原理。 在代码中注释自己不清楚的点困惑的点，上传给GPT用提示词要求详尽解释。
7，之后需要，更改这些模型的流程。   在src的源代码中定义单个模块，然后细节处理，维度变换，多个模块接续处理这些，在主程序中进行就好。 不要全部打包到模型代码文件中。 流程打包到模块代码本身就很怪。 所以之后要改。
8，要把ddp并行的代码搞清楚，学会用。
9，由于是多个模型块组成数据流，训练的时候，需要写一个函数打包这些块吗？ 构成一个总模型？ 
10，需要把AdamW搞清楚
11，对照我写的train函数和GPT写的train函数，学习
'''



@dataclass
class TrainConfig:
    zarr_path:str
    save_dir:str="./data/mask_predictor"
    batch_size:int=4
    base_channels:int=64
    max_val_samples:int=2048
    max_test_samples:Optional[int]=None
    seed:int=42
    amp:bool = True
    epochs: int=50
    lr:float=1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"   # 先在一个GPU上训练，之后再看并行
    lead_steps:int = 4

def parse_args() -> TrainConfig:
    '''
nohup python3 -u mask_predictor_v2.py --zarr_path '/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr'  --save_dir './data/mask_predictor_1/' --epochs 50  --batch_size 4 --no_amp > train1.log 2>&1 &

    '''
    p = argparse.ArgumentParser(description="Train a cloud_probability mask model for China region in multiple levels and 4 vars")
    p.add_argument("--zarr_path", type=str,default='/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr' )  # 这里应该加一个default   required=True 表示命令行必须传入这个参数。
    p.add_argument("--save_dir", type=str, default="./data/mask_predictor_1")   #  模型保存地址
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--base_channels",type=int,default=64)
    p.add_argument("--max_val_samples", type=int, default=2048)   # 验证集以后都是用全部。。 验证集是一年，最多大概有 1447个，一年有1447个样本
    p.add_argument("--max_test_samples", type=int, default=None)   # 测试集的样本数量，None表示不限制。那到底是多少个测试集样本数量？
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp",action="store_true")
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--lead_steps",type=int,default=4)   # 预报lead time，需要将其减小到4测试一下
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")    # "cuda" if torch.cuda.is_available() else "cpu"
    args = p.parse_args()
    return TrainConfig(
        zarr_path=args.zarr_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        base_channels = args.base_channels,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
        amp = not args.no_amp,
        epochs = args.epochs,
        lr = args.lr,
        lead_steps = args.lead_steps,
        device=args.device,)



def get_Mask_ic(x_ic,pressures,device)->torch.Tensor:
    '''从当前时刻(不是输入的两个时刻)获取IC index的分布。 输入数据是(B,2,13,H,W) 要素顺序： 'temperature', 'specific_humidity'  输出数据是(B,13,H,W)
    原始温度数据单位kelvin
    所有mask构造都应该在torch中完成，不应该反复numpy-tensor转化，回降低速度
    '''
    Tc = x_ic[:,0,:,:,:] - 273.15
    Qk = x_ic[:,1,:,:,:]
    es = 6.10094 * torch.exp(17.625*Tc/(Tc+243.04))
    epsilon = 0.622
    p = torch.as_tensor(pressures,dtype=x_ic.dtype,device=x_ic.device)
    fQ = 2. * (p[None,:,None,None]*Qk/(epsilon*es)-0.5)
    fT = Tc * (Tc+14.)/(-49)
    ICk = fQ*fT    # torch.Size([4, 13, 41, 71]) ✅
    return ICk      # 快速转化为可以运算的tensor？


def get_Mask_cloud(x_cloud,):
    '''从当前时刻的4个云要素的tensor，获取(B,4,13,H,W)的mask数组。  因为有云格点本身就少，异常极小值格点可以忽略，又是使用log(1+q/q0),q0=1e-8的标准化，所以直接使用标准化后的y做mask
    输入x_cloud的数组维度：  torch.Size([4, 4, 13, 41, 71])   最终还需要变换为([4,52,41,71])
    '''
    cloud_threshold = 0.0
    cloud_mask = x_cloud > cloud_threshold    # 非零值替换为True，零值替换为False。
    cloud_mask = cloud_mask.reshape(cloud_mask.shape[0],cloud_mask.shape[1]*cloud_mask.shape[2],cloud_mask.shape[3],cloud_mask.shape[4])
    return cloud_mask


def inverse_norm(x,loaders,device):  # 反标准化
    mean_all = loaders.train.dataset.mean_all
    mean_all = torch.from_numpy(mean_all).to(device)
    std_all = loaders.train.dataset.std_all
    std_all = torch.from_numpy(std_all).to(device)
    B = x.shape[0]
    H,W = x.shape[-2:]
    x = x.reshape(B,2,9,13,H,W)
    # 
    x_basic = x[:,:,[0,1,2,3,4],:,:,:]    # 如何提取前4个维度的数组
    x_cloud = x[:,:,[5,6,7,8],:,:,:]
    # broadcast
    x_mean = mean_all[[0,1,2,3,4],:][None,None,:,:,None,None]  # 
    x_std = std_all[[0,1,2,3,4],:][None,None,:,:,None,None]  
    x_basic_ = x_basic*x_std+x_mean   # 标准化后的取值
    x_cloud_ = (10.**(x_cloud) -1.) * 1e-8
    return x_basic_,x_cloud_   # 这种args.device能否作为全局参数，在函数内直接使用？   # ### `args.device` 不能直接在函数内当全局变量




class MaskPredictorBlockCombine_v1(nn.Module):
    '''需要输入Atmo_Encoder的待输入数据： [B,234,H,W], Phys_Enocder的待输入数据[B,65,H,W]。 两个block分别处理后合并，在通过SwinContextModule和MaskUpsampleDecoder两个block的接续处理得出mask预报值
    '''
    def __init__(
        self,
        base_channels_Aencoder:int=64,   # 以下是各个block的模型参数
        in_channels_Aencoder:int = 2*9*13,
        out_channels_Aencoder: int = 128,
        in_channels_Pencoder: int=65,
        base_channels_Pencoder: int=64,
        out_channels_Pencoder: int= 128,
        in_channels_Swin:int = 256,
        embed_dim_Swin:int = 128,
        num_heads_Swin:int=4,
        window_size_Swin: int=6,
        in_channels_decoder: int = 128,
        mid_channels_decoder:int=96,
        hidden_channels_decoder:int=64,
        lead_steps:int=4
    ):
        super().__init__()    # ?
        self.base_channels_Aencoder = base_channels_Aencoder
        self.in_channels_Aencoder = in_channels_Aencoder
        self.out_channels_Aencoder = out_channels_Aencoder
        self.base_channels_Pencoder = base_channels_Pencoder
        self.in_channels_Pencoder = in_channels_Pencoder
        self.out_channels_Pencoder = out_channels_Pencoder
        self.num_heads_Swin = num_heads_Swin
        self.window_size_Swin = window_size_Swin
        self.in_channels_Swin = in_channels_Swin
        self.embed_dim_Swin = embed_dim_Swin
        self.in_channels_decoder = in_channels_decoder
        self.mid_channels_decoder = mid_channels_decoder
        self.hidden_channels_decoder = hidden_channels_decoder
        self.lead_steps = lead_steps
        self.atmo_encode = AtmosphericEncoder(in_channels= self.in_channels_Aencoder,   # 输入2个时刻，9个要素，13个高度的数据
                    base_channels= self.base_channels_Aencoder,  # 默认64个第一层特征通道数
                    out_channels= self.out_channels_Aencoder)    # 输出128个通道的数据的特征
        self.phys_encode = PhysicalEncoder(in_channels = self.in_channels_Pencoder,   # 输入13+4*13 = 65个mask。  
                    base_channels = self.base_channels_Pencoder,
                    out_channels = self.out_channels_Pencoder)  # 和Atmo_encoder的输出对齐 
        self.swin_context = SwinContextModule(in_channels=self.in_channels_Swin,  # 是Atmo_encoder和Phys_encoder的输出数组的合并
                embed_dim = self.embed_dim_Swin,
                num_heads = self.num_heads_Swin,   
                window_size = self.window_size_Swin)   # 这些参数还不清楚含义，都使用默认设置
        self.upsample = MaskUpsampleDecoder(in_channels=self.in_channels_decoder,
                    mid_channels=self.mid_channels_decoder,
                    hidden_channels=self.hidden_channels_decoder,
                    lead_steps=self.lead_steps,)
    def forward(self,x_atmo:torch.Tensor,x_phys:torch.Tensor) -> torch.Tensor:
        ''''''
        x_atmo_ = self.atmo_encode(x_atmo)
        x_phys_ = self.phys_encode(x_phys)
        x_atmo_phys = torch.concat([x_atmo_,x_phys_],dim=1)
        x_swin = self.swin_context(x_atmo_phys)
        mask_logits = self.upsample(x_swin)
        return mask_logits



def _batch_to_device(batch, device: str):  # ？
    x, y, meta = batch    # meta是什么意思？   metadata？  元数据？  
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), meta



@torch.no_grad()
def validate_mask_predictor(
    model: nn.Module,
    loaders: DataLoaders,
    criterion: nn.Module,
    args: TrainConfig,
    threshold: float = 0.5,
) -> dict:
    """
    在验证集上计算 mask predictor 的平均 loss 和简单分类指标。
    threshold:
        将 sigmoid 概率转成 binary mask 的阈值。
    """
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_focal = 0.0
    n_samples = 0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for batch in loaders.val:
        x_atm, E_mask, mask_target = prepare_mask_predictor_batch(
            batch,
            loaders,
            args,)
        pred_logits = model(x_atm, E_mask)
        loss_log = criterion.compute_components(pred_logits, mask_target)
        loss = loss_log["total"]
        bs = x_atm.shape[0]
        n_samples += bs
        total_loss += loss.detach().item() * bs
        total_bce += loss_log["alpha_balanced_bce"].detach().item() * bs
        total_focal += loss_log["focal"].detach().item() * bs
        # 概率图
        pred_prob = torch.sigmoid(pred_logits)
        # binary mask
        pred_mask = pred_prob >= threshold
        true_mask = mask_target >= 0.5
        tp += (pred_mask & true_mask).sum().item()
        fp += (pred_mask & (~true_mask)).sum().item()
        fn += ((~pred_mask) & true_mask).sum().item()
    eps = 1e-8
    csi = tp / (tp + fp + fn + eps)
    pod = tp / (tp + fn + eps)
    far = fp / (tp + fp + eps)
    return {
        "loss": total_loss / max(n_samples, 1),
        "bce": total_bce / max(n_samples, 1),
        "focal": total_focal / max(n_samples, 1),
        "csi": csi,
        "pod": pod,
        "far": far,}     # CSI[lead, variable, level] CSI这些指数可以扩展为。


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    args: TrainConfig,
) -> None:
    """
    保存最佳模型参数和训练状态。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "mask_predictor": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": asdict(args),}
    torch.save(ckpt, path)

def prepare_mask_predictor_batch(batch,loaders,args):
    """
    将普通 DataLoader 输出的 batch 转换成 Mask Predictor 需要的数据。
    输入 batch:
        x: [B, 18, 13, H, W]
        y: [B, lead_steps*4, 13, H, W]
    输出:
        x_atm:       [B, 234, H, W]
        E_mask:      [B, 65, H, W]
        mask_target: [B, lead_steps*4*13, H, W]
    """    
    x,y,_ = _batch_to_device(batch,args.device)
    B, C, D, H, W = x.shape
    x_basic,x_cloud = inverse_norm(x,loaders,args.device)
    mask_ic = get_Mask_ic(x_basic[:,-1,[1,2],:,:,:],loaders.train.dataset.pressure_levels,args.device)
    mask_cloud = get_Mask_cloud(x_cloud[:,-1,:,:,:,:])
    E_mask = torch.concat((mask_cloud,mask_ic),dim=1).to(args.device)    # torch.Size([4, 65, 41, 71])
    x_atm = x.reshape(B,C*D,H,W)  # 变换维度
    mask_target = get_Mask_cloud(y.view(B,args.lead_steps*4,D,H,W))   # 获取target
    mask_target = mask_target.to(args.device)
    ## x_atm维度：torch.Size([4, 234, 41, 71])  234 = 2*9*13 ; E_mask::torch.Size([4, 65, 41, 71])  ; mask_target : torch.Size([4, 208, 41, 71])  208 = 4*4*13
    return x_atm.float(),E_mask.float(),mask_target.float()    



def train_mask_predictor(
    model:nn.Module,
    loaders:DataLoaders,
    criterion: nn.Module,     # ? 这里是标注参数类型
    args:TrainConfig,
)->float:
    '''
    '''
    device = args.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)   # 优化器是否有更合适选择？ 
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    best_val = math.inf
    save_dir = Path(args.save_dir)
    for epoch in range(1,args.epochs+1):
        t0 = time.time()
        model.train()
        for batch in loaders.train:
            # # 需要数据处理
            x_atm,E_mask,mask_target = prepare_mask_predictor_batch(batch,loaders,args)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=args.amp):
                pred_logits = model(x_atm,E_mask)
                loss_log = criterion.compute_components(pred_logits,mask_target)
                loss = loss_log['total']
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        val_metrics = validate_mask_predictor(model,loaders.val,criterion,args)
        t1 = time.time()
        logging.info(
            f"[Train {epoch:03d}/{args.epochs}] "
            f"{t1-t0:.2f} sec "
            f"val_loss={val_metrics['loss']:.6f} ")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(save_dir / "best_mask_predictor.pt",
            model,)
            logging.info(f"  -> saved best mask_predictor, val_csi_norm0={val_metrics['csi_norm0']:.4f}")    # 这里验证集，还是需要计算CSI指数。综合评估mask的预报精度





# GPT写的新版
def train_mask_predictor(
    model: nn.Module,
    loaders: DataLoaders,
    criterion: nn.Module,
    args: TrainConfig,
) -> float:
    """
    训练 Mask Predictor。
    使用验证集 loss 保存最佳模型。
    """
    device = args.device
    model.to(device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # 是否启用 AMP：只有 CUDA 设备才启用
    use_amp = args.amp and str(device).startswith("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,)
    best_val = math.inf
    bad_epochs = 0
    patience = 10
    min_delta = 1e-4   # ？
    best_path = save_dir / "best_mask_predictor.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_bce_sum = 0.0
        train_focal_sum = 0.0
        n_samples = 0
        for batch in loaders.train:
            x_atm, E_mask, mask_target = prepare_mask_predictor_batch(
                batch,
                loaders,
                args,)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_logits = model(x_atm, E_mask)
                loss_log = criterion.compute_components(
                    pred_logits,
                    mask_target,)
                loss = loss_log["total"]
            scaler.scale(loss).backward()
            # 防止梯度爆炸，尤其 Swin/Transformer 模块建议加
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,)
            scaler.step(optimizer)
            scaler.update()
            bs = x_atm.shape[0]
            n_samples += bs
            train_loss_sum += loss.detach().item() * bs
            train_bce_sum += loss_log["alpha_balanced_bce"].detach().item() * bs
            train_focal_sum += loss_log["focal"].detach().item() * bs
        train_loss = train_loss_sum / max(n_samples, 1)
        train_bce = train_bce_sum / max(n_samples, 1)
        train_focal = train_focal_sum / max(n_samples, 1)
        val_metrics = validate_mask_predictor(
            model=model,
            loaders=loaders,
            criterion=criterion,
            args=args,
            threshold=0.5,)
        t1 = time.time()
        logging.info(
            f"[MaskTrain {epoch:03d}/{args.epochs}] "
            f"{t1 - t0:.2f} sec | "
            f"train_loss={train_loss:.6f} "
            f"train_bce={train_bce:.6f} "
            f"train_focal={train_focal:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_csi={val_metrics['csi']:.4f} "
            f"val_pod={val_metrics['pod']:.4f} "
            f"val_far={val_metrics['far']:.4f}")
        # --------------------------------------------------
        # 保存验证集 loss 最优模型
        # --------------------------------------------------
        if val_metrics["loss"] < best_val - min_delta:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val,
                args=args,)
            logging.info(
                f"  -> saved best mask predictor: "
                f"epoch={epoch}, best_val_loss={best_val:.6f}")
        else:
            bad_epochs += 1
        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------
        if bad_epochs >= patience:
            logging.info(
                f"Early stopping at epoch {epoch}. "
                f"Best val_loss={best_val:.6f}")
            break
    return best_val
                




if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    setuplogging(args.save_dir+"0.log")
    log_args(args)
    loaders = make_dataloaders(args.zarr_path,
                batch_size = args.batch_size,
                max_val_samples=args.max_val_samples,
                max_test_samples=args.max_test_samples,
                forecast_steps=args.lead_steps,
                seed = args.seed)
    # 引入各个模型：
    model = MaskPredictorBlockCombine_v1(base_channels_Aencoder=args.base_channels,
                base_channels_Pencoder=args.base_channels,
                lead_steps = args.lead_steps)

    # loaders_unnorm = make_dataloaders(zarr_path=args.zarr_path,  # 没有进行标准化的loaders
    #     batch_size = args.batch_size,
    #     num_workers = 0,
    #     max_val_samples = args.max_val_samples,
    #     max_test_samples = args.max_test_samples,
    #     seed = args.seed,
    #     forecast_steps = args.forecast_steps,
    #     normalize = False)   # 返回无标准化的loaders

    pressures = loaders.train.dataset.pressure_levels

    criterion = MaskPredictorLoss(alpha_bce_pos=0.85,
            focal_alpha=0.25,
            focal_gamma=1.5,
            lambda_bce=1.,
            lambda_focal=1.).to(args.device)

    best_val = train_mask_predictor(model,loaders,criterion,args)
    logging.info(f"Training finished. Best val_loss={best_val:.6f}")
    














