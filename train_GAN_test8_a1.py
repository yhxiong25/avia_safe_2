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



改进对抗损失函数生成的逻辑。
直接调用y的有云区域生成mask，然后计算有云区域的adv loss。
advloss 不在是单独的MSE或者是BCEWithLogits， 这里选用： 0.5 * MSE + 0.5 * focal loss
使用pixel GAN，而不是patch GAN，改写discriminator的结构，使用kernel size 为3，stride为1，padding为1，输出通道数为1，这样生成pixel logits，这样每个像素判别，而不是patch判别。
直接一刀切式生成adv loss，完全忽略无云区域，只计算有云区域的adv loss。

discriminator是结合X，Y一起判断，还是应该仅根据Y判断？      ---直接根据Y判断
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
from src.losses.GAN_losses import CompositeGeneratorLoss, split_cloud_temp
from src.metrics.metrics import latitude_weighted_rmse,csi_score
from src.models.generator import UNet3DGenerator
from src.models.generator_convGRU import ConvGRUUNet3DGenerator
from src.models.generator_simVP import SimVP3DGenerator
from src.models.generator_SwinTransformer import SwinTransformer3DGenerator 
from src.models.discriminator_a1 import VoxelLocalDiscriminator
from src.utils.log import setuplogging,log_args
from src.utils.loss_epoch_plot import plot_loss_curves
from src.data.fixed_info import INPUT_VARIABLES,CLOUD_VARIABLES,CLOUD_SHORT_NAMES


'''

# 尝试：修改discriminator，原输入为x+y现在只输入y？；
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup torchrun --standalone --nproc_per_node=gpu train_GAN_test7_ConvGRU.py \
  --save_dir './logs/GAN_test11_ConvGRU_5/' \
  --gan_epochs 40 \
  --pretrain_epochs 30 \
  --lambda_adv 0.05 \
  --lr_g 2e-5  \
  --lr_d 1e-5 \
  --base_channels_d 4 \
  --batch_size 8 \
  --num_workers 0 \
  --real_label_value 0.8 \
  --fake_label_value 0.2 \
  --lead_steps 4 \
  --generator_version 'convgru' \
  > train_4.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup torchrun --standalone --nproc_per_node=gpu train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test12_a1/' \
  --gan_epochs 50 \
  --pretrain_epochs 30 \
  --lambda_adv 0.1 \
  --lr_g 2e-5  \
  --lr_d 1e-5 \
  --base_channels_d 8 \
  --batch_size 8 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lead_steps 4 \
  --generator_version 'convgru' \
  > train_4.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup torchrun --standalone --nproc_per_node=gpu train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test12_a1_/' \
  --gan_epochs 300 \
  --pretrain_epochs 30 \
  --lambda_adv 0.15 \
  --lr_g 2e-5  \
  --lr_d 1e-5 \
  --base_channels_d 8 \
  --batch_size 8 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lead_steps 4 \
  --generator_version 'convgru' \
  > train_4.log 2>&1 &


# 9月6日，简化，GAN loss中只用Charbonnier，保存每一个epoch的model，
CUDA_VISIBLE_DEVICES=0,1，2，3，4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test12_a11/' \
  --gan_epochs 70 \
  --pretrain_epochs 30 \
  --lambda_adv 0.15 \
  --lr_g 2e-5  \
  --lr_d 1e-5 \
  --base_channels_d 8 \
  --base_channels_g 64 \
  --batch_size 4 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lead_steps 4 \
  --generator_version 'convgru' \
  > train_4.log 2>&1 &

# 9月10日，缩小lead time = 1重新实验一下看效果
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test13_a1_lead1/' \
  --gan_epochs 70 \
  --pretrain_epochs 30 \
  --lambda_adv 0.15 \
  --lr_g 2e-5  \
  --lr_d 1e-5 \
  --base_channels_d 8 \
  --base_channels_g 64 \
  --batch_size 16 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lead_steps 1 \
  --generator_version 'convgru' \
  > train_4.log 2>&1 &

# 
# 9月12日，训练时间延长到250个epoch，看看效果
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test13_a1_epochs250/' \
  --gan_epochs 250 \
  --pretrain_epochs 30 \
  --lambda_adv 0.2 \
  --lr_g 2e-5  \
  --lr_d 1e-5 \
  --base_channels_d 8 \
  --base_channels_g 64 \
  --batch_size 16 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lead_steps 1 \
  --generator_version 'convgru' \
  > train_4.log 2>&1 &


# 9月18日，连续极值加权 Charbonnier：w(y)=1+α(y/y_ref)^β
# y 在物理量 kg/kg 上加权，y_ref=1e-5 约 70%-80% 分位。alpha=0 退回原 Charbonnier。
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test14_extreme_weight_epoch250/' \
  --gan_epochs 250 \
  --pretrain_epochs 30 \
  --lambda_adv 0.2 \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --base_channels_d 8 \
  --base_channels_g 64 \
  --batch_size 64 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lead_steps 1 \
  --generator_version 'convgru' \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --train_message '连续极值加权 Charbonnier: alpha=2 beta=0.5 y_ref=1e-5 w_max=100；其余设置对齐 GAN_test13_a1_epochs250' \
  > train_extreme_weight.log 2>&1 &

# 训练结束后评估 train/test CSI@q90，并与 GAN_test13 对比 400 hPa lead1
python evaluation/General_eval.py \
  --checkpoint_dir './logs/GAN_test14_extreme_weight/check_points' \
  --splits train,test \
  --percentiles 90 \
  --skip_cases
python evaluation/compare_csi_q90.py

# 怎么完全评估？ 


# 9月23日，直接无pretrain训练GAN，80个epoch。 
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test15_nopretrain_epoch80/' \
  --gan_epochs 80 \
  --pretrain_epochs 0 \
  --lambda_adv 0.2 \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --base_channels_d 8 \
  --base_channels_g 64 \
  --batch_size 64 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lead_steps 1 \
  --generator_version 'convgru' \
  --save_every_epoch \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --train_message '无pretrain训练GAN，80个epoch。 其余设置对齐 GAN_test13_a1_epochs250' \
  > train_nopretrain.log 2>&1 &
# 评估
python evaluation/General_eval.py \
  --checkpoint_dir './logs/GAN_test15_nopretrain_epoch80/check_points' \
  --splits train,val,test \     # 这里就自动传入字符串形式了，不用再写成 "train,val,test"形式？？


# 9月23日，同样pretrain80个epoch
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test16_noGAN_pretrain80/' \
  --gan_epochs 0 \
  --pretrain_epochs 80 \
  --lambda_adv 0.2 \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --base_channels_d 8 \
  --base_channels_g 64 \
  --batch_size 64 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lead_steps 1 \
  --generator_version 'convgru' \
  --save_every_epoch \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --train_message '同样pretrain80个epoch。 其余设置对齐 GAN_test15_nopretrain_epoch80' \
  > train_noGAN_pretrain80.log 2>&1 &
## 评估
python evaluation/General_eval.py \
  --checkpoint_dir './logs/GAN_test16_noGAN_pretrain80/check_points' \
  --splits train,val,test \

## 9月22日，增加到4个lead time，重新原版训练GAN，80个epoch。 
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test17_nopretrain_epoch80_lead4/' \
  --gan_epochs 80 \
  --pretrain_epochs 0 \
  --lambda_adv 0.2 \
  --base_channels_d 8 \
  --base_channels_g 64 \
  --batch_size 32 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --lead_steps 4 \
  --generator_version 'convgru' \
  --save_every_epoch \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --train_message '增加到4个lead time，重新原版训练GAN，80个epoch。 其余设置对齐 GAN_test15_nopretrain_epoch80' \
  > train_nopretrain_lead4.log 2>&1 &
  ## 评估
python evaluation/General_eval.py \
  --checkpoint_dir './logs/GAN_test17_nopretrain_epoch80_lead4/check_points' \
  --splits train,val,test \


# 9月24日，增加温度预报，无pretrain直接训练GAN80epoch，lead time设置为12.
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test18_nopretrain_epoch80_lead12/' \
  --batch_size 4 \
  --base_channels_d 8 \
  --base_channels_g 64 \
  --gan_epochs 80 \
  --pretrain_epochs 0 \
  --lambda_adv 0.2 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --lead_steps 12 \
  --generator_version 'convgru' \
  --save_every_epoch \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --lambda_temp 0.2 \
  --train_message '增加温度预报，无pretrain直接训练GAN80epoch，lead time设置为12. 其余设置对齐 GAN_test15_nopretrain_epoch80_lead4' \
  > train_nopretrain_lead12.log 2>&1 &
## 评估
python evaluation/General_eval.py \
  --checkpoint_dir './logs/GAN_test18_nopretrain_epoch80_lead12/check_points' \
  --splits train,val,test \


# 9月25日，运行M0实验： 纯generator训练80个epoch，纯charbonnier损失函数。保存所有模型。 挑选best pretrain的模型的指标确定后，再去挨个运行指标挑选
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test19_M0_epoch80_purecharbonnier/' \
  --batch_size 4 \
  --base_channels_d 32 \
  --base_channels_g 64 \
  --gan_epochs 0 \
  --pretrain_epochs 80 \
  --lambda_adv 0.2 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --lead_steps 12 \
  --generator_version 'convgru' \
  --save_every_epoch \
  --extreme_alpha 0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --lambda_temp 0.2 \
  --train_message '纯generator训练80个epoch，纯charbonnier损失函数。保存所有模型。 挑选best pretrain的模型的指标确定后，再去挨个运行指标挑选' \
  > train_M0_epoch80_purecharbonnier.log 2>&1 &
# M0实验训练模型的评估代码,需要有nohup
nohup python3 -u evaluation/General_eval_2.py \
  --checkpoint_dir './logs/GAN_test19_M0_epoch80_purecharbonnier/check_points' \
  > eval_M0_epoch80_purecharbonnier.log 2>&1 &


# 9月25日，运行M1实验,仅添加极值加权的损失
CUDA_VISIBLE_DEVICES=0,1,2,3,4
nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test19_M1_epoch80_extremeweight/' \
  --batch_size 4 \
  --base_channels_d 32 \
  --base_channels_g 64 \
  --gan_epochs 0 \
  --pretrain_epochs 80 \
  --lambda_adv 0.2 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --lead_steps 12 \
  --generator_version 'convgru' \
  --save_every_epoch \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --lambda_temp 0.2 \
  --train_message '纯generator训练80个epoch，带极值加权的charbonnier损失函数。保存所有模型。 挑选best pretrain的模型的指标确定后，再去挨个运行指标挑选' \
  > train_M1_epoch80_extremeweight.log 2>&1 &
## 评估函数
nohup python3 -u evaluation/General_eval_2.py \
  --checkpoint_dir './logs/GAN_test19_M1_epoch80_extremeweight/check_points' \
  > eval_M1_epoch80_extremeweight.log 2>&1 &


# SimVP 对照 M1：纯 generator 预训练 80 个 epoch，极值加权 Charbonnier，保存每个 epoch。训练成功退出后自动评估。
CUDA_VISIBLE_DEVICES=0,1,2,3,4 nohup bash -c 'torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir "./logs/GAN_test22_SimVP_epoch80_extremeweight/" \
  --batch_size 4 \
  --base_channels_d 32 \
  --base_channels_g 64 \
  --gan_epochs 0 \
  --pretrain_epochs 80 \
  --lambda_adv 0.2 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --lead_steps 12 \
  --generator_version simvp \
  --save_every_epoch \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --lambda_temp 0.2 \
  --train_message "纯generator训练80个epoch，SimVP，带极值加权的charbonnier损失函数。保存所有模型。" \
  > train_SimVP_epoch80_extremeweight.log 2>&1 \
&& python3 -u evaluation/General_eval_2.py \
  --checkpoint_dir "./logs/GAN_test22_SimVP_epoch80_extremeweight/check_points" \
  --generator_version simvp \
  > eval_SimVP_epoch80_extremeweight.log 2>&1' &


# 运行M2实验。 M0实验的第63个epoch训练效果最好(依据每个模型的12个lead上overallCSI指标的平均最大值选定)
# 在此pretrain模型的基础上，继续GAN训练80个epoch。
CUDA_VISIBLE_DEVICES=5,2,3,4,6 nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test19_M2_epoch80_GANonpretrain/' \
  --batch_size 4 \
  --base_channels_d 32 \
  --base_channels_g 64 \
  --gan_epochs 80 \
  --pretrain_epochs 0 \
  --pretrain_path '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M0_epoch80_purecharbonnier/check_points/pretrain_epoch_063.pt' \
  --lambda_adv 0.1 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --lead_steps 12 \
  --generator_version 'convgru' \
  --save_every_epoch \
  --extreme_alpha 0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --lambda_temp 0.2 \
  --train_message '在M0实验的第63个epoch训练效果最好的pretrain模型的基础上，继续GAN训练80个epoch。GAN训练的lambda_adv设置为0.1，generator依然用纯charbonnier损失函数。' \
  > train_M2_epoch80_GANonpretrain.log 2>&1 &
## 评估
nohup python3 -u evaluation/General_eval_2.py \
  --checkpoint_dir './logs/GAN_test19_M2_epoch80_GANonpretrain/check_points' \
  --no_overwrite \
  > eval_M2_epoch80_GANonpretrain.log 2>&1 &


# 进行M3实验。在M1实验最好overallCSI指标的那个epoch的模型基础上，继续GAN训练80个epoch。
CUDA_VISIBLE_DEVICES=0,1,2,3,4 nohup torchrun --standalone --nproc_per_node=5 train_GAN_test8_a1.py \
  --save_dir './logs/GAN_test19_M3_epoch80_GANonExtremeweightM1pretrain/' \
  --batch_size 4 \
  --base_channels_d 32 \
  --base_channels_g 64 \
  --gan_epochs 80 \
  --pretrain_epochs 0 \
  --pretrain_path '/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M1_epoch80_extremeweight/check_points/pretrain_epoch_027.pt' \
  --lambda_adv 0.1 \
  --num_workers 0 \
  --real_label_value 1. \
  --fake_label_value 0. \
  --lr_g 2e-5 \
  --lr_d 1e-5 \
  --lead_steps 12 \
  --generator_version 'convgru' \
  --save_every_epoch \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  --lambda_temp 0.2 \
  --train_message '在M1实验的训练效果最好的pretrain模型的基础上，继续GAN训练80个epoch。GAN训练的lambda_adv设置为0.1，generator用带极值加权的charbonnier损失函数。' \
  > train_M3_epoch80_GANonExtremeweightM1pretrain.log 2>&1 &
## 评估
nohup python3 -u evaluation/General_eval_2.py \
  --checkpoint_dir './logs/GAN_test19_M3_epoch80_GANonExtremeweightM1pretrain/check_points' \
  > eval_M3_epoch80_GANonExtremeweightM1pretrain.log 2>&1 &


import pandas as pd
lead_steps = 12
modelOverallCSI_dic = {}
checkpoint_path = '/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M1_epoch80_extremeweight/check_points'
for lead in range(1,lead_steps+1):
    modelOverallCSI_dic[lead] = pd.read_csv(checkpoint_path+'/'+'modelOverallCSI_epochCurve_lead'+str(lead)+'.csv',index_col=0)['val']

bb = [value for i,value in modelOverallCSI_dic.items()]
bbb = pd.concat(bb,axis=1)
bbb.mean(axis=1).idxmax()    #  epoch 27     0.170


import pandas as pd
lead_steps = 12
modelOverallCSI_dic = {}
checkpoint_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M0_epoch80_purecharbonnier/check_points'
for lead in range(1,lead_steps+1):
    modelOverallCSI_dic[lead] = pd.read_csv(checkpoint_path+'/'+'modelOverallCSI_epochCurve_lead'+str(lead)+'.csv',index_col=0)['val']

bb = [value for i,value in modelOverallCSI_dic.items()]
bbb = pd.concat(bb,axis=1)
bbb.mean(axis=1).idxmax()    #  epoch 63     0.135


import pandas as pd
lead_steps = 12
modelOverallCSI_dic = {}
checkpoint_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M2_epoch80_GANonpretrain/check_points'
for lead in range(1,lead_steps+1):
    modelOverallCSI_dic[lead] = pd.read_csv(checkpoint_path+'/'+'modelOverallCSI_epochCurve_lead'+str(lead)+'.csv',index_col=0)['val']

bb = [value for i,value in modelOverallCSI_dic.items()]
bbb = pd.concat(bb,axis=1)
bbb.mean(axis=1).idxmax()    #  epoch 1     0.134481




import pandas as pd
lead_steps = 12
modelOverallCSI_dic = {}
checkpoint_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M3_epoch80_GANonExtremeweightM1pretrain/check_points'
for lead in range(1,lead_steps+1):
    modelOverallCSI_dic[lead] = pd.read_csv(checkpoint_path+'/'+'modelOverallCSI_epochCurve_lead'+str(lead)+'.csv',index_col=0)['val']

bb = [value for i,value in modelOverallCSI_dic.items()]
bbb = pd.concat(bb,axis=1)
bbb.mean(axis=1).idxmax()    #  epoch 4     0.16771472932652035





'''


def parse_args() -> TrainConfig:    # 这里的  ->什么含义？   

    p = argparse.ArgumentParser(description="Train a conditional GAN for ERA5 China cloud forecasting.")
    p.add_argument("--zarr_path", type=str,default='/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr' )  # 这里应该加一个default   required=True 表示命令行必须传入这个参数。
    p.add_argument("--save_dir", type=str, default="./logs/checkpoints_cloud_gan")   # 
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)    #不再有 4 个 worker 子进程，也就绕开了多进程读 Zarr/xarray 的锁和进程通信问题，所以可以正常取 batch。
    p.add_argument("--max_val_samples", type=int, default=2048)   # 验证集以后都是用全部。。 验证集是一年，最多大概有 1447个，一年有1447个样本
    p.add_argument("--max_test_samples", type=int, default=None)   # 测试集的样本数量，None表示不限制。那到底是多少个测试集样本数量？
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--pretrain_path", type=str, default=None,
                   help="GAN 开始前加载的预训练生成器。不传时使用 save_dir/best_pretrain_generator.pt")
    p.add_argument("--gan_epochs", type=int, default=50)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--lambda_adv", type=float, default=1e-3)   # 突然想明白： 为什么discriminator过强的时候应该降低lambda_adv_g，因为降低了，generator就可以更专注于从真实样本中更新参数，容易变得强起来，和discriminator对抗
    p.add_argument("--base_channels_g", type=int, default=32)
    p.add_argument("--base_channels_d", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")    # "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--no_amp", action="store_true")    # 采用no_amp     action='store_true':当命令行中出现该参数时，会将对应变量设为 True；未出现时则为 False
    p.add_argument("--real_label_value",type=float,default=1.0)   #
    p.add_argument("--fake_label_value",type=float,default=0.)  # 平滑标签，可以增大d_adv_loss ?  
    p.add_argument("--lead_steps",type=int,default=12)   # 预报lead time，需要将其减小到4测试一下
    p.add_argument("--generator_version",type=str,default='unet')
    p.add_argument("--extreme_alpha", type=float, default=0.0,
                   help="连续极值权重系数。0 退化为原 Charbonnier；实验默认传 2.0")
    p.add_argument("--extreme_beta", type=float, default=0.5,
                   help="连续极值权重指数，w=1+alpha*(y/y_ref)^beta")
    p.add_argument("--y_ref", type=float, default=1e-5,
                   help="物理量参考值 kg/kg，约多数高度/要素 70%-80% 分位")
    p.add_argument("--w_max", type=float, default=100.0,
                   help="极值权重上限，防止极端格点梯度爆炸")
    p.add_argument("--lambda_temp", type=float, default=0.2,
                   help="温度纯 Charbonnier 相对云要素损失的权重")
    p.add_argument("--save_every_epoch", action="store_true",
                   help="每个 epoch 额外保存 check_points/*_epoch_XXX.pt")
    p.add_argument("--train_message",type=str,help="details of this train type",\
    default="本次训练的配置细节信息： 对discriminator使用了条件参数更新： 当对抗损失小于0.2不让其更新参数")
    args = p.parse_args()
    pretrain_path = args.pretrain_path or str(Path(args.save_dir) / "best_pretrain_generator.pt")
    return TrainConfig(
        zarr_path=args.zarr_path,
        train_message = args.train_message,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_path=pretrain_path,
        gan_epochs=args.gan_epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_adv=args.lambda_adv,
        base_channels_g=args.base_channels_g,    # 这也需要自定义？  都是默认32，不要去定义这个
        base_channels_d=args.base_channels_d,   # 就是定义卷积核的数量，第一次卷积操作的卷积核数量，之后卷积的数量在依次翻倍
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,   #？默认amp 为True的bool值
        real_label_value = args.real_label_value,
        fake_label_value = args.fake_label_value,
        lead_steps = args.lead_steps,
        generator_version = args.generator_version,
        extreme_alpha=args.extreme_alpha,
        extreme_beta=args.extreme_beta,
        y_ref=args.y_ref,
        w_max=args.w_max,
        save_every_epoch=args.save_every_epoch,
        lambda_temp=args.lambda_temp,
    )



@dataclass     
class TrainConfig:   ## 单纯定义参数和参数类型？  不能在命令行中运行文件时，在通过args自定义一些参数？   
    '''  NOTE  这个函数的意思是： 如果没有使用parse_args函数，就使用这里的默认的配置
    是不是在这里定义各种超参数不用在linux命令行运行程序也可以？   那在命令行中运行这个程序，带上超参数，谁的优先级更高呢？   这里训练模式，有输出各种训练超参数的设置吗？
TrainConfig 是一个 dataclass，用来集中保存训练超参数。
    命令行参数优先。‘

    如果你不想每次在命令行写，也可以直接修改 TrainConfig 默认值或 parse_args() 的 default。
    '''
    zarr_path: str     #  /mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr/
    train_message: str
    save_dir: str = "./logs/checkpoints_cloud_gan"    # /home/yx/projects/common_data/sais2025/test1/logs/某个test文件夹/    需要在正式存储内容之前，检查一下文件夹名字是否存在，并创建一下。
    batch_size: int = 4
    num_workers: int = 0   # 什么意思？  子进程？
    max_val_samples: int = 2048   # >>> len(loaders.val.dataset)   1447..    
    max_test_samples: Optional[int] = None  # 这里表示这个变量可以是None，也可以是int吗？ 
    pretrain_epochs: int = 30  # 与训练的轮次
    pretrain_path: Optional[str] = None   # 预训练模型的路径
    gan_epochs: int = 50   #正式训练的轮次
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.9   # 
    beta2: float = 0.999 #   adam优化器的动量相关的参数
    lambda_adv: float = 1e-3
    base_channels_g: int = 32  # generator的卷积核数量
    base_channels_d: int = 8  # 判别器第一层通道数，之后两层依次翻倍
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"   # 先在一个GPU上训练，之后再看并行
    amp: bool = True  # 什么意思？ 自动混合精度训练（Automatic Mixed Precision） ？它会自动把部分运算用 float16 / bfloat16 执行，以节省显存并加速。
    real_label_value:float=1.
    fake_label_value:float = 0.
    lead_steps:int = 12 
    generator_version : str = 'unet'
    extreme_alpha: float = 0.0
    extreme_beta: float = 0.5
    y_ref: float = 1e-5
    w_max: float = 100.0
    save_every_epoch: bool = False
    lambda_temp: float = 0.2
# `18个变量，之后的cfg也就这18个变量`






# =========================
# DDP helper functions
# =========================

def ddp_is_initialized() -> bool:
    """Return True if torch.distributed has been initialized."""
    return dist.is_available() and dist.is_initialized()     # ？什么含义？


def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0


def is_main_process() -> bool:   # 
    return get_rank() == 0


def setup_distributed(cfg: TrainConfig):
    """Initialize DDP from torchrun environment variables.

    torchrun will provide RANK, LOCAL_RANK, WORLD_SIZE.
    In single-GPU/single-process runs these variables do not exist, so this
    function leaves the original cfg.device unchanged.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))    # DDP这些参数的具体含义？
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:   # 
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
        dist.barrier()   # ？ 


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when using DistributedDataParallel."""
    return model.module if isinstance(model, DDP) else model     # ?  去掉model的module属性？


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
def inverse_log10p_cloud(z:torch.Tensor,
    q0:float = 1e-8,
    clamp_min:bool=True,)->torch.Tensor:
    '''
    将 log-transform 空间的云变量反变换回原始物理量 q。
    训练数据变换:
        z = log10(1 + q / q0)
    反变换:
        q = q0 * (10**(z) - 1)
    Parameters
    ----------
    z:
        log-transform 后的云变量。
        shape 通常为 [B, lead_steps*4, D, H, W]
    q0:
        log-transform 中使用的尺度参数。
    clamp_min:
        是否将反变换后的负值截断为 0。
        云水凝物物理量 q 不应为负，因此建议 True。
    Returns
    -------
    q:
        原始物理空间云变量，单位仍是 kg/kg。
    '''
    q = q0*(torch.pow(10.0,z.float()) - 1.0)    # 这里不会报device的错误
    if clamp_min:    # 确实会出现极小负值的情况。因为Charbonnier会把0值区域可能生成一个极小负值。
        q = q.clamp_min(0.0)
    return q





@torch.no_grad()
def _temperature_scale(dataset) -> tuple[np.ndarray, np.ndarray]:
    idx = list(dataset.variable_names).index("temperature")
    return dataset.mean_all[idx], dataset.std_all[idx]


def validate_generator(
    generator: nn.Module,
    loader: DataLoader,
    criterion: CompositeGeneratorLoss,
    latitudes: np.ndarray,
    device: str,
    temp_mean: Optional[np.ndarray] = None,
    temp_std: Optional[np.ndarray] = None,
) -> dict:
    '''具体是在干嘛呢？  返回某一个批次，generator的指标。。。    用于用验证集评价generator效果，以保存最佳模型。
    csi的metric这里需要改
    '''
    generator.eval()
    total_loss = 0.0
    total_csi = 0.0
    total_temp_mae = 0.0
    n = 0
    mean_t = std_t = None
    if temp_mean is not None and temp_std is not None:
        mean_t = torch.as_tensor(temp_mean, device=device, dtype=torch.float32).view(1, 1, -1, 1, 1)
        std_t = torch.as_tensor(temp_std, device=device, dtype=torch.float32).view(1, 1, -1, 1, 1)
    for batch in loader:   # 为了计算所有数样本的误差平均。  # 这里应该不能迭代2048次，样本数最多1700个
        '''这里值迭代了362次？  为什么只迭代了362次呢？   val总共有1447个样本，这里batch_size = 4。  '''
        x, y, _ = _batch_to_device(batch, device)
        pred = generator(x)   # 维度： B, C, D, H, W
        loss = criterion(pred, y)
        pred_cloud, pred_temp = split_cloud_temp(pred)
        y_cloud, y_temp = split_cloud_temp(y)
        pred_q = inverse_log10p_cloud(pred_cloud)
        y_q = inverse_log10p_cloud(y_cloud)
        csi = csi_score(pred_q, y_q, threshold=0.0)
        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_csi += csi.item() * bs
        if mean_t is not None:
            temp_mae = ((pred_temp * std_t + mean_t) - (y_temp * std_t + mean_t)).abs().mean()
            total_temp_mae += temp_mae.item() * bs
        n += bs
    return {
        "loss": total_loss / max(n, 1),
        "csi_0": total_csi / max(n, 1),
        "temp_mae": total_temp_mae / max(n, 1),
    }



def pretrain_generator(
    generator: nn.Module,
    loaders: DataLoaders,
    criterion: CompositeGeneratorLoss,
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
    loss_log_ = {}

    for epoch in range(1, cfg.pretrain_epochs + 1):
        if hasattr(loaders.train, "sampler") and hasattr(loaders.train.sampler, "set_epoch"):
            loaders.train.sampler.set_epoch(epoch)

        t0 = time.time()
        generator.train()
        average_loss = {'total_loss': [], 'weighted_recon': [], 'temp_loss': []}
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
            average_loss['temp_loss'].append(loss_log['temp'].detach().item() * bs)

            scaler.scale(loss).backward()
            scaler.step(opt_g)
            scaler.update()
            seen += bs

        if is_main_process():
            # 用 unwrap_model 避免 rank 0 单独验证 DDP 包装模型时触发分布式 buffer 同步。
            temp_mean, temp_std = _temperature_scale(loaders.val_dataset)
            val_metrics = validate_generator(
                unwrap_model(generator), loaders.val, criterion,
                loaders.val_dataset.latitudes, device, temp_mean, temp_std,
            )
            average_loss = {key: sum(val) / max(seen, 1) for key, val in average_loss.items()}
            loss_log_[epoch] = {
                'total_loss_train': average_loss['total_loss'],
                'weighted_recon_train': average_loss['weighted_recon'],
                'total_loss_val': val_metrics['loss'],
                'temp_loss': average_loss['temp_loss'],
            }
            t1 = time.time()
            logging.info(
                f"[Pretrain {epoch:03d}/{cfg.pretrain_epochs}] "
                f"{t1-t0:.2f} sec train_loss={loss_log_[epoch]['total_loss_train']:.6f} "
                f"val_loss={val_metrics['loss']:.6f}  "
                f"val_csi_0={val_metrics['csi_0']:.4f}  "
                f"val_temp_mae={val_metrics['temp_mae']:.4f}"
            )
            if getattr(cfg, "save_every_epoch", False):    # ?
                (save_dir / "check_points").mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    save_dir / "check_points" / f"pretrain_epoch_{epoch:03d}.pt",
                    generator,
                    None,
                    {"epoch": epoch, "val_metrics": val_metrics, "stage": "pretrain"},
                )
            elif val_metrics["loss"] < best_val:
                # save_every_epoch 时每轮已落盘，不必再写 best。
                best_val = val_metrics["loss"]
                save_checkpoint(
                    save_dir / "best_pretrain_generator.pt",
                    generator,
                    None,
                    {"epoch": epoch, "val_metrics": val_metrics, "stage": "pretrain"},
                )
                logging.info(f"  -> saved best pretrain generator, val_csi_0={val_metrics['csi_0']:.4f}")

        # 确保非主进程等待 rank 0 完成验证与保存后再进入下一轮。
        barrier()

    if is_main_process():
        plot_loss_curves(loss_log_, 'pretrain', save_dir / "pretrain_loss_curves.png")

    barrier()
    return best_val


@torch.no_grad()
def evaluate_test(generator: nn.Module, loaders: DataLoaders, criterion: CompositeGeneratorLoss, cfg: TrainConfig) -> dict:
    '''
    '''
    generator.to(cfg.device)
    temp_mean, temp_std = _temperature_scale(loaders.test_dataset)
    metrics = validate_generator(
        generator, loaders.test, criterion, loaders.test_dataset.latitudes,
        cfg.device, temp_mean, temp_std,
    )
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
    return ckpt.get("extra", {})  # 返回额外的信息，比如epoch，val_metrics，stage等。??


def set_seed(seed: int = 42) -> None:   # 设置随机种子
    np.random.seed(seed)    # 这三个随机种子，各自具体什么作用和含义？
    torch.manual_seed(seed)   # torch.manual_seed(seed) is used in PyTorch to set the seed for generating random numbers on all devices (CPU and GPU), ensuring reproducibility of experiments. It returns a torch.Generator object that can be reused for controlled randomness.  确保每次生成的随机数是一样的。
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子 为所有可用的 GPU 设置随机数生成器的种子，以确保多 GPU 环境下的结果可复现。
    torch.backends.cudnn.benchmark = True   # 用于启用 cuDNN 的自动优化功能。它会根据硬件和输入数据的大小，选择最优的卷积算法，从而提升模型的运行效率。；一般场景下，只要简单地在 PyTorch 程序开头将其值设置为 True，就可以大大提升卷积神经网络的运行速度。；适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
# 把所有随机性设置随机种子


def masked_adv_loss(
    logits: torch.Tensor,
    cloud_mask: torch.Tensor,
    label_value: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """有云格点上的概率 MSE。cloud_mask 与 logits 同为 [B, 4*lead, D, H, W]。"""
    if logits.shape != cloud_mask.shape:
        raise ValueError(
            f"discriminator logits {tuple(logits.shape)} must match cloud_mask {tuple(cloud_mask.shape)}"
        )
    weight_mask = cloud_mask
    prob = torch.sigmoid(logits)
    target = torch.full_like(prob, label_value)
    mse_map = F.mse_loss(prob, target, reduction="none")
    return (mse_map * weight_mask).sum() / weight_mask.sum().clamp_min(eps)


###########  以上多个函数，是辅助train_gan用的


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    loaders: DataLoaders,
    criterion: CompositeGeneratorLoss,
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
    # adv_criterion = nn.BCEWithLogitsLoss()
    adv_criterion = nn.MSELoss()    # LSGAN ? 
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
            'total_loss': [], 'weighted_recon': [], 'discriminator_loss': [], 'adv_loss_g': [],'temp_loss': []
        }
        seen = 0
        for batch in loaders.train:
            x, y, _ = _batch_to_device(batch, device)
            y_cloud, _ = split_cloud_temp(y)
            y_cloud_mask = (y_cloud > 0.0).float()
            bs = x.shape[0]   # batch size
            # -----------------------
            # 1. Update discriminator
            # -----------------------
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_y = generator(x)
                fake_cloud, _ = split_cloud_temp(fake_y)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                real_logits = discriminator(x, y_cloud)
                fake_logits = discriminator(x, fake_cloud.detach())
                real_labels = cfg.real_label_value
                fake_labels = cfg.fake_label_value
                d_loss_real = masked_adv_loss(
                    logits=real_logits,
                    label_value=cfg.real_label_value,
                    cloud_mask=y_cloud_mask,
                )
                d_loss_fake = masked_adv_loss(
                    logits=fake_logits,
                    label_value=cfg.fake_label_value,
                    cloud_mask=y_cloud_mask,
                )
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                # 2. 跨卡同步：取 8 卡 d_loss 的平均值   ？
                d_loss_for_gate = d_loss.detach().clone()
                if ddp_is_initialized():
                    dist.all_reduce(d_loss_for_gate, op=dist.ReduceOp.AVG)
                # 3. 所有 rank 用同一个布尔值决策
                update_d = d_loss_for_gate.item() > 0.15    # d_loss大于0.15才会更新discriminator
                if update_d:
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(opt_d)
                    scaler_d.update()
            
            average_loss['discriminator_loss'].append(d_loss.detach().item() * bs)
            # ------------------
            # 2. Update generator
            # ------------------
            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp):
                fake_y = generator(x)
                fake_cloud, _ = split_cloud_temp(fake_y)
                fake_logits = discriminator(x, fake_cloud)
                adv_loss_g = masked_adv_loss(
                    logits=fake_logits,
                    label_value=cfg.real_label_value,
                    cloud_mask=y_cloud_mask,
                )
                loss_log = criterion.compute_components(fake_y, y)
                g_loss = loss_log['total'] + cfg.lambda_adv * adv_loss_g

            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()     #更新参数的这三个代码，各自什么含义？

            average_loss['total_loss'].append(g_loss.detach().item() * bs)
            average_loss['weighted_recon'].append(loss_log['recon'].detach().item() * bs)   # recon是Charbonnier loss
            average_loss['adv_loss_g'].append(adv_loss_g.detach().item() * bs)
            average_loss['temp_loss'].append(loss_log['temp'].detach().item()*bs)
            seen += bs

        if is_main_process():  # 具体什么含义？
            temp_mean, temp_std = _temperature_scale(loaders.val_dataset)
            val_metrics = validate_generator(
                unwrap_model(generator), loaders.val, criterion,
                loaders.val_dataset.latitudes, device, temp_mean, temp_std,
            )
            average_loss = {key: sum(val) / max(seen, 1) for key, val in average_loss.items()}    # 每个batch一个指数，所有batch的指数和除以这个epoch的总batch数： seen
            loss_log_[epoch] = {
                'discriminator_loss': average_loss['discriminator_loss'],
                'total_loss_train': average_loss['total_loss'],
                'weighted_recon_train': average_loss['weighted_recon'],
                'total_loss_val': val_metrics['loss'],
                'adv_loss_g': average_loss['adv_loss_g'],
                'temp_loss': average_loss['temp_loss'],
            }

            t1 = time.time()
            logging.info(
                f"[GAN {epoch:03d}/{cfg.gan_epochs}] "
                f"{t1-t0:.2f} sec "
                f"D_loss={loss_log_[epoch]['discriminator_loss']:.6f} "
                f"G_adv_raw={loss_log_[epoch]['adv_loss_g']:.6f} "
                f"Recon={loss_log_[epoch]['weighted_recon_train']:.6f} "    # 
                f"G_total_loss={loss_log_[epoch]['total_loss_train']:.6f}  "   # 就是generator在训练集上的损失函数
                f"val_loss={val_metrics['loss']:.6f}  val_csi_0={val_metrics['csi_0']:.4f}  "
                f"val_temp_mae={val_metrics['temp_mae']:.4f}"
            )
            if val_metrics['loss'] < best_val:   # 直接事宜损失函数(compositegeneratorLoss_v2的值的大小来评判最佳的模型的。)
                # best_val = val_metrics['loss']     # 这样每一轮都会保存模型
                save_checkpoint(
                    save_dir / "check_points" / f"gan_epoch_{epoch:03d}.pt",
                    generator,
                    discriminator,     # 每一轮都会保存，没用的discriminator。
                    {"epoch": epoch, "val_metrics": val_metrics, "stage": "gan"},
                )
                logging.info(f"  -> saved best GAN checkpoint, val_csi_0={val_metrics['csi_0']:.4f}")

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
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "check_points").mkdir(parents=True, exist_ok=True)

        if is_main_process():
            setuplogging(str(save_dir / "0.log"))
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
            forecast_steps = cfg.lead_steps,
            seed=cfg.seed,
        )
        rebuild_train_loader_for_ddp(loaders, cfg)    # ?

        if is_main_process():
            print(f"[INFO] train samples: {len(loaders.train_dataset)}")
            print(f"[INFO] val samples:   {len(loaders.val_dataset)}")
            print(f"[INFO] test samples:  {len(loaders.test_dataset)}")

            # One batch shape check only on rank 0.
            xb, yb, meta = next(iter(loaders.train))
            print(f"[INFO] x batch shape: {tuple(xb.shape)}  expected [B,9*2,13,41,71]")
            print(f"[INFO] y batch shape: {tuple(yb.shape)}  expected [B,lead_steps*5,13,41,71]")
            print(f"[INFO] first sample meta: {meta}")
            logging.info(cfg.train_message)
        if cfg.generator_version == 'unet':
            generator = UNet3DGenerator(in_channels=18, out_channels=5*cfg.lead_steps, base_channels=cfg.base_channels_g).to(cfg.device)
        elif cfg.generator_version == 'convgru':
            generator = ConvGRUUNet3DGenerator(in_channels=18,out_channels=5*cfg.lead_steps,base_channels=cfg.base_channels_g,input_steps=2,input_vars=9,output_vars=5,forecast_steps=cfg.lead_steps).to(cfg.device)
        elif cfg.generator_version == 'simvp':
            generator = SimVP3DGenerator(in_channels=18,out_channels=5*cfg.lead_steps,base_channels=cfg.base_channels_g,input_steps=2,input_vars=9,output_vars=5,forecast_steps=cfg.lead_steps).to(cfg.device)
        elif cfg.generator_version == 'swin':
            generator = SwinTransformer3DGenerator(in_channels=18,out_channels=5*cfg.lead_steps,base_channels=cfg.base_channels_g,input_steps=2,input_vars=9,output_vars=5,forecast_steps=cfg.lead_steps).to(cfg.device)
        # generator: 每个 lead 4 个云要素 + 1 个温度
        discriminator = VoxelLocalDiscriminator(
            in_channels=18 + 4 * cfg.lead_steps,
            out_channels=4 * cfg.lead_steps,
            base_channels=cfg.base_channels_d,
        ).to(cfg.device)
        # discriminator 只看云通道，不看温度
        criterion = CompositeGeneratorLoss(
            latitudes=loaders.train_dataset.latitudes,
            extreme_alpha=cfg.extreme_alpha,
            extreme_beta=cfg.extreme_beta,
            y_ref=cfg.y_ref,
            w_max=cfg.w_max,
            lambda_temp=cfg.lambda_temp,
        ).to(cfg.device)
        if is_main_process():
            logging.info(
                "Charbonnier extreme weight: "
                f"alpha={cfg.extreme_alpha}, beta={cfg.extreme_beta}, "
                f"y_ref={cfg.y_ref}, w_max={cfg.w_max}, lambda_temp={cfg.lambda_temp}"
            )

        # DDP wrapping must happen after .to(device) and before optimizer creation.
        if ddp_is_initialized():
            generator = DDP(generator, device_ids=[local_rank], output_device=local_rank)
            discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)

        if is_main_process():
            logging.info("Pretraining generator...")
        pretrain_generator(generator, loaders, criterion, cfg)

        # 只需要pretrain  80个epoch，如何在这里break？   不需要break
        
        pretrain_path = Path(cfg.pretrain_path) if cfg.pretrain_path else Path(cfg.save_dir) / "best_pretrain_generator.pt"
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
            extra = load_generator_checkpoint(generator, best_gan_path, cfg.device)   # 这个函数就已经把generator的权重加载进去了。
            logging.info(f"Loaded best GAN generator: {extra}")
            metrics = evaluate_test(unwrap_model(generator), loaders, criterion, cfg)
            with open(Path(cfg.save_dir) / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()




