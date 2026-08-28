

'''
打算在这里训练一个mask_predictor，模型配置和流程逻辑仿照：AviaSafe: A physics-informed data-driven model for aviation safety-critical cloud forecasts
1，先在这里走流程，写完了代码在封装
2，先自己写写代码，之后再给GPT检查，这样：一是可以对代码细节掌握更加透彻熟悉，二是可以提升自己项目能力。

Atmospheric Encoder:
    Conv stem
    Residual Conv blocks
    Downsample to H/4, W/4
    output F_atm
Physics Encoder E_Mask:
    Conv stem
    Residual Conv blocks
    Downsample to H/4, W/4
    output F_mask
Fusion:
    concat(F_atm, F_mask)
    1×1 Conv / MLP projection
Context:
    1–2 Swin Blocks
Decoder:
    Upsample + Conv
    Upsample + Conv
    Output Conv
Output:
    mask_logits: [B, lead_steps*13, H, W]
    mask_prob = sigmoid(mask_logits)

TODO: 
1,把ConvGRU那个代码弄清楚
2，把这里的残差，等网络结构的原理弄清楚
3，把Gelu，Silu这些激活函数的含义弄清楚。
4，把transformer用ide一行行运行弄清楚原理。
5，git上传代码到github上去。   git运用要更加熟悉熟练，融入到开发中去。
6，仔细研读代码，用ide一行行过数据流弄清楚原理。 在代码中注释自己不清楚的点困惑的点，上传给GPT用提示词要求详尽解释。
7，之后需要，更改这些模型的流程。   在src的源代码中定义单个模块，然后细节处理，维度变换，多个模块接续处理这些，在主程序中进行就好。 不要全部打包到模型代码文件中。 流程打包到模块代码本身就很怪。 所以之后要改。
8，要把ddp并行的代码搞清楚，学会用。
9，由于是多个模型块组成数据流，训练的时候，需要写一个函数打包这些块吗？ 构成一个总模型？ 
10，需要把AdamW搞清楚
'''

import sys
sys.path.append("/home/yx/projects/common_data/sais2025/test1/")  # 不能把src也加上
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr  # 没必要使用try 和 except   。   
import numpy as np
from src.data.data_loader_v2 import DataLoaders,make_dataloaders
from src.data.fixed_info import INPUT_VARIABLES,CLOUD_VARIABLES,CLOUD_SHORT_NAMES,THRESHOLDS,THRESHOLDS_normalized

from src.models.Atmospheric_Encoder import AtmosphericEncoder
from src.models.Physical_Encoder import PhysicalEncoder
from src.models.transformer import SwinContextModule
from src.models.mask_decoder import MaskUpsampleDecoder

zarr_path = '/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr'
batch_size = 4
forecast_steps = 4
seed = 42
device = "cuda:0"   #?





loaders_unnorm = make_dataloaders(zarr_path=zarr_path,  # 没有进行标准化的loaders
    batch_size = 4,
    num_workers = 0,
    max_val_samples = 2048,
    max_test_samples = None,
    seed = 42,
    forecast_steps = forecast_steps,
    normalize = False)   # 返回无标准化的loaders
pressures = loaders.train.dataset.pressure_levels

x,y,_ = next(iter(loaders_unnorm.train))   # 生成一个样本测试
# x.shape : torch.Size([4, 18, 13, 41, 71])

x_ = x.view(batch_size,2,9,13,41,71)   # tensor的维度变换是行优先，不用担心数据错乱
x_ic = x_[:,-1,[1,2],:,:,:]   # 要素顺序： 'temperature', 'specific_humidity'
# [4, 2, 13, 41, 71])

def get_Mask_ic(x_ic,pressures,):
    '''从当前时刻(不是输入的两个时刻)获取IC index的分布。 输入数据是(B,2,13,H,W)输出数据是(B,13,H,W)
    原始温度数据单位kelvin
    '''
    Tc = x_ic[:,0,:,:,:] - 273.15
    Qk = x_ic[:,1,:,:,:]
    es = 6.10094 * np.exp(17.625*Tc/(Tc+243.04))
    epsilon = 0.622
    p = torch.tensor(pressures)
    fQ = 2. * (p[None,:,None,None]*Qk/(epsilon*es)-0.5)
    fT = Tc * (Tc+14.)/(-49)
    ICk = fQ*fT    # torch.Size([4, 13, 41, 71]) ✅
    return ICk


x_cloud = x_[:,-1,-4:,:,:,:]   #   


def get_Mask_cloud(x_cloud,):
    '''从当前时刻的4个云要素的tensor，获取(B,4,13,H,W)的mask数组。
    输入x_cloud的数组维度：  torch.Size([4, 4, 13, 41, 71])   最终还需要变换为([4,52,41,71])
    '''
    cloud_mask = x_cloud != 0    # 非零值替换为True，零值替换为False。
    cloud_mask = cloud_mask.reshape(cloud_mask.shape[0],cloud_mask.shape[1]*cloud_mask.shape[2],cloud_mask.shape[3],cloud_mask.shape[4])
    return cloud_mask


Mask_cloud = get_Mask_cloud(x_cloud)   # torch.Size([4, 52, 41, 71])
Mask_ic = get_Mask_ic(x_ic,pressures)   # torch.Size([4, 13, 41, 71])

E_mask = torch.concat((Mask_cloud,Mask_ic),dim=1)    # torch.Size([4, 65, 41, 71])



# 处理所有大气数据
x_all_flat = x_.view(x_.shape[0],x_.shape[1]*x_.shape[2]*x_.shape[3],x_.shape[4],x_.shape[5])     
# 从torch.Size([4, 2, 9, 13, 41, 71])  -->  torch.Size([4, 234, 41, 71])

'''Atmospheric Encoder  结构框架：
Conv2d stem
↓
Residual Conv Blocks
↓
Downsample to H/2, W/2
↓
Downsample to H/4, W/4
↓
F_atm

'''

encoder = AtmosphericEncoder(in_channels = 234,   # 输入数据的维度是234
        base_channels=64,
        out_channels=128)

F_atm = encoder(x_all_flat)   # torch.Size([4, 128, 11, 18])  神经网络提取出的 128 个抽象特征通道





# Physical Encoder
'''Physics Encoder E_Mask:
    Conv stem
    Residual Conv blocks
    Downsample to H/4, W/4
    output F_mask
'''
physcial_encoder = PhysicalEncoder(in_channels=65,
        base_channels=64,
        out_channels=128,)
F_mask = physcial_encoder(E_mask)    # torch.Size([4, 128, 11, 18])


F = torch.cat([F_atm,F_mask],dim=1)   # torch.Size([4, 256, 11, 18])






# Context, transformer 处理

Swin_block = SwinContextModule(in_channels=256,
    embed_dim=128,
    num_heads=4,
    window_size=6)
F_context = Swin_block(F)   # torch.Size([4, 128, 11, 18])

decoder = MaskUpsampleDecoder(in_channels=128,
    mid_channels=96,
    hidden_channels=64,
    lead_steps=4,
    num_cloud_vars=4,
    num_levels=13,
    output_size=(41,71),)

mask_logits = decoder(F_context)  # torch.Size([4, 208, 41, 71])
mask_prob = decoder.logits_to_prob(mask_logits)  # torch.Size([4, 4, 4, 13, 41, 71])
mask_binary = decoder.logits_to_binary_mask(mask_logits, threshold=0.5)    # 通过选定概率阈值，生成哪里有云哪里没云的binary mask



