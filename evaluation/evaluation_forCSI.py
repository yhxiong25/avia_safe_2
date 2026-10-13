import pandas as pd 
import sys
sys.path.append("/mnt/sdf/common_data/sais2025/era5_cloud_gan")
from src.visualizations.evaluate_test3 import NormallyGANmodelEvaluation
import os
from pathlib import Path

splits = ['test', 'val', 'train']


model_list = []

ganModelPath = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test13_a1_lead1/check_points'     # pretrain_model也需要弄到checkpoints里面
model_list.extend(sorted(os.listdir(ganModelPath)))
# pretrainModel = '/home/yx/projects/common_data/sais2025/test1/logs/GAN_test12_a11/best_pretrain_generator.pt'
# model_list.append('best_pretrain_generator.pt')

model_list = [filename for filename in model_list if filename.endswith('pt')]

def get_csi_metrics_enhanced(split):
    for model in model_list:
        model_path = Path(ganModelPath) / model
        evaluator = NormallyGANmodelEvaluation(
            model_path = model_path,
            lead_steps=1,
            base_channels_g=64,
            generator_version='convgru',
            split=split)
        evaluator.get_csi_metrics_enhanced()


for split in splits:
    get_csi_metrics_enhanced(split)



sys.exit()  # 终止python进程


for model in model_list:   # 绘图
    model_path = Path(ganModelPath) / model
    evaluator = NormallyGANmodelEvaluation(   # 每次绘一个图像，都要重新加载一遍模型，极大拖慢速度
        model_path = model_path,
        lead_steps=4,
        base_channels_g=64,
        generator_version='convgru',
        )   # split参数这个用默认的就行
    date_index = pd.date_range(
        start="2022-01-15", 
        end="2022-12-15", 
        freq=pd.DateOffset(months=5))    # ?   
    dates_ = date_index.strftime("%Y-%m-%d").tolist()
    lead_steps = [1,2,3,4]
    level_ = [1000,850,925,500,400,150]
    for date in dates_:
        for lead_step in lead_steps:
            for level in level_:
                evaluator.visualize_one_case(anchor_time=date,lead_step=lead_step,plot_level=level)

















