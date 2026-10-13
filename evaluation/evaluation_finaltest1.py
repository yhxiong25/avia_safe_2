
import pandas as pd 
import sys
sys.path.append("/mnt/sdf/common_data/sais2025/era5_cloud_gan")
from src.visualizations.evaluate_test3 import NormallyGANmodelEvaluation

'''
现阶段对训练好的模型的评估：
1，绘制500hPa上，不同时刻的预报效果图。
2，生成250，500，850hPa上四个云要素的各种指标随lead time的变化。

'''


# 要检验的模型
model_path =  '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test7_modifiedADVLoss_5/best_pretrain_generator.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test1/best_pretrain_generator.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test1/best_gan.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test9_NewlyNormalized_debug/best_gan.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test9_NewlyNormalized_debug/best_pretrain_generator.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test_ddp/best_gan.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test_ddp/best_pretrain_generator.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test10_reducedLead/best_pretrain_generator.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test10_reducedLead/best_gan.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test11_ConvGRU/best_pretrain_generator.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test11_ConvGRU_2/best_pretrain_generator.pt'
model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test12_a1_/best_gan.pt'




evaluator = NormallyGANmodelEvaluation(
    model_path = model_path,
    lead_steps=4,
    generator_version='convgru')




# 每个月15日的都画出来
date_index = pd.date_range(
    start="2022-01-15", 
    end="2022-12-15", 
    freq=pd.DateOffset(months=4))    # ?    间隔一个月来取日期index
dates_ = date_index.strftime("%Y-%m-%d").tolist()
# lead_steps = [1,6,12]   # 多画几个lead step
lead_steps = [1,2,3,4]
level_ = [1000,850,925,500,400,150]
for date in dates_:
    for lead_step in lead_steps:
        for level in level_:
            evaluator.visualize_one_case(anchor_time=date,lead_step=lead_step,plot_level=level)

levels_ = [250,500,850]    # 
metrics_ = ["CSI", "POD", "FAR", "ACC", "RMSE"]
for level in levels_:
    evaluator.get_metrics_df(level=level)
    for metric in metrics_:
        evaluator.plot_metric(metric =metric ,level=level)




















