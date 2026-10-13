
'''
准备M0-M3实验的图件。
首先需要挑选M0-M3的4歌实验中，最佳的model，最佳的挑选条件是： overallCSI
我挑选的标准是： 每个epoch生成的模型，以1e-5为阈值的overallCSI在各个高度各个要素所有lead time上的平均值最大的那个epoch为最佳模型
挑选最佳epoch的代码：
import pandas as pd
lead_steps = 12
modelOverallCSI_dic = {}
checkpoint_path = '/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M1_epoch80_extremeweight/check_points'
for lead in range(1,lead_steps+1):
    modelOverallCSI_dic[lead] = pd.read_csv(checkpoint_path+'/'+'modelOverallCSI_epochCurve_lead'+str(lead)+'.csv',index_col=0)['val']

bb = [value for i,value in modelOverallCSI_dic.items()]
bbb = pd.concat(bb,axis=1)
bbb.mean(axis=1).idxmax()    #  epoch 27     0.170


TODO：
1，按照上述方法： 挑选出M0-M3四个实验的，各自最佳模型，所在的路径。
2，制作模型的数据流程图
3，

'''
M0_model_path = '/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M0_epoch80_purecharbonnier/check_points/pretrain_epoch_063.pt'
M1_model_path = '/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M1_epoch80_extremeweight/check_points/pretrain_epoch_027.pt'
M2_model_path = '/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M2_epoch80_GANonpretrain/check_points/gan_epoch_001.pt'
M3_model_path = '/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/GAN_test19_M3_epoch80_GANonExtremeweightM1pretrain/check_points/gan_epoch_004.pt'








