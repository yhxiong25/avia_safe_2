

from src.visualizations.evaluate_test2 import NormallyGANmodelEvaluation

if __name__ == "__main__":
    evaluator = GANForecastEvaluator(
        model_path="/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test7_modifiedADVLoss_5/best_gan.pt",
        save_dir="/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test6/figure/",    # save_dir不需要特别指定，直接在模型文件目录下面，创建一个目录即可。 
        zarr_path="/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr",
        level_hpa=500.0,
        batch_size=4,
        num_workers=0,
        device="cuda:0",
        base_channels_g=32,
    )

    df = evaluator.run()
    print(df)




evaluator = NormallyGANmodelEvaluation(
    model_path = '/home/yx/projects/common_data/sais2025/era5_cloud_gan/logs/GAN_test7_modifiedADVLoss_5/best_gan.pt')

evaluator.visualize_one_case(anchor_time="2022-06-15 00:00:00")









