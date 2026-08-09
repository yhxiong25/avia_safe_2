'''
存放固定的，列表信息。

使用方法：
import sys
sys.path.append('/home/yx/projects/common_data/sais2025/test1')
from src.data.fixed_info import INPUT_VARIABLES,CLOUD_SHORT_NAMES,THRESHOLDS,THRESHOLDS_normalized


'''




INPUT_VARIABLES = [   # 文件名路径？ 
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
]

CLOUD_VARIABLES = [
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
]

CLOUD_SHORT_NAMES = ["CIWC", "CLWC", "CRWC", "CSWC"]   # 顺序至关重要

LEVELS = [1000, 925, 850, 700, 600, 500, 400,
        300, 250, 200, 150, 100, 50]


SPLIT_RANGES = {
    "train": ("2018-01-01T00:00:00", "2020-12-31T18:00:00"),
    "val": ("2021-01-01T00:00:00", "2021-12-31T18:00:00"),
    "test": ("2022-01-01T00:00:00", "2022-12-31T18:00:00"),
}


THRESHOLDS = {'CIWC_1000': 2.2e-06, 'CLWC_1000': 8.4e-05, 'CRWC_1000': 7.2e-06, 'CSWC_1000': 1.1e-05, 
            'CIWC_925': 2.9e-06, 'CLWC_925': 3e-05, 'CRWC_925': 6.8e-06, 'CSWC_925': 1.2e-05, 
            'CIWC_850': 4.9e-06, 'CLWC_850': 3.8e-05, 'CRWC_850': 7.2e-06, 'CSWC_850': 1.4e-05, 
            'CIWC_700': 6.2e-06, 'CLWC_700': 3.4e-05, 'CRWC_700': 7.7e-06, 'CSWC_700': 1.6e-05, 
            'CIWC_600': 4.8e-06,'CLWC_600': 3e-05, 'CRWC_600': 7e-06, 'CSWC_600': 1.7e-05,
             'CIWC_500': 8.9e-06, 'CLWC_500': 3.2e-05, 'CRWC_500': 3e-07, 'CSWC_500': 2.2e-05, 
             'CIWC_400': 1.8e-05, 'CLWC_400': 1.1e-05, 'CRWC_400': 2e-07, 'CSWC_400': 1.9e-05, 
             'CIWC_300': 2e-05, 'CLWC_300': 6.5e-06, 'CRWC_300': 1.4e-07, 'CSWC_300': 1.3e-05, 
             'CIWC_250': 1.8e-05, 'CLWC_250': 2.6e-06, 'CRWC_250': 3.9e-08, 'CSWC_250': 9.2e-06, 
             'CIWC_200': 2e-05, 'CLWC_200': None, 'CRWC_200': 4.1e-09, 'CSWC_200': 5.8e-06, 
             'CIWC_150': 1.5e-05, 'CLWC_150': None, 'CRWC_150': None, 'CSWC_150': 2.2e-06, 
             'CIWC_100': 1.8e-06, 'CLWC_100': None, 'CRWC_100': None, 'CSWC_100': 2.4e-07, 
             'CIWC_50': 1e-07, 'CLWC_50': None, 'CRWC_50': None, 'CSWC_50': 2e-08}

# 先只考虑250hPa及以下的七层,由于云主要发生在对流层中低层，同时era5数据中，四个云要素在高层都容易出现异常极小数据，所以损失函数省略高层？


THRESHOLDS_normalized = {'CIWC_1000': 1.6526988338883148, 'CLWC_1000': 1.7888226297687304, 'CRWC_1000': 0.18939840924422907, 'CSWC_1000': 0.5299592356698657, 
                        'CIWC_925': 1.5437755791870338, 'CLWC_925': 0.4006857156957535, 'CRWC_925': 0.15573793347130765, 'CSWC_925': 0.5361342106577827, 
                        'CIWC_850': 1.4221595863413221, 'CLWC_850': 0.38681384751660675, 'CRWC_850': 0.16417015170500673, 'CSWC_850': 0.5467590101871423, 
                        'CIWC_700': 0.9753383971347047, 'CLWC_700': 0.369354356769835, 'CRWC_700': 0.2034172926721428, 'CSWC_700': 0.369893055838116, 
                        'CIWC_600': 0.4323877212052874, 'CLWC_600': 0.4064710630395145, 'CRWC_600': 0.2832732991389197, 'CSWC_600': 0.1769179080314436,
                        'CIWC_500': 0.4525095232371367, 'CLWC_500': 0.7621056268932181, 'CRWC_500': 0.20606136387582277, 'CSWC_500': 0.07860186371917027, 
                        'CIWC_400': 0.5641619520824772, 'CLWC_400': 0.6508862884757455, 'CRWC_400': 1.1346176313961287, 'CSWC_400': 0.10532727755291572, 
                        'CIWC_300': 0.4465428630268969, 'CLWC_300': 1.7537389439953481, 'CRWC_300': 17.412180209855002, 'CSWC_300': 0.15861711396207046, 
                        'CIWC_250': 0.4501636961407084, 'CLWC_250': 2.3569643817494055, 'CRWC_250': 78.17126968642975, 'CSWC_250': 0.18960736761773644, 
                        'CIWC_200': 0.7203084834912152, 'CLWC_200': None, 'CRWC_200': 1871.4227727223415, 'CSWC_200': 0.23153087397040956, 
                        'CIWC_150': 0.8623992034838112, 'CLWC_150': None, 'CRWC_150': None, 'CSWC_150': 0.18918994088979246, 
                        'CIWC_100': 0.40058825294501776, 'CLWC_100': None, 'CRWC_100': None, 'CSWC_100': 0.06463525743442416, 
                        'CIWC_50': 308.41037646064905, 'CLWC_50': None, 'CRWC_50': None, 'CSWC_50': 742.5446509683686}

# THRESHOLDS_normalized['CLWC_50'] == None   True




