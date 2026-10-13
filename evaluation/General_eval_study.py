

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# PROJECT_ROOT = Path(__file__).resolve().parents[1]  #  /home/yx/projects/common_data/sais2025/test1/evaluation/General_eval.py
# Path(__file__) :  /home/yx/projects/common_data/sais2025/test1/evaluation/General_eval.py
# .resolve返回绝对路径

PROJECT_ROOT = Path('/mnt/sdf/common_data/sais2025/era5_cloud_gan')
# parents[1]什么含义？   parents[0]返回当前文件目录；parents[0]返回当前文件目录的父目录。 parents[2]同理


sys.path.insert(0, str(PROJECT_ROOT))   # 将PROJECT_ROOT添加到首位？✅

from src.data.data_loader_v2 import make_dataloaders
from src.data.fixed_info import CLOUD_SHORT_NAMES, CLOUD_VARIABLES, LEVELS
from src.models.generator import UNet3DGenerator
from src.models.generator_convGRU import ConvGRUUNet3DGenerator

CHECKPOINT_DIR = PROJECT_ROOT / "logs/GAN_test14_extreme_weight_epoch250/check_points"
THRESHOLD_JSON = (
    PROJECT_ROOT
    / "free_trail/data_statistics/figures/cloud_variable_statistics"
    / "cloud_variable_statistics.json"    # 选取的是一个总文件
)
ZARR_PATH = "/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr"

SPLITS = ("train", "val", "test")
PERCENTILES = (25, 30, 50, 70, 75, 90, 95, 99)
LEAD_STEPS = 1
BASE_CHANNELS_G = 64
GENERATOR_VERSION = "convgru"
BATCH_SIZE = 4
DEVICE = "cuda:0"
OVERWRITE = True

CASE_DATES = pd.date_range(
    start="2022-01-15",
    end="2022-12-15",
    freq=pd.DateOffset(months=5),
).strftime("%Y-%m-%d").tolist()
CASE_LEVELS = (1000, 850, 500, 400, 150)

COLORS = {
    "CIWC": "#5B6CFF",
    "CLWC": "#2CA25F",
    "CRWC": "#F28E2B",
    "CSWC": "#4ECDC4",
}


####

checkpoint_dir = Path(CHECKPOINT_DIR)

models = sorted(path for path in checkpoint_dir.iterdir() if path.suffix == ".pt")

thresholds, valid_thresholds = load_thresholds(THRESHOLD_JSON)
'''
>>> thresholds.shape
(8, 4, 13)
>>> 
>>> valid_thresholds.shape
(8, 4, 13)    # 8是8个分位数的意思
'''


device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
loaders = make_dataloaders(
    ZARR_PATH,
    batch_size=BATCH_SIZE,
    num_workers=0,
    max_test_samples=None,
    forecast_steps=LEAD_STEPS,
    seed=42,
    target_variables=CLOUD_VARIABLES,
)

split_loaders = {
    "train": loaders.train,
    "val": loaders.val,
    "test": loaders.test,
}











