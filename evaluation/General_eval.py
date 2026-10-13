"""统一完成多分位数 CSI、跨 epoch 曲线和典型个例图评估。"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]  #  /home/yx/projects/common_data/sais2025/test1/evaluation/General_eval.py
# .resolve返回绝对路径
sys.path.insert(0, str(PROJECT_ROOT))   # 将PROJECT_ROOT添加到首位？

from src.data.data_loader_v2 import make_dataloaders
from src.data.fixed_info import CLOUD_SHORT_NAMES, CLOUD_VARIABLES, LEVELS
from src.models.generator import UNet3DGenerator
from src.models.generator_convGRU import ConvGRUUNet3DGenerator


# =========================
# 运行配置
# =========================
CHECKPOINT_DIR = PROJECT_ROOT / "logs/GAN_test14_extreme_weight_epoch250/check_points"
THRESHOLD_JSON = (
    PROJECT_ROOT
    / "free_trail/data_statistics/figures/cloud_variable_statistics"
    / "cloud_variable_statistics.json"    # 选取的是一个总文件
)
ZARR_PATH = "/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr"

SPLITS = ("train", "val", "test")
PERCENTILES = (25, 30, 50, 70, 75, 90, 95, 99)
LEAD_STEPS = 12
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
CASE_LEVELS = (1000, 850, 400, 150)

COLORS = {
    "CIWC": "#5B6CFF",
    "CLWC": "#2CA25F",
    "CRWC": "#F28E2B",
    "CSWC": "#4ECDC4",
}

CSI_THRESHOLD_fixed = 1e-5


def inverse_log10p_cloud(values: torch.Tensor, q0: float = 1e-8) -> torch.Tensor:
    """将模型的 log10(1 + q/q0) 输出还原为非负物理量。"""
    return (q0 * (torch.pow(10.0, values.float()) - 1.0)).clamp_min(0.0)


def find_models(checkpoint_dir: Path) -> list[Path]:
    """读取 check_points 下全部 .pt 文件并稳定排序。"""
    models = sorted(path for path in checkpoint_dir.iterdir() if path.suffix == ".pt")
    if not models:   # 如果models列表为空？
        raise FileNotFoundError(f"没有在 {checkpoint_dir} 中找到 .pt 模型")
    return models


def load_thresholds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """返回 [Q,V,L] 原始物理量阈值和有效位置 mask。"""
    with path.open("r", encoding="utf-8") as file:
        stats = json.load(file)
    thresholds = np.full(    #np.full 是 numpy 库中的一个函数，用于创建一个填充了指定值的数组。
        (len(PERCENTILES), len(CLOUD_SHORT_NAMES), len(LEVELS)),
        np.nan,
        dtype=np.float32,
    )
    for q_id, percentile in enumerate(PERCENTILES):
        for var_id, variable in enumerate(CLOUD_SHORT_NAMES):
            for level_id, level in enumerate(LEVELS):
                value = stats[variable][str(level)].get(f"raw_p{percentile}")    # 提取的是raw
                if value is not None:
                    thresholds[q_id, var_id, level_id] = float(value)    # 这些都需要自己ide里面检验一遍
    return thresholds, np.isfinite(thresholds)    # 得到一个按照PERCENTILE，CLOUD_SHORT_NAMES，LEVELS顺序排列的，thresholds数组
# np.isfinite(thresholds) 是 numpy 库中的一个函数，用于判断一个数组中的元素是否是有限的（即不是无穷大或NaN）。
# 返回一个与thresholds形状相同的布尔数组，其中True表示该元素是有限的，False表示该元素是无穷大或NaN。
# 这个布尔数组可以用于在后续的计算中，排除掉无穷大或NaN的元素。
# 这个布尔数组可以用于在后续的计算中，排除掉无穷大或NaN的元素。

def load_generator(model_path: Path, device: torch.device) -> torch.nn.Module:
    if GENERATOR_VERSION == "unet":
        model = UNet3DGenerator(
            in_channels=18,
            out_channels=4 * LEAD_STEPS,
            base_channels=BASE_CHANNELS_G,
        )
    elif GENERATOR_VERSION == "convgru":
        model = ConvGRUUNet3DGenerator(
            in_channels=18,
            out_channels=4 * LEAD_STEPS,
            base_channels=BASE_CHANNELS_G,
            input_steps=2,
            input_vars=9,
            output_vars=4,
            forecast_steps=LEAD_STEPS,
        )
    else:
        raise ValueError(f"未知 generator_version: {GENERATOR_VERSION}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("generator", checkpoint)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"[INFO] Generator loaded from: {model_path}")
    return model


def outputs_complete(eval_dir: Path, split: str) -> bool:
    """判断一个模型、一个 split 的全部分位数结果是否已经存在。判断这个epoch地模型，所有csi评价是否都存在"""
    for percentile in PERCENTILES:
        csi_dir = eval_dir / f"csi_{split}_q{percentile}"
        for level in LEVELS:
            if not (csi_dir / f"csi_lead_var_{level}.csv").exists():
                return False
            if not (csi_dir / f"csi_lead_var_{level}hPa_curve.png").exists():
                return False
    return True


@torch.no_grad()
def calculate_all_csi(
    model: torch.nn.Module,
    loader,
    thresholds: np.ndarray,
    valid_thresholds: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """一次模型推理遍历，计算全部分位数 CSI，返回 [Q,T,V,L]。"""
    shape = (len(PERCENTILES), LEAD_STEPS, len(CLOUD_SHORT_NAMES), len(LEVELS))
    tp = torch.zeros(shape, dtype=torch.float64, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)

    threshold_tensor = torch.as_tensor(
        np.nan_to_num(thresholds, nan=np.inf),   # 将thresholds数组中的NaN替换为无穷大。? 
        dtype=torch.float32,
        device=device,
    ).view(len(PERCENTILES), 1, 1, len(CLOUD_SHORT_NAMES), len(LEVELS), 1, 1)

    for batch in loader:
        x, y, _ = batch
        x = x.to(device, non_blocking=True)  # non_blocking=True 表示异步传输。? 是否可以提高效率。
        y = y.to(device, non_blocking=True)
        prediction = model(x)
        batch_size, _, depth, height, width = prediction.shape
        prediction = prediction.view(
            batch_size, LEAD_STEPS, len(CLOUD_SHORT_NAMES), depth, height, width
        )
        y = y.view(
            batch_size, LEAD_STEPS, len(CLOUD_SHORT_NAMES), depth, height, width
        )
        prediction = inverse_log10p_cloud(prediction)
        y = inverse_log10p_cloud(y)

        # 逐分位数比较，避免同时创建 Q 份大型空间布尔张量。
        for q_id in range(len(PERCENTILES)):
            threshold = threshold_tensor[q_id]   #索引q_id的threshold_tensor数组。
            pred_event = prediction >= threshold
            obs_event = y >= threshold
            tp[q_id] += (pred_event & obs_event).sum(dim=(0, 4, 5)).double()
            fp[q_id] += (pred_event & ~obs_event).sum(dim=(0, 4, 5)).double()
            fn[q_id] += (~pred_event & obs_event).sum(dim=(0, 4, 5)).double()

    csi = tp / (tp + fp + fn + 1e-12)
    valid = torch.as_tensor(valid_thresholds, device=device).view(
        len(PERCENTILES), 1, len(CLOUD_SHORT_NAMES), len(LEVELS)
    )  # as_tensor 是 torch 库中的一个函数，用于将一个数组转换为张量。获得了一个bool型张量。valid_thresholds是之前得到的，一个按照PERCENTILE，CLOUD_SHORT_NAMES，LEVELS顺序排列的，bool型数组。
    csi = torch.where(valid, csi, torch.nan)
    return csi.cpu().numpy()


def plot_lead_curve(
    frame: pd.DataFrame,
    level: int,
    split: str,
    percentile: int,
    save_path: Path,
) -> None:
    lead_hours = frame.index.to_numpy() * 6
    plt.figure(figsize=(8.5, 5.8))
    for variable in CLOUD_SHORT_NAMES:
        plt.plot(
            lead_hours,
            frame[variable],
            marker="o",
            linewidth=2,
            color=COLORS[variable],
            label=variable,
        )
    plt.xlabel("Forecast lead time (hour)")
    plt.ylabel("CSI")
    plt.title(f"CSI at {level} hPa | {split} set | threshold P{percentile}")
    plt.xticks(lead_hours)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_model_csi(
    csi: np.ndarray,
    eval_dir: Path,
    split: str,
) -> None:
    """保存一个模型的所有分位数、层次 CSI CSV 和 lead 曲线。"""
    lead_index = pd.Index(range(1, LEAD_STEPS + 1), name="lead_step")
    for q_id, percentile in enumerate(PERCENTILES):
        csi_dir = eval_dir / f"csi_{split}_q{percentile}"
        csi_dir.mkdir(parents=True, exist_ok=True)  # parents=True 表示创建父目录。  子目录父目录一起创建
        for level_id, level in enumerate(LEVELS):
            frame = pd.DataFrame(
                csi[q_id, :, :, level_id],
                index=lead_index,
                columns=CLOUD_SHORT_NAMES,
            )
            csv_path = csi_dir / f"csi_lead_var_{level}.csv"
            figure_path = csi_dir / f"csi_lead_var_{level}hPa_curve.png"
            frame.to_csv(csv_path)
            plot_lead_curve(frame, level, split, percentile, figure_path)


def model_epoch(model_path: Path, fallback: int) -> int:
    match = re.search(r"epoch_(\d+)", model_path.stem)   # model_path.stem 是模型文件名。
    if match:
        return int(match.group(1))
    if "pretrain" in model_path.stem.lower() and not match:   # 我的pretrain model的所有epoch，这里都返回的是0？
    # 有pretrain_epoch_000.pt这种模型文件，则规避这种直接返回epoch=0的。
        return 0
    return fallback


def plot_epoch_curves(
    models: list[Path],
    split: str,
    percentile: int,
    checkpoint_dir: Path,
    overwrite: bool,
) -> None:
    """按 level/lead 绘制四个变量的 CSI 随 epoch 变化。"""
    save_dir = checkpoint_dir / f"csi_epochs_{split}_q{percentile}"
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = [model_epoch(model, index) for index, model in enumerate(models)]  # 具体怎么返回总epoch数量

    for level in LEVELS:
        level_complete = all(
            (save_dir / f"csi_epoch_level{level}_lead{lead}.csv").exists()
            and (save_dir / f"csi_epoch_level{level}_lead{lead}.png").exists()
            for lead in range(1, LEAD_STEPS + 1)     # 检验图像文件是否已经存在。
        )
        if level_complete and not overwrite:
            continue

        frames = []
        for model in models:
            csv_path = (
                model.parent
                / f"{model.stem}_evaluation"
                / f"csi_{split}_q{percentile}"
                / f"csi_lead_var_{level}.csv"
            )
            frames.append(pd.read_csv(csv_path, index_col=0))

        for lead in range(1, LEAD_STEPS + 1):
            csv_output = save_dir / f"csi_epoch_level{level}_lead{lead}.csv"
            figure_output = save_dir / f"csi_epoch_level{level}_lead{lead}.png"
            if csv_output.exists() and figure_output.exists() and not overwrite:
                continue

            values = np.stack(
                [frame.iloc[lead - 1][CLOUD_SHORT_NAMES].to_numpy() for frame in frames]
            )
            history = pd.DataFrame(
                values,
                index=pd.Index(epochs, name="epoch"),
                columns=CLOUD_SHORT_NAMES,
            )
            history.to_csv(csv_output)

            plt.figure(figsize=(14, 5.8))
            for variable in CLOUD_SHORT_NAMES:   # 写出来的代码。逻辑清晰，无懈可击。估计运行一个晚上
                plt.plot(    
                    history.index,
                    history[variable],
                    marker="o",
                    linewidth=2,
                    color=COLORS[variable],
                    label=variable,
                    markersize=0.5,   # 要比默认的小。默认是4。
                )
            plt.xlabel("Epoch")
            plt.ylabel("CSI")
            plt.title(
                f"CSI at {level} hPa, lead {lead} | {split} set | "
                f"threshold P{percentile}"
            )
            plt.ylim(0, 1)
            plt.grid(True, linestyle="--", alpha=0.35)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                figure_output,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def plot_case_figures(model_path: Path) -> None:
    """按三个日期、六个高度和全部 lead 绘制典型个例。"""
    from src.visualizations.evaluate_test3 import NormallyGANmodelEvaluation

    evaluator = NormallyGANmodelEvaluation(
        model_path=model_path,
        zarr_path=ZARR_PATH,
        lead_steps=LEAD_STEPS,
        base_channels_g=BASE_CHANNELS_G,
        generator_version=GENERATOR_VERSION,
        split="test",
        device=DEVICE,
    )
    for date in CASE_DATES:
        anchor = pd.Timestamp(date)
        anchor_str = anchor.strftime("%Y%m%d_%H%M")
        for lead in range(1, LEAD_STEPS + 1):   # ✅
            for level in CASE_LEVELS:
                figure_path = (
                    evaluator.save_dir
                    / f"test_anchor_{anchor_str}_lead{lead:02d}_{level}hPa_new2.png"
                )
                if figure_path.exists() and not OVERWRITE:
                    continue
                evaluator.visualize_one_case(
                    anchor_time=date,
                    lead_step=lead,
                    plot_level=level,
                )


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAN checkpoints with CSI curves.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(CHECKPOINT_DIR),
        help="Directory containing gan_epoch_*.pt files",
    )
    parser.add_argument(
        "--splits",
        type=str,  # 默认值为 "train,val,test"。  argparse无法加入tuple吗？ 所以需要转化为字符串。
        default=",".join(SPLITS),   #?  .join() 是字符串方法，用于将一个列表中的元素连接成一个字符串。  "train,val,test" ? 
        help="Comma-separated splits, e.g. train,test",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default=",".join(str(p) for p in PERCENTILES),    # 要输入的话写成 "25,50,75,90"形式
        help="Comma-separated CSI percentiles, e.g. 90",
    )
    parser.add_argument(
        "--generator_version",
        type=str,
        default=GENERATOR_VERSION,
        help="unet、convgru 或 simvp，须与 checkpoint 的生成器一致",
    )
    parser.add_argument(
        "--skip_cases",
        action="store_true", # 默认为False。 如果这个参数被设置，那么 skip_cases 就会被设置为 True。  
        help="Skip typical-case figure generation",
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true", # 默认为True。 如果这个参数被设置，那么 overwrite 就会被设置为 False。  
        help="Reuse existing CSI outputs instead of recomputing",
    )
    return parser.parse_args()



###############计算modelOverallCSI_lead######
# 对于每一个方案每一个epoch的模型，计算test set上所有高度，var，lat，lon上的CSI_1e-5的平均值， 这个平均值仅为lead 的函数。作为评估这个模型CSI预报的能力。
@torch.no_grad()
def calculate_csi_fixed(
    model: torch.nn.Module,
    loader,
    lead_steps: int,
    device: torch.device,
    threshold: float = CSI_THRESHOLD_fixed,
) -> np.ndarray:
    """返回 [lead, variable, level] 的 CSI，
    返回某个split下的dataloader的，所有样本的CSI平均
    阈值统一为物理量 threshold。"""
    shape = (lead_steps, len(CLOUD_SHORT_NAMES), len(LEVELS))
    tp = torch.zeros(shape, dtype=torch.float64, device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    for batch in loader:
        x, y, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prediction = model(x)
        batch_size, _, depth, height, width = prediction.shape
        view_shape = (batch_size, lead_steps, len(CLOUD_SHORT_NAMES), depth, height, width)
        prediction = inverse_log10p_cloud(prediction.view(*view_shape))
        y = inverse_log10p_cloud(y.view(*view_shape))
        pred_event = prediction >= threshold
        obs_event = y >= threshold
        tp += (pred_event & obs_event).sum(dim=(0, 4, 5)).double()   # 在batch，H，W维度上平均
        fp += (pred_event & ~obs_event).sum(dim=(0, 4, 5)).double()   # 这里为什么是sum？
        fn += (~pred_event & obs_event).sum(dim=(0, 4, 5)).double()   # 这个函数需要ide检查一遍
    return (tp / (tp + fp + fn + 1e-12)).cpu().numpy()    # 这是split所有样本下的，tp数，fp数，fn数。







def overall_csi_csv_path(eval_dir: Path, split: str, model_path: Path) -> Path:
    return eval_dir / f"modelOverallCSI_lead_{split}_{model_path.stem}.csv"    # model_path.name 是模型文件名。返回的是'gan_epoch_018.pt'形式


def save_model_overall_csi(
    csi: np.ndarray,
    eval_dir: Path,
    split: str,
    model_path: Path,
    lead_steps: int,
) -> Path:
    """csi: [lead, variable, level] -> 每个 lead 一个全高度全要素平均。"""
    eval_dir.mkdir(parents=True, exist_ok=True)
    overall = np.nanmean(csi, axis=(1, 2))
    frame = pd.DataFrame(
        {"modelOverallCSI": overall},
        index=pd.Index(range(1, lead_steps + 1), name="lead_step"),
    )
    csv_path = overall_csi_csv_path(eval_dir, split, model_path)
    frame.to_csv(csv_path)
    return csv_path


def plot_overall_csi_epoch_curves(
    models: list[Path],
    splits: tuple[str, ...],
    lead_steps: int,
    checkpoint_dir: Path,
) -> None:
    """读取各模型 CSV，按 lead 画 train/val/test 的 overall CSI–epoch 曲线。"""
    epochs = [model_epoch(model, index + 1) for index, model in enumerate(models)]
    split_colors = {"train": "#5B6CFF", "val": "#2CA25F", "test": "#F28E2B"}

    for lead in range(1, lead_steps + 1):
        history = pd.DataFrame(index=pd.Index(epochs, name="epoch"))
        for split in splits:
            values = []
            for model in models:
                csv_path = overall_csi_csv_path(
                    model.parent / f"{model.stem}_evaluation",
                    split,
                    model,
                )
                frame = pd.read_csv(csv_path, index_col=0)
                values.append(float(frame.loc[lead, "modelOverallCSI"]))
            history[split] = values

        history.to_csv(checkpoint_dir / f"modelOverallCSI_epochCurve_lead{lead}.csv")
        plt.figure(figsize=(10, 5.4))
        for split in splits:     # 这里把train，val，test的modelOverallCSI_lead画在一起。
            plt.plot(
                history.index,
                history[split],
                marker="o",
                linewidth=2,
                color=split_colors.get(split, "black"),
                label=split,
                markersize=4,
            )
        plt.xlabel("Epoch")
        plt.ylabel("modelOverallCSI")
        plt.title(f"modelOverallCSI vs epoch | lead {lead} | threshold 1e-5")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        figure_path = checkpoint_dir / f"modelOverallCSI_epochCurve_lead{lead}.png"
        plt.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[INFO] wrote {figure_path}")


def main() -> None:
    args = parse_eval_args()
    checkpoint_dir = Path(args.checkpoint_dir)   # 
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())   # 将 "train,test,split"转化为("train", "test", "split")  ???   为什么不直接导入tuple而是转化字符串？ 
    percentiles = tuple(int(p.strip()) for p in args.percentiles.split(",") if p.strip())
    overwrite = not args.no_overwrite   # 默认为True。 如果这个参数被设置，那么 overwrite 就会被设置为 False。  

    models = find_models(checkpoint_dir)
    thresholds, valid_thresholds = load_thresholds(THRESHOLD_JSON)
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

    print(f"[INFO] Found {len(models)} models in {checkpoint_dir}")

    # NOTE  这里的循环计算modelOverallCSI_lead，并且保存为csv文件。如果需要一般的evaluation，则隐藏这段循环代码
    for model_path in models:
        eval_dir = model_path.parent / f"{model_path.stem}_evaluation"   # .stem返回'gan_epoch_018'形式str。
        pending_splits = [
            split
            for split in splits
            if overwrite or not overall_csi_csv_path(eval_dir, split, model_path).exists()
        ]
        if not pending_splits:    # 如果pending_splits列表为空，则跳过。
            print(f"[SKIP] modelOverallCSI exists: {model_path.name}")
            continue
        print(f"[INFO] modelOverallCSI {model_path.name} on {pending_splits}")
        model = load_generator(model_path, device)
        for split in pending_splits:
            csi = calculate_csi_fixed(model, split_loaders[split], LEAD_STEPS, device)
            save_model_overall_csi(csi, eval_dir, split, model_path, LEAD_STEPS)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plot_overall_csi_epoch_curves(models, splits, LEAD_STEPS, checkpoint_dir)
    print("[DONE] modelOverallCSI_lead calculated")
    return     # main函数到此终止了？

    for model_path in models:
        eval_dir = model_path.parent / f"{model_path.stem}_evaluation"
        pending_splits = [
            split
            for split in splits
            if overwrite or not outputs_complete(eval_dir, split)   # 如果确认overwrite或者是没有完成。
        ]

        if pending_splits:   # 如果pending_splits列表不为空，则继续执行。
            model = load_generator(model_path, device)
            for split in pending_splits:
                print(f"[INFO] Evaluating {model_path.name} on {split}")
                csi = calculate_all_csi(
                    model,
                    split_loaders[split],
                    thresholds,
                    valid_thresholds,
                    device,
                )
                save_model_csi(csi, eval_dir, split)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            print(f"[SKIP] CSI outputs complete: {model_path.name}")

        if not args.skip_cases:
            plot_case_figures(model_path)

    for split in splits:
        for percentile in percentiles:
            plot_epoch_curves(models, split, percentile, checkpoint_dir, overwrite)

    print("[DONE] General evaluation completed")


if __name__ == "__main__":
    main()
