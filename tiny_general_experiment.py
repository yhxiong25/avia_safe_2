"""Tiny 对照：加权 Charbonnier 下 ConvGRU 预训练 vs 从零 GAN。

评估阈值统一为物理量 1e-5，记录每个 epoch 在 train/val/test 上
对全部 lead / 云要素 / 高度的平均 CSI。
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.data_loader_v2 import make_dataloaders
from src.data.fixed_info import CLOUD_SHORT_NAMES, LEVELS
from src.losses.GAN_losses import CompositeGeneratorLoss
from src.models.discriminator_a1 import VoxelLocalDiscriminator
from src.models.generator import UNet3DGenerator
from src.models.generator_convGRU import ConvGRUUNet3DGenerator
from src.utils.log import log_args, setuplogging
from train_GAN_test8_a1 import (
    TrainConfig,
    inverse_log10p_cloud,
    pretrain_generator,
    set_seed,
    train_gan,
)

COLORS = {
    "CIWC": "#5B6CFF",
    "CLWC": "#2CA25F",
    "CRWC": "#F28E2B",
    "CSWC": "#4ECDC4",
}

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ZARR = "/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr"
CSI_THRESHOLD = 1e-5
SPLITS = ("train", "val", "test")


'''
nohup python tiny_general_experiment.py \
  --epochs 10 \
  --lead_steps 1 \
  --batch_size 16 \
  --device cuda:0 \
  --generator_version convgru \
  --extreme_alpha 2.0 \
  --extreme_beta 0.5 \
  --y_ref 1e-5 \
  --w_max 100 \
  > logs/tiny_general_experiment/tiny_run.log 2>&1 &
echo "PID=$!"




'''




def parse_tiny_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny CSI compare: pretrain vs GAN from scratch.")
    p.add_argument("--zarr_path", type=str, default=DEFAULT_ZARR)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lead_steps", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--lambda_adv", type=float, default=1e-3)
    p.add_argument("--base_channels_g", type=int, default=32)
    p.add_argument("--base_channels_d", type=int, default=32)
    p.add_argument("--generator_version", type=str, default="convgru")
    p.add_argument("--extreme_alpha", type=float, default=2.0)
    p.add_argument("--extreme_beta", type=float, default=0.5)
    p.add_argument("--y_ref", type=float, default=1e-5)
    p.add_argument("--w_max", type=float, default=100.0)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument(
        "--root_dir",
        type=str,
        default=str(PROJECT_ROOT / "logs" / "tiny_general_experiment"),
    )
    return p.parse_args()


def make_cfg(args: argparse.Namespace, save_dir: str, scheme: str) -> TrainConfig:
    return TrainConfig(
        zarr_path=args.zarr_path,
        train_message=f"tiny {scheme}: weighted Charbonnier + {args.generator_version}",
        save_dir=save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pretrain_epochs=args.epochs,
        gan_epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_adv=args.lambda_adv,
        base_channels_g=args.base_channels_g,
        base_channels_d=args.base_channels_d,
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,
        lead_steps=args.lead_steps,
        generator_version=args.generator_version,
        extreme_alpha=args.extreme_alpha,
        extreme_beta=args.extreme_beta,
        y_ref=args.y_ref,
        w_max=args.w_max,
        save_every_epoch=True,
    )


def build_models(cfg: TrainConfig):
    if cfg.generator_version == "unet":
        generator = UNet3DGenerator(
            in_channels=18,
            out_channels=4 * cfg.lead_steps,
            base_channels=cfg.base_channels_g,
        ).to(cfg.device)
    elif cfg.generator_version == "convgru":
        generator = ConvGRUUNet3DGenerator(
            in_channels=18,
            out_channels=4 * cfg.lead_steps,
            base_channels=cfg.base_channels_g,
            input_steps=2,
            input_vars=9,
            output_vars=4,
            forecast_steps=cfg.lead_steps,
        ).to(cfg.device)
    else:
        raise ValueError(cfg.generator_version)
    discriminator = VoxelLocalDiscriminator(
        in_channels=18 + 4 * cfg.lead_steps,
        base_channels=cfg.base_channels_d,
    ).to(cfg.device)
    return generator, discriminator


def load_generator(model_path: Path, cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    generator, _ = build_models(cfg)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state = checkpoint.get("generator", checkpoint)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    generator.load_state_dict(state, strict=True)
    generator.to(device).eval()
    return generator


def find_epoch_models(checkpoint_dir: Path) -> list[Path]:
    models = sorted(path for path in checkpoint_dir.glob("*.pt") if "epoch_" in path.stem)
    if not models:
        raise FileNotFoundError(f"没有在 {checkpoint_dir} 找到 epoch checkpoint")
    return models


def model_epoch(model_path: Path, fallback: int) -> int:
    match = re.search(r"epoch_(\d+)", model_path.stem)
    if match:
        return int(match.group(1))
    return fallback


def average_csi(csi: np.ndarray) -> float:
    return float(np.nanmean(csi))


@torch.no_grad()
def calculate_csi_fixed(
    model: torch.nn.Module,
    loader,
    lead_steps: int,
    device: torch.device,
    threshold: float = CSI_THRESHOLD,
) -> np.ndarray:
    """返回 [lead, variable, level] 的 CSI，阈值统一为物理量 threshold。"""
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
        tp += (pred_event & obs_event).sum(dim=(0, 4, 5)).double()
        fp += (pred_event & ~obs_event).sum(dim=(0, 4, 5)).double()
        fn += (~pred_event & obs_event).sum(dim=(0, 4, 5)).double()

    return (tp / (tp + fp + fn + 1e-12)).cpu().numpy()


def save_split_csi(csi: np.ndarray, eval_dir: Path, split: str, lead_steps: int) -> None:
    csi_dir = eval_dir / f"csi_{split}_thr1e-5"
    csi_dir.mkdir(parents=True, exist_ok=True)
    lead_index = pd.Index(range(1, lead_steps + 1), name="lead_step")
    for level_id, level in enumerate(LEVELS):
        pd.DataFrame(
            csi[:, :, level_id],
            index=lead_index,
            columns=CLOUD_SHORT_NAMES,
        ).to_csv(csi_dir / f"csi_lead_var_{level}.csv")


def plot_csi_epoch_curves(
    models: list[Path],
    split: str,
    lead_steps: int,
    checkpoint_dir: Path,
) -> None:
    save_dir = checkpoint_dir / f"csi_epochs_{split}_thr1e-5"
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = [model_epoch(model, index + 1) for index, model in enumerate(models)]

    for level in LEVELS:
        frames = []
        for model in models:
            csv_path = (
                model.parent
                / f"{model.stem}_evaluation"
                / f"csi_{split}_thr1e-5"
                / f"csi_lead_var_{level}.csv"
            )
            frames.append(pd.read_csv(csv_path, index_col=0))
        for lead in range(1, lead_steps + 1):
            history = pd.DataFrame(
                np.stack([frame.iloc[lead - 1][CLOUD_SHORT_NAMES].to_numpy() for frame in frames]),
                index=pd.Index(epochs, name="epoch"),
                columns=CLOUD_SHORT_NAMES,
            )
            history.to_csv(save_dir / f"csi_epoch_level{level}_lead{lead}.csv")
            plt.figure(figsize=(14, 5.8))
            for variable in CLOUD_SHORT_NAMES:
                plt.plot(
                    history.index,
                    history[variable],
                    marker="o",
                    linewidth=2,
                    color=COLORS[variable],
                    label=variable,
                    markersize=3,
                )
            plt.xlabel("Epoch")
            plt.ylabel("CSI")
            plt.title(f"CSI at {level} hPa, lead {lead} | {split} set | threshold 1e-5")
            plt.ylim(0, 1)
            plt.grid(True, linestyle="--", alpha=0.35)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f"csi_epoch_level{level}_lead{lead}.png", dpi=200, bbox_inches="tight")
            plt.close()



def evaluate_scheme(cfg: TrainConfig, loaders, device: torch.device) -> list[dict]:
    checkpoint_dir = Path(cfg.save_dir) / "check_points"
    models = find_epoch_models(checkpoint_dir)
    split_loaders = {"train": loaders.train, "val": loaders.val, "test": loaders.test}
    rows = []
    for model_path in models:
        epoch = model_epoch(model_path, 0)
        eval_dir = model_path.parent / f"{model_path.stem}_evaluation"
        model = load_generator(model_path, cfg, device)
        for split in SPLITS:
            csi = calculate_csi_fixed(model, split_loaders[split], cfg.lead_steps, device)
            save_split_csi(csi, eval_dir, split, cfg.lead_steps)
            rows.append(
                {
                    "scheme": Path(cfg.save_dir).name,
                    "model_type": cfg.generator_version,
                    "lead_steps": cfg.lead_steps,
                    "epoch": epoch,
                    "split": split,
                    "mean_csi": average_csi(csi),
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for split in SPLITS:
        plot_csi_epoch_curves(models, split, cfg.lead_steps, checkpoint_dir)
    return rows


def add_gain_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["gain_from_epoch1"] = np.nan
    frame["gain_vs_pretrain"] = np.nan
    for (scheme, split), group in frame.groupby(["scheme", "split"], sort=False):
        first = group.loc[group["epoch"].idxmin(), "mean_csi"]
        frame.loc[group.index, "gain_from_epoch1"] = group["mean_csi"] - first
    pretrain = frame[frame["scheme"].str.startswith("pretrain")].set_index(["epoch", "split"])["mean_csi"]
    for idx, row in frame.iterrows():
        key = (row["epoch"], row["split"])
        if key in pretrain.index:
            frame.at[idx, "gain_vs_pretrain"] = row["mean_csi"] - float(pretrain.loc[key])
    return frame


def train_scheme(cfg: TrainConfig, loaders, scheme: str) -> None:
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "check_points").mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    setuplogging(str(save_dir / "0.log"))
    log_args(cfg)
    logging.info(cfg.train_message)

    generator, discriminator = build_models(cfg)
    criterion = CompositeGeneratorLoss(
        latitudes=loaders.train_dataset.latitudes,
        extreme_alpha=cfg.extreme_alpha,
        extreme_beta=cfg.extreme_beta,
        y_ref=cfg.y_ref,
        w_max=cfg.w_max,
    ).to(cfg.device)

    if scheme == "pretrain":
        logging.info("Scheme A: pretrain only")
        pretrain_generator(generator, loaders, criterion, cfg)
    elif scheme == "gan":
        logging.info("Scheme B: GAN from scratch, no pretrain")
        train_gan(generator, discriminator, loaders, criterion, cfg)
    else:
        raise ValueError(scheme)   # ValueError


def main() -> None:
    args = parse_tiny_args()
    root = Path(args.root_dir)
    root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    loaders = make_dataloaders(
        args.zarr_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        forecast_steps=args.lead_steps,
        seed=args.seed,
    )
    print(f"[INFO] train={len(loaders.train_dataset)} val={len(loaders.val_dataset)} test={len(loaders.test_dataset)}")

    schemes = [
        ("pretrain", root / f"pretrain_{args.generator_version}"),
        ("gan", root / f"gan_nopretrain_{args.generator_version}"),
    ]
    ###
    all_rows = []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 这里手动运行一个个方案的exp好些
    for scheme, save_dir in schemes:
        cfg = make_cfg(args, str(save_dir), scheme)
        print(f"[INFO] start {scheme} -> {save_dir}")
        train_scheme(cfg, loaders, scheme)
        all_rows.extend(evaluate_scheme(cfg, loaders, device))

    table = add_gain_columns(pd.DataFrame(all_rows))
    csv_path = root / "csi_gain.csv"
    table.to_csv(csv_path, index=False)
    print(table.to_string(index=False))
    print(f"[DONE] wrote {csv_path}")


if __name__ == "__main__":
    main()
