"""Validation SSIM and FSS for M2/M3 GAN epochs and the selected M0/M1 checkpoints.

The score is Wang's Gaussian-window SSIM in the same log space as the
generator output, z = log10(1 + q / 1e-8). Temperature is not included.
Each map is one lead, one cloud variable and one pressure level.
Climatologically empty levels are left out of the average:

    CLWC at 200, 150, 100 and 50 hPa
    CRWC at 150, 100 and 50 hPa

Parameters, and where they enter the formula:
    data_range L = 5
    K1 = 0.01, K2 = 0.03
    C1 = (K1 * L)^2, C2 = (K2 * L)^2
    window = 7 x 7 Gaussian, sigma = 1.5, weights sum to 1

    SSIM = (2 mu_x mu_y + C1) (2 cov_xy + C2)
           / ((mu_x^2 + mu_y^2 + C1) (var_x + var_y + C2))

The reported epoch score is the equal-weight mean of those maps on split='val'.

FSS is the Roberts and Lean fraction skill score on the physical field
q = 1e-8 * (10**z - 1). An event is q >= 1e-5. The neighborhood is a
7 x 7 square, the same width as the SSIM window. Fractions use a uniform
kernel. The score pools every valid grid point:

    FSS = 1 - sum((p_f - p_o)^2) / sum(p_f^2 + p_o^2)

TODO:
1,SSIM和FSS指数的具体算法需要掌握。 这里的代码需要理解透彻。

"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader_v2 import ERA5CloudWindowDataset
from src.data.fixed_info import CLOUD_SHORT_NAMES, LEVELS
from src.losses.GAN_losses import split_cloud_temp
from src.models.generator_convGRU import ConvGRUUNet3DGenerator


ZARR_PATH = "/mnt/sdd/era5/sais2025/era5_china_2018_2022.zarr"
EXPERIMENTS = {
    "M0": Path(
        "/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/"
        "GAN_test19_M0_epoch80_purecharbonnier/check_points/pretrain_epoch_063.pt"
    ),
    "M1": Path(
        "/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/"
        "GAN_test19_M1_epoch80_extremeweight/check_points/pretrain_epoch_027.pt"
    ),
    "M2": Path(
        "/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/"
        "GAN_test19_M2_epoch80_GANonpretrain/check_points"
    ),
    "M3": Path(
        "/mnt/sdf/common_data/sais2025/era5_cloud_gan/logs/"
        "GAN_test19_M3_epoch80_GANonExtremeweightM1pretrain/check_points"
    ),
}
LEAD_STEPS = 12
N_CLOUD = len(CLOUD_SHORT_NAMES)
DATA_RANGE = 5.0
K1 = 0.01
K2 = 0.03
WIN_SIZE = 7
SIGMA = 1.5
EMPTY_LEVELS = {
    "CLWC": {200, 150, 100, 50},
    "CRWC": {150, 100, 50},
}
FSS_THRESHOLD = 1e-5
FSS_NEIGHBORHOOD = 7
Q0 = 1e-8


def gaussian_window(win_size: int = WIN_SIZE, sigma: float = SIGMA) -> torch.Tensor:
    """Normalized 7x7 Gaussian. Center weight is largest and the weights sum to 1."""
    coords = torch.arange(win_size, dtype=torch.float32) - (win_size // 2)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    window = kernel[:, None] * kernel[None, :]
    return (window / window.sum()).view(1, 1, win_size, win_size)


def valid_level_mask() -> torch.Tensor:
    """[n_var, n_level] is False on climatologically empty variable-level pairs."""
    mask = torch.ones(N_CLOUD, len(LEVELS), dtype=torch.bool)
    for var_id, name in enumerate(CLOUD_SHORT_NAMES):
        for level_id, level in enumerate(LEVELS):
            if level in EMPTY_LEVELS.get(name, set()):
                mask[var_id, level_id] = False
    return mask


def ssim_maps(prediction: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Mean SSIM of each 41x71 map.

    prediction, target: [N, 1, H, W] in log space.
    returns: [N]
    """
    pad = WIN_SIZE // 2
    c1 = (K1 * DATA_RANGE) ** 2
    c2 = (K2 * DATA_RANGE) ** 2

    def local_moment(image: torch.Tensor) -> torch.Tensor:
        padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")   # ?reflect是什么意思？
        return F.conv2d(padded, window)

    mu_x = local_moment(prediction)
    mu_y = local_moment(target)
    var_x = local_moment(prediction * prediction) - mu_x * mu_x
    var_y = local_moment(target * target) - mu_y * mu_y
    cov_xy = local_moment(prediction * target) - mu_x * mu_y
    var_x = var_x.clamp_min(0.0)
    var_y = var_y.clamp_min(0.0)

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    return (numerator / denominator).mean(dim=(1, 2, 3))


def uniform_window(size: int = FSS_NEIGHBORHOOD) -> torch.Tensor:
    """Uniform n x n kernel. Each entry is 1/n^2, so the convolution is a fraction."""
    return torch.full((1, 1, size, size), 1.0 / (size * size))  # torch.Size([1, 1, 7, 7]) 数值全为1/49


def inverse_log10p(values: torch.Tensor, q0: float = Q0) -> torch.Tensor:
    """log10(1 + q/q0) -> q, with negative numerical noise clipped to 0."""
    return (q0 * (torch.pow(10.0, values.float()) - 1.0)).clamp_min(0.0)


def neighborhood_fraction(binary: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Fraction of event points inside each n x n neighborhood."""
    pad = window.shape[-1] // 2
    padded = F.pad(binary, (pad, pad, pad, pad), mode="reflect")   # ?
    return F.conv2d(padded, window)


def fss_components(prediction: torch.Tensor, target: torch.Tensor, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared fraction error and reference for each map.

    prediction, target: [N, 1, H, W] physical cloud water, kg/kg.
    returns mse and ref, each [N], summed over the map.
    """
    forecast_fraction = neighborhood_fraction((prediction >= FSS_THRESHOLD).float(), window)  # window?
    observed_fraction = neighborhood_fraction((target >= FSS_THRESHOLD).float(), window)
    mse = (forecast_fraction - observed_fraction).square().sum(dim=(1, 2, 3))
    reference = (forecast_fraction.square() + observed_fraction.square()).sum(dim=(1, 2, 3))
    return mse, reference


def cloud_maps(tensor: torch.Tensor) -> torch.Tensor:
    """[B, lead*5, D, H, W] -> [B, lead, 4, D, H, W], still in log space."""
    cloud, _ = split_cloud_temp(tensor)
    batch, _, depth, height, width = cloud.shape
    return cloud.view(batch, LEAD_STEPS, N_CLOUD, depth, height, width)


def epoch_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    ssim_window: torch.Tensor,
    fss_window: torch.Tensor,
    level_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """One validation pass. SSIM stays in log space; FSS uses physical q >= 1e-5."""
    ssim_sums = torch.zeros(N_CLOUD, device=device, dtype=torch.float64)
    ssim_counts = torch.zeros(N_CLOUD, device=device, dtype=torch.float64)
    fss_mse = torch.zeros(N_CLOUD, device=device, dtype=torch.float64)
    fss_ref = torch.zeros(N_CLOUD, device=device, dtype=torch.float64)
    for batch in loader:   # 正式逐batch计算指标
        x, y, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prediction = cloud_maps(model(x))
        target = cloud_maps(y)
        batch_size, leads, n_var, depth, height, width = prediction.shape
        pred_flat = prediction.reshape(-1, 1, height, width)
        target_flat = target.reshape(-1, 1, height, width)
        scores = ssim_maps(pred_flat, target_flat, ssim_window).view(batch_size, leads, n_var, depth)
        mse, reference = fss_components(inverse_log10p(pred_flat), inverse_log10p(target_flat), fss_window)
        mse = mse.view(batch_size, leads, n_var, depth)
        reference = reference.view(batch_size, leads, n_var, depth)
        for var_id in range(n_var):
            valid_levels = level_mask[var_id]
            kept = scores[:, :, var_id, valid_levels]
            ssim_sums[var_id] += kept.sum().double()
            ssim_counts[var_id] += kept.numel()
            fss_mse[var_id] += mse[:, :, var_id, valid_levels].sum().double()
            fss_ref[var_id] += reference[:, :, var_id, valid_levels].sum().double()
    ssim_per_var = ssim_sums / ssim_counts.clamp_min(1)
    fss_per_var = 1.0 - fss_mse / fss_ref.clamp_min(1e-12)
    return {
        "ssim": float((ssim_sums.sum() / ssim_counts.sum()).cpu()),
        "fss": float((1.0 - fss_mse.sum() / fss_ref.sum().clamp_min(1e-12)).cpu()),
        **{f"ssim_{name}": float(ssim_per_var[i].cpu()) for i, name in enumerate(CLOUD_SHORT_NAMES)},
        **{f"fss_{name}": float(fss_per_var[i].cpu()) for i, name in enumerate(CLOUD_SHORT_NAMES)},
    }


def load_generator(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = ConvGRUUNet3DGenerator(
        in_channels=18,
        out_channels=5 * LEAD_STEPS,
        base_channels=64,
        input_steps=2,
        input_vars=9,
        output_vars=5,
        forecast_steps=LEAD_STEPS,
    )
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    state = saved["generator"] if isinstance(saved, dict) and "generator" in saved else saved
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def checkpoint_paths(path: Path) -> list[Path]:
    """A file is scored alone. A directory still means every gan_epoch_*.pt."""
    if path.is_file():
        return [path]
    paths = sorted(path.glob("gan_epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no gan_epoch_*.pt in {path}")
    return paths


def evaluate_experiment(
    name: str,
    checkpoint_path: Path,
    loader: DataLoader,
    ssim_window: torch.Tensor,
    fss_window: torch.Tensor,
    level_mask: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    output_path = output_dir / f"{name}_val_ssim.csv"
    done: set[int] = set()
    rows: list[dict] = []
    if output_path.exists():
        previous = pd.read_csv(output_path)
        # Rows written before FSS existed must be recomputed.
        if "fss" in previous.columns and "seconds" in previous.columns:
            done = set(int(epoch) for epoch in previous["epoch"])
            rows = previous.to_dict("records")

    for checkpoint in checkpoint_paths(checkpoint_path):
        epoch = int(checkpoint.stem.rsplit("_", 1)[-1])
        if epoch in done:
            print(f"[SKIP] {name} epoch {epoch:03d}")
            continue
        started = time.perf_counter()
        model = load_generator(checkpoint, device)
        with torch.inference_mode():
            metrics = epoch_scores(model, loader, ssim_window, fss_window, level_mask, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seconds = time.perf_counter() - started
        row = {"experiment": name, "epoch": epoch, **metrics, "seconds": seconds}
        rows = [item for item in rows if int(item["epoch"]) != epoch]
        rows.append(row)
        frame = pd.DataFrame(rows).sort_values("epoch")
        frame.to_csv(output_path, index=False)
        print(
            f"[OK] {name} epoch {epoch:03d}  val SSIM {metrics['ssim']:.6f}  "
            f"val FSS(n={FSS_NEIGHBORHOOD}, q={FSS_THRESHOLD:g}) {metrics['fss']:.6f}  "
            f"time {seconds:.1f}s"
        )
    return pd.DataFrame(rows).sort_values("epoch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute validation SSIM and FSS for selected M0-M3 checkpoints.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiments", nargs="+", default=["M2", "M3"], choices=sorted(EXPERIMENTS))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = ERA5CloudWindowDataset(
        ZARR_PATH,
        split="val",
        input_steps=2,
        forecast_steps=LEAD_STEPS,
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(
        f"[INFO] val samples: {len(dataset)}  device: {device}  "
        f"FSS neighborhood {FSS_NEIGHBORHOOD}  threshold {FSS_THRESHOLD:g} kg/kg"
    )
    ssim_window = gaussian_window().to(device)
    fss_window = uniform_window().to(device)
    level_mask = valid_level_mask().to(device)
    output_dir = Path(__file__).resolve().parent

    for name in args.experiments:
        frame = evaluate_experiment(
            name, EXPERIMENTS[name], loader, ssim_window, fss_window, level_mask, device, output_dir
        )
        best_ssim = frame.loc[frame["ssim"].idxmax()]
        best_fss = frame.loc[frame["fss"].idxmax()]
        print(
            f"[BEST] {name} SSIM epoch {int(best_ssim['epoch']):03d} {best_ssim['ssim']:.6f}  "
            f"FSS epoch {int(best_fss['epoch']):03d} {best_fss['fss']:.6f}"
        )


if __name__ == "__main__":
    main()
