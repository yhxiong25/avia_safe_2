"""评估带温度目标的 checkpoint。

每个 lead 的输出是 4 个云要素加 1 个温度。CSI 与 modelOverallCSI 只使用前 4 个云通道，
并按原来的 log 反变换和 1e-5 阈值计算。温度不进入 CSI。
"""

from __future__ import annotations    # ？有什么用？

import importlib.util    # ? 似乎就是从别人文件加载函数的工具
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader_v2 import make_dataloaders
from src.data.fixed_info import CLOUD_SHORT_NAMES
from src.models.generator import UNet3DGenerator
from src.models.generator_convGRU import ConvGRUUNet3DGenerator
from src.models.generator_simVP import SimVP3DGenerator
from src.utils.log import setuplogging,log_args


def _load_general_eval():    # 加载母文件的函数。
    path = Path(__file__).with_name("General_eval.py")
    spec = importlib.util.spec_from_file_location("general_eval_cloud", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ge = _load_general_eval()

N_CLOUD = len(CLOUD_SHORT_NAMES)
N_TARGET = N_CLOUD + 1


def cloud_channels(tensor: torch.Tensor, lead_steps: int) -> torch.Tensor:
    """[B, lead*5, D, H, W] -> 前 4 个云通道 [B, lead, 4, D, H, W]。"""
    batch_size, _, depth, height, width = tensor.shape
    view = tensor.view(batch_size, lead_steps, N_TARGET, depth, height, width)
    return view[:, :, :N_CLOUD]


def load_generator(model_path: Path, device: torch.device, generator_version: str = ge.GENERATOR_VERSION) -> torch.nn.Module:
    if generator_version == "unet":
        model = UNet3DGenerator(
            in_channels=18,
            out_channels=N_TARGET * ge.LEAD_STEPS,
            base_channels=ge.BASE_CHANNELS_G,
        )
    elif generator_version == "convgru":
        model = ConvGRUUNet3DGenerator(
            in_channels=18,
            out_channels=N_TARGET * ge.LEAD_STEPS,
            base_channels=ge.BASE_CHANNELS_G,
            input_steps=2,
            input_vars=9,
            output_vars=N_TARGET,
            forecast_steps=ge.LEAD_STEPS,
        )
    elif generator_version == "simvp":
        model = SimVP3DGenerator(
            in_channels=18,
            out_channels=N_TARGET * ge.LEAD_STEPS,
            base_channels=ge.BASE_CHANNELS_G,
            input_steps=2,
            input_vars=9,
            output_vars=N_TARGET,
            forecast_steps=ge.LEAD_STEPS,
        )
    else:
        raise ValueError(f"未知 generator_version: {generator_version}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("generator", checkpoint)
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"[INFO] Generator loaded from: {model_path}")
    return model


@torch.no_grad()
def calculate_cloud_csi(
    model: torch.nn.Module,
    loader,
    thresholds,
    valid_thresholds,
    device: torch.device,
    lead_steps: int,
    fixed_threshold: float = ge.CSI_THRESHOLD_fixed,
):
    """一次推理同时得到分位数 CSI [Q, lead, var, level] 和固定阈值 CSI [lead, var, level]。"""
    import numpy as np

    n_percentile = len(ge.PERCENTILES)
    n_level = len(ge.LEVELS)
    tp_q = torch.zeros((n_percentile, lead_steps, N_CLOUD, n_level), dtype=torch.float64, device=device)
    fp_q = torch.zeros_like(tp_q)
    fn_q = torch.zeros_like(tp_q)
    tp_fixed = torch.zeros((lead_steps, N_CLOUD, n_level), dtype=torch.float64, device=device)
    fp_fixed = torch.zeros_like(tp_fixed)
    fn_fixed = torch.zeros_like(tp_fixed)
    threshold_tensor = torch.as_tensor(
        np.nan_to_num(thresholds, nan=np.inf),
        dtype=torch.float32,
        device=device,
    ).view(n_percentile, 1, 1, N_CLOUD, n_level, 1, 1)

    for batch in loader:
        x, y, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prediction = ge.inverse_log10p_cloud(cloud_channels(model(x), lead_steps))
        y = ge.inverse_log10p_cloud(cloud_channels(y, lead_steps))
        for q_id in range(n_percentile):
            threshold = threshold_tensor[q_id]
            pred_event = prediction >= threshold
            obs_event = y >= threshold
            tp_q[q_id] += (pred_event & obs_event).sum(dim=(0, 4, 5)).double()
            fp_q[q_id] += (pred_event & ~obs_event).sum(dim=(0, 4, 5)).double()
            fn_q[q_id] += (~pred_event & obs_event).sum(dim=(0, 4, 5)).double()
        pred_event = prediction >= fixed_threshold
        obs_event = y >= fixed_threshold
        tp_fixed += (pred_event & obs_event).sum(dim=(0, 4, 5)).double()
        fp_fixed += (pred_event & ~obs_event).sum(dim=(0, 4, 5)).double()
        fn_fixed += (~pred_event & obs_event).sum(dim=(0, 4, 5)).double()   # 这里直接融合进一起算了。 overallCSI和quantile CSI，一次性算

    csi_q = tp_q / (tp_q + fp_q + fn_q + 1e-12)
    valid = torch.as_tensor(valid_thresholds, device=device).view(n_percentile, 1, N_CLOUD, n_level)
    csi_q = torch.where(valid, csi_q, torch.nan)
    csi_fixed = tp_fixed / (tp_fixed + fp_fixed + fn_fixed + 1e-12)
    return csi_q.cpu().numpy(), csi_fixed.cpu().numpy()


def plot_case_figures(model_path: Path, generator_version: str = ge.GENERATOR_VERSION) -> None:
    """按 General_eval 的日期和高度绘制云预报个例。温度通道不参与画图。"""
    from src.visualizations.evaluate_test3 import NormallyGANmodelEvaluation

    class CloudTargetEvaluator(NormallyGANmodelEvaluation):
        def load_generator(self):
            if self._model is not None:
                return self._model
            if self.generator_version == "unet":
                model = UNet3DGenerator(
                    in_channels=18,
                    out_channels=N_TARGET * self.lead_steps,
                    base_channels=self.base_channels_g,
                ).to(self.device)
            elif self.generator_version == "convgru":
                model = ConvGRUUNet3DGenerator(
                    in_channels=18,
                    out_channels=N_TARGET * self.lead_steps,
                    base_channels=self.base_channels_g,
                    input_steps=2,
                    input_vars=9,
                    output_vars=N_TARGET,
                    forecast_steps=self.lead_steps,
                ).to(self.device)
            elif self.generator_version == "simvp":
                model = SimVP3DGenerator(
                    in_channels=18,
                    out_channels=N_TARGET * self.lead_steps,
                    base_channels=self.base_channels_g,
                    input_steps=2,
                    input_vars=9,
                    output_vars=N_TARGET,
                    forecast_steps=self.lead_steps,
                ).to(self.device)
            else:
                raise ValueError(f"未知 generator_version: {self.generator_version}")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get("generator", checkpoint)
            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._model = model
            print(f"[INFO] Generator loaded from: {self.model_path}")
            return self._model

        @torch.no_grad()
        def forecast_one_case(self, anchor_time):
            cache_key = pd.Timestamp(anchor_time).isoformat()
            if cache_key in self._forecast_cache:
                return self._forecast_cache[cache_key]
            model = self.load_generator()
            case_dict = self.get_one_case_dict(anchor_time=anchor_time)
            x_tensor = torch.from_numpy(case_dict["x_model"]).unsqueeze(0).to(self.device)
            prediction = model(x_tensor).detach().cpu().numpy()[0]
            depth, height, width = prediction.shape[1:]
            cloud = prediction.reshape(self.lead_steps, N_TARGET, depth, height, width)[:, :N_CLOUD]
            q0 = 1e-8
            cloud = np.clip((10.0 ** cloud - 1.0) * q0, 0.0, None).astype(np.float32)
            self._forecast_cache[cache_key] = cloud
            return cloud

    evaluator = CloudTargetEvaluator(
        model_path=model_path,
        zarr_path=ge.ZARR_PATH,
        lead_steps=ge.LEAD_STEPS,
        base_channels_g=ge.BASE_CHANNELS_G,
        generator_version=generator_version,
        split="test",
        device=ge.DEVICE,
    )
    for date in ge.CASE_DATES:
        anchor_str = pd.Timestamp(date).strftime("%Y%m%d_%H%M")
        for lead in range(1, ge.LEAD_STEPS + 1,5):   # 绘制1，6，11步长的图像
            for level in ge.CASE_LEVELS:
                figure_path = (
                    evaluator.save_dir
                    / f"test_anchor_{anchor_str}_lead{lead:02d}_{level}hPa_new2.png"
                )
                if figure_path.exists() and not ge.OVERWRITE:
                    continue
                evaluator.visualize_one_case(
                    anchor_time=date,
                    lead_step=lead,
                    plot_level=level,
                )




def main() -> None:
    args = ge.parse_eval_args()
    log_args(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    percentiles = tuple(int(p.strip()) for p in args.percentiles.split(",") if p.strip())
    overwrite = not args.no_overwrite    # 不设定args.no_overweite则这里为True

    models = ge.find_models(checkpoint_dir)
    thresholds, valid_thresholds = ge.load_thresholds(ge.THRESHOLD_JSON)
    device = torch.device(ge.DEVICE if torch.cuda.is_available() else "cpu")
    loaders = make_dataloaders(
        ge.ZARR_PATH,
        batch_size=ge.BATCH_SIZE,
        num_workers=0,
        max_test_samples=None,
        forecast_steps=ge.LEAD_STEPS,
        seed=42,
    )
    split_loaders = {
        "train": loaders.train,
        "val": loaders.val,
        "test": loaders.test,
    }

    print(f"[INFO] Found {len(models)} models in {checkpoint_dir}")
    print(f"[INFO] target channels per lead: {N_TARGET} (cloud CSI uses first {N_CLOUD})")

    for model_path in models:
        eval_dir = model_path.parent / f"{model_path.stem}_evaluation"
        pending_splits = []
        for split in splits:
            need_overall = overwrite or not ge.overall_csi_csv_path(eval_dir, split, model_path).exists()
            need_quantile = overwrite or not ge.outputs_complete(eval_dir, split)
            if need_overall or need_quantile:
                pending_splits.append((split, need_overall, need_quantile))
        if not pending_splits:   #？如果pending_results为空
            print(f"[SKIP] CSI outputs exist: {model_path.name}")
        else:
            print(f"[INFO] CSI {model_path.name} on {[split for split, _, _ in pending_splits]}")
            model = load_generator(model_path, device, args.generator_version)
            for split, need_overall, need_quantile in pending_splits:
                csi_q, csi_fixed = calculate_cloud_csi(
                    model,
                    split_loaders[split],
                    thresholds,
                    valid_thresholds,
                    device,
                    ge.LEAD_STEPS,
                )
                if need_quantile:
                    ge.save_model_csi(csi_q, eval_dir, split)
                if need_overall:
                    ge.save_model_overall_csi(csi_fixed, eval_dir, split, model_path, ge.LEAD_STEPS)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not args.skip_cases:
            print(f"[INFO] case figures {model_path.name}")
            plot_case_figures(model_path, args.generator_version)

    ge.plot_overall_csi_epoch_curves(models, splits, ge.LEAD_STEPS, checkpoint_dir)
    for split in splits:
        for percentile in percentiles:
            ge.plot_epoch_curves(models, split, percentile, checkpoint_dir, overwrite)
    print("[DONE] General evaluation completed")


if __name__ == "__main__":
    main()
