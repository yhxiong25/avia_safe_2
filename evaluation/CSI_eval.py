import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.fixed_info import CLOUD_SHORT_NAMES, LEVELS


COLORS = {
    "CIWC": "#5B6CFF",
    "CLWC": "#2CA25F",
    "CRWC": "#F28E2B",
    "CSWC": "#4ECDC4",
}


def load_csi_history(base_path: Path, split: str, gan_epochs: int = 70) -> np.ndarray:
    """读取预训练及各 GAN epoch 的 CSI，返回 [epoch, lead, var, level]。"""
    dir_list = ["best_pretrain_generator_evaluation"]
    dir_list.extend(
        f"gan_epoch_{epoch:03d}_evaluation"
        for epoch in range(1, gan_epochs + 1)
    )

    arr_final = []
    for model_dir in dir_list:
        csi_dir = base_path / model_dir / f"csi_{split}"
        csv_list = [name for name in os.listdir(csi_dir) if name.endswith(".csv")]
        csv_list.sort(
            key=lambda name: int(name.split("_")[-1].split(".")[0]),
            reverse=True,
        )

        level_values = [
            pd.read_csv(csi_dir / filename, index_col=0).values
            for filename in csv_list
        ]
        # [level, lead, var] -> [lead, var, level]
        arr_final.append(np.transpose(np.stack(level_values), axes=(1, 2, 0)))

    return np.stack(arr_final)


def plot_csi_epoch_curves(
    csi_history: np.ndarray,
    lead: int,
    split: str,
    save_path: Path,
) -> None:
    """绘制指定 lead time 在各高度层上的 CSI-epoch 曲线。"""
    lead_count = csi_history.shape[1]
    if not 1 <= lead <= lead_count:
        raise ValueError(f"lead 应在 1 到 {lead_count} 之间，当前为 {lead}")

    save_path.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(csi_history.shape[0])

    for level_id, level in enumerate(LEVELS):
        values = csi_history[:, lead - 1, :, level_id]
        df = pd.DataFrame(values, index=epochs, columns=CLOUD_SHORT_NAMES)

        plt.figure(figsize=(14, 5.8))
        for name in CLOUD_SHORT_NAMES:
            plt.plot(
                df.index,
                df[name],
                marker="o",
                linewidth=2.0,
                color=COLORS[name],
                label=name,
            )
        plt.xlabel("epochs", fontsize=12)
        plt.ylabel("CSI", fontsize=12)
        plt.title(
            f"csi_{level}_lead_{lead} vs epochs | {split} set",
            fontsize=12,
            fontweight="bold",
        )
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.xticks(epochs)
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()

        fig_path = save_path / f"csi_epoch_level{level}_lead{lead}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] csi figure saved to: {fig_path}")


if __name__ == "__main__":
    split = "val"
    lead = 1
    base_path = Path(
        "/home/yx/projects/common_data/sais2025/era5_cloud_gan/"
        "logs/GAN_test13_a1_lead1/check_points"
    )

    csi_history = load_csi_history(base_path, split, gan_epochs=70)
    plot_csi_epoch_curves(
        csi_history,
        lead=lead,
        split=split,
        save_path=base_path / f"csi_epochs_{split}",
    )
