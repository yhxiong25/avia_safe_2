


'''
GPTprompt:   现在有
需要写一份代码，在所有epoch训练结束之时，会得到一个    的字典，包含了每个epoch中，5种损失函数随时间变化的值。
现在需要写一份利用这份字典，绘制5种损失函数的值随epoch的变化曲线。     需要给出大标题，标注训练的阶段。  要求代码要简洁简短清晰优美不需要冗长。  附带适量的注释。
对于训练集中的total_loss和验证集中的total_loss用实线，三种子损失函数用虚线。  

'''



import os
import matplotlib.pyplot as plt


def plot_loss_curves(loss_log_, train_stage, save_path=None):
    """
    绘制训练过程中各类 loss 随 epoch 的变化曲线。
    Parameters
    ----------
    loss_log_ : dict
        形如：
        {
            epoch: {
                "total_loss_train": float,
                "weighted_recon_train": float,
                "total_loss_val": float,
            }
        }

    train_stage : str
        训练阶段名称，例如 "pretrain", "gan_train", "finetune"。
        会写入图像大标题中。

    save_path : str or None
        图像保存路径。若为 None，则只显示不保存。
    """
    # 按 epoch 从小到大排序，避免字典顺序异常
    epochs = sorted(loss_log_.keys())

    total_train = [loss_log_[e]["total_loss_train"] for e in epochs]
    total_val = [loss_log_[e]["total_loss_val"] for e in epochs]
    recon_train = [loss_log_[e]["weighted_recon_train"] for e in epochs]
    temp_loss = [loss_log_[e]["temp_loss"] for e in epochs]
    if train_stage != 'pretrain':
        discriminator_loss = [loss_log_[e]["discriminator_loss"] for e in epochs]
        adv_loss_g = [loss_log_[e]["adv_loss_g"] for e in epochs]

    plt.figure(figsize=(10, 6))

    # 总损失：实线
    plt.plot(epochs, total_train, label="Train Total Loss",color='orange', linewidth=2.3)
    plt.plot(epochs, total_val, label="Val Total Loss", color='blue',linewidth=2.3)

    # 重构损失：虚线
    plt.plot(epochs, recon_train, "--", label="Train Weighted Charbonnier", color='black',linewidth=1.8)
    plt.plot(epochs, temp_loss, "--", label="temp_loss",color='green', linewidth=1.8)
    if train_stage != 'pretrain':
        plt.plot(epochs, discriminator_loss, "--", label="discriminator_loss",color='darkred', linewidth=1.8)
        plt.plot(epochs, adv_loss_g, "--", label="adv_loss_g",color='salmon', linewidth=1.8)
    plt.title(f"Loss Curves during {train_stage}", fontsize=15, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=10,loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Loss curve saved to: {save_path}")
    else:
        plt.show()

