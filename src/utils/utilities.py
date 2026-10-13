import numpy as np
import torch


def ceil_to_sigfig(x: float, sigfig: int = 2) -> float:
    """   通过一个很有效的方法，将小数值第二位向上取整
    将数值按有效数字向上取整。

    例如：
        8.83e-06 -> 8.9e-06
        3.12e-05 -> 3.2e-05
        2.01e-07 -> 2.1e-07
    """
    if x == 0 or not np.isfinite(x):
        return x
    x_abs = abs(x) 
    exponent = np.floor(np.log10(x_abs))  # ?    小数点第二位向上取整，为什么能这样？
    scale = 10 ** (sigfig - 1 - exponent)   #?  scale : 1d7
    return float(np.ceil(x_abs * scale) / scale)   # ? 




def set_seed(seed: int = 42) -> None:   # 设置随机种子
    np.random.seed(seed)    # 这三个随机种子，各自具体什么作用和含义？
    torch.manual_seed(seed)   # torch.manual_seed(seed) is used in PyTorch to set the seed for generating random numbers on all devices (CPU and GPU), ensuring reproducibility of experiments. It returns a torch.Generator object that can be reused for controlled randomness.  确保每次生成的随机数是一样的。
    torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子 为所有可用的 GPU 设置随机数生成器的种子，以确保多 GPU 环境下的结果可复现。
    torch.backends.cudnn.benchmark = True   # 用于启用 cuDNN 的自动优化功能。它会根据硬件和输入数据的大小，选择最优的卷积算法，从而提升模型的运行效率。；一般场景下，只要简单地在 PyTorch 程序开头将其值设置为 True，就可以大大提升卷积神经网络的运行速度。；适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。

