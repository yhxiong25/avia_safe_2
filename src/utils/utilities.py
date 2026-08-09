import numpy as np



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





