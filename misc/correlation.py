import torch


def estimate_correlation(tensor):
    """
    Estimate the correlation between succesive frames of the tensor.

    Args:
        tensor (torch.tensor): dimensions [T, B, C, H, W]
    """
    t, b, c, h, w = tensor.shape
    x_t = tensor[:-1]
    x_t_plus_1 = tensor[1:]
    mean = tensor.reshape(-1, c, h, w).mean(dim=0)
    mean = mean.reshape(1, 1, c, h, w).repeat(t-1, b, 1, 1, 1)
    var = tensor.reshape(-1, c, h, w).var(dim=0)
    var = var.reshape(1, 1, c, h, w).repeat(t-1, b, 1, 1, 1)
    corr = (x_t - mean) * (x_t_plus_1 - mean) / var
    return corr.mean()
