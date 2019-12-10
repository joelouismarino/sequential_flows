import torch
import torch.nn as nn
from torch.distributions import constraints
from .transform_module import TransformModule
from ..networks import get_network, ConvolutionalNetwork
from ..layers import FullyConnectedLayer, ConvolutionalLayer

# def mean_dim(tensor, dim=None, keepdims=False):
#     """Take the mean along multiple dimensions.
#
#     Args:
#         tensor (torch.Tensor): Tensor of values to average.
#         dim (list): List of dimensions along which to take the mean.
#         keepdims (bool): Keep dimensions rather than squeezing.
#
#     Returns:
#         mean (torch.Tensor): New tensor of mean value(s).
#     """
#     if dim is None:
#         return tensor.mean()
#     else:
#         if isinstance(dim, int):
#             dim = [dim]
#         dim = sorted(dim)
#         for d in dim:
#             tensor = tensor.mean(dim=d, keepdim=True)
#         if not keepdims:
#             for i, d in enumerate(dim):
#                 tensor.squeeze_(d-i)
#         return tensor

class ActNormLayer(nn.Module):
    def __init__(self, input_size):
        super(ActNormLayer, self).__init__()

        self.input_size = input_size

        self.bias = nn.Parameter(torch.zeros(1, input_size, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, input_size, 1, 1))
        self._init = False

    def init_params(self, x):
        """
        parameters are to be initialized at reverse call
        """
        self._init = True

        with torch.no_grad():
            # bias = -mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
            # v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            # logs = (1. / (v.sqrt() + 1e-6)).log()

            with torch.no_grad():
                flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
                bias = (
                        flatten.mean(1)
                        .unsqueeze(1)
                        .unsqueeze(2)
                        .unsqueeze(3)
                        .permute(1, 0, 2, 3)
                )
                std = (
                        flatten.std(1)
                        .unsqueeze(1)
                        .unsqueeze(2)
                        .unsqueeze(3)
                        .permute(1, 0, 2, 3)
                )

            self.bias.data.copy_(-bias)
            self.logs.data.copy_(torch.log(1/(std+1e-6)))

    def forward(self, x, inverse=True):
        if not self._init:
            self.init_params(x)

        if inverse:
            return self.logs.exp() * (x + self.bias)
        else:
            return x * torch.exp(-self.logs) - self.bias


class ActNormTransform(TransformModule):
    """
    An activition normalization transform.

    Args:
        network_config (dict): configuration for networks
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, input_size, mask_size=None):
        super(ActNormTransform, self).__init__()

        if mask_size is None:
            self.actnorm = ActNormLayer(input_size)
        else:
            self.actnorm = ActNormLayer(mask_size)
        self._ready = True
        # self._init = False

        self.input_size = input_size
        # mask for multiscale
        self.mask_size = mask_size

    def _inverse(self, y):
        """
        x = y / exp(logs) - bias
        """

        if self.mask_size is not None:
            y1 = y[:,:self.mask_size,...]
            y2 = y[:,self.mask_size:,...]
            return torch.cat([self.actnorm(y1, inverse=True), y2], dim=1)

        else:
            return self.actnorm(y, inverse=True)

    def _call(self, x):
        """
        y = exp(logs) * (x + bias)
        """

        if self.mask_size is not None:
            x1 = x[:,:self.mask_size,...]
            x2 = x[:,self.mask_size:,...]
            return torch.cat([self.actnorm(x1, inverse=False), x2], dim=1)
        else:
            return self.actnorm(x, inverse=False)

    def inv(self, y):
        return self._inverse(y)

    def step(self, x):
        """
        Step the transform to the next time step.
        """
        pass

    def ready(self):
        return self._ready

    # @property
    # def sign(self):
    #     return self._scale.sign()

    @property
    def device(self):
        return self.logs.device

    def log_abs_det_jacobian(self, x, y):
        if self.mask_size is None:
            return -self.actnorm.logs.repeat(x.size(0), 1, x.size(2), x.size(3))
        else:
            non_mask_size = self.actnorm.logs.size()
            non_mask_size[1] = self.input_size - self.mask_size
            return torch.cat([-self.actnorm.logs.repeat(x.size(0), 1, 1, 1), torch.zeros(non_mask_size).repeat(x.size(0), 1, 1, 1)], dim=1)

    def reset(self, batch_size):
        """
        Resets the actnorm transform.
        """
        pass

