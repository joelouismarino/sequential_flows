import torch
import torch.nn.functional as F
from torch.distributions import constraints
from .transform_module import TransformModule
import numpy as np
from .actnorm_transform import ActNormLayer
from .coupling_transform import ZeroConv2d


class SplitTransform(TransformModule):
    """
    An invertible split transform x -> [x1, x2]
    As x2 will remain the same till evaluation, we use affine transform to turn it into standard gaussian in inverse call
    [x1, x2] -> [x1, (x2 - shift(x1)) / scale(x1)]

    Args:
        input_size : number of channles in the input
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, input_size, mask_size=None):
        super(SplitTransform, self).__init__()
        self.input_size = input_size
        self.mask_size = mask_size

        if mask_size is not None:
            self.conv = ZeroConv2d(mask_size//2, mask_size, 3, 1, 1)
        else:
            self.conv = ZeroConv2d(input_size//2, input_size, 3, 1, 1)


    def _call(self, x):
        if self.mask_size is not None:
            x1 = x[:, :self.mask_size, ...]
            x2 = x[:, self.mask_size:, ...]
            input = x1
        else:
            input = x

        z1, z2 = torch.chunk(input, 2, dim=1)
        h = self.conv_zero(z1)
        shift, log_scale = torch.chunk(h, 2, dim=1)
        z2 = z2 * log_scale + shift

        if self.mask_size is not None:
            return torch.cat([z1, z2, x2])
        else:
            return torch.cat([z1, z2])

        # return x

    def _inverse(self, y):

        if self.mask_size is not None:
            y1 = y[:,:self.mask_size,...]
            y2 = y[:,self.mask_size:,...]
            input = y1
        else:
            input = y

        z1, z2 = torch.chunk(input, 2, dim=1)
        h = self.conv_zero(z1)
        shift, log_scale = torch.chunk(h, 2, dim=1)
        z2 = (z2 - shift) / log_scale

        if self.mask_size is not None:
            return torch.cat([z1, z2, y2])
        else:
            return torch.cat([z1, z2])


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
        return torch.zeros_like(x)

    def reset(self, batch_size):
        """
        Resets the invertible convolution transform.
        """
        pass

