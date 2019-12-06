import torch
import torch.nn.functional as F
from torch.distributions import constraints
from .transform_module import TransformModule
import numpy as np
from .actnorm_transform import ActNormLayer


class SqueezeTransform(TransformModule):
    """
    An invertible squeeze transform reshaping 1*2*2 volume into 4*1*1 volume

    Args:
        input_size : number of channles in the input
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, input_size):
        super(SqueezeTransform, self).__init__()
        self.input_size = input_size

        self._ready = True

    # switch the inverse and call
    def _inverse(self, y):
        b, c, h, w = y.size()

        y = y.view(b, c, h // 2, 2, w // 2, 2)
        y = y.permute(0, 1, 3, 5, 2, 4).contiguous()
        y = y.view(b, c * 2 * 2, h // 2, w // 2)

        return y

    def _call(self, x):
        b, c, h, w = x.size()

        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)

        return x


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

