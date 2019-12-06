import torch
import torch.nn.functional as F
from torch.distributions import constraints
from .transform_module import TransformModule
import numpy as np
from .actnorm_transform import ActNormLayer

class ZeroConv2d(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super().__init__()

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = torch.nn.Parameter(torch.zeros(1, output_size, 1, 1))

    def forward(self, input):
        # out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(input)
        out = out * torch.exp(self.scale * 3)

        return out


class AdditiveCouplingTransform(TransformModule):
    """
    An invertible convolution transform.

    Args:
        input_size : number of channles in the input
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, input_size, mask_size=None):
        super(AdditiveCouplingTransform, self).__init__()
        self.input_size = input_size
        self.mask_size = mask_size

        nc = mask_size if mask_size is not None else input_size

        self.network = torch.nn.Sequential(*[torch.nn.Conv2d(nc//2, 512, 3, 1, 1),
                                             ActNormLayer(512),
                                             torch.nn.ReLU(inplace=True),
                                             torch.nn.Conv2d(512, 512, 1, 1, 0),
                                             ActNormLayer(512),
                                             torch.nn.ReLU(inplace=True),
                                             ZeroConv2d(512, nc//2, 3, 1, 1),
                                             ActNormLayer(nc//2)])


        self._ready = True


    def _call(self, x):
        """
        y = [x1, x2+NN(x1)]
        """

        if self.mask_size is not None:
            x1 = x[:,:self.mask_size,...]
            x2 = x[:,self.mask_size:,...]
            input = x1
        else:
            input = x

        z1, z2 = torch.chunk(input, 2, dim=1)
        h = self.network(z1)
        z2 += h

        if self.mask_size is not None:
            return torch.cat([z1, z2, x2], dim=1)
        else:
            return torch.cat([z1, z2], dim=1)

    def _inverse(self, y):
        """
        x = [y1, y2-NN(y1)]
        """

        if self.mask_size is not None:
            y1 = y[:,:self.mask_size,...]
            y2 = y[:,self.mask_size:,...]
            input = y1
        else:
            input = y

        z1, z2 = torch.chunk(input, 2, dim=1)
        h = self.network(z1)

        z2 -= h

        if self.mask_size is not None:
            return torch.cat([z1, z2, y2], dim=1)
        else:
            return torch.cat([z1, z2], dim=1)


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


class AffineCouplingTransform(TransformModule):
    """
    An invertible convolution transform.

    Args:
        input_size : number of channles in the input
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, input_size, mask_size=None):
        super(AffineCouplingTransform, self).__init__()
        self.input_size = input_size
        self.mask_size = mask_size

        nc = mask_size if mask_size is not None else input_size

        self.network = torch.nn.Sequential(*[torch.nn.Conv2d(nc, 512, 3, 1, 1),
                                             ActNormLayer(512),
                                             torch.nn.ReLU(inplace=True),
                                             torch.nn.Conv2d(512, 512, 1, 1, 0),
                                             ActNormLayer(512),
                                             torch.nn.ReLU(inplace=True),
                                             ZeroConv2d(512, nc, 3, 1, 1),
                                             ActNormLayer(nc)])


        self._ready = True


    def _call(self, x):
        """
        y = [x[:n/2], f(x[n/2:])]
        """

        if self.mask_size is not None:
            x1 = x[:,:self.mask_size,...]
            x2 = x[:,self.mask_size:,...]
            input = x1
        else:
            input = x

        z1, z2 = torch.chunk(input, 2, dim=1)
        h = self.network(z1)
        self.shift = h[:, 0::2]
        self.scale = F.sigmoid(h[:, 1::2] + 2.)
        z2 += self.shift
        z2 *= self.scale

        if self.mask_size is not None:
            return torch.cat([z1, z2, x2], dim=1)
        else:
            return torch.cat([z1, z2], dim=1)

    def _inverse(self, y):
        """
        x = [y[:n/2, f^(-1)(y[n/2:)]
        """

        if self.mask_size is not None:
            y1 = y[:,:self.mask_size,...]
            y2 = y[:,self.mask_size:,...]
            input = y1
        else:
            input = y

        z1, z2 = torch.chunk(input, 2, dim=1)
        h = self.network(z1)
        self.shift = h[:, 0::2]
        self.scale = F.sigmoid(h[:, 1::2] + 2.)

        z2 /= self.scale
        z2 -= self.shift

        if self.mask_size is not None:
            return torch.cat([z1, z2, y2], dim=1)
        else:
            return torch.cat([z1, z2], dim=1)


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
        return torch.cat([torch.zeros_like(self.scale), self.scale.log()], dim=1)

    def reset(self, batch_size):
        """
        Resets the invertible convolution transform.
        """
        pass

