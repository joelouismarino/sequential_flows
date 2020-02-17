import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from .transform_module import TransformModule
import numpy as np
from scipy import linalg as la
from ..networks import get_network, ConvolutionalNetwork
from ..layers import FullyConnectedLayer, ConvolutionalLayer

class ShuffleTransform(TransformModule):
    """
    An invertible convolution transform.

    Args:
        input_size : number of channles in the input
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, input_size, mask_size=None):
        super(ShuffleTransform, self).__init__()
        self.input_size = input_size
        self.mask_size = mask_size

        if mask_size is not None:
            indices = np.arange(mask_size)
        else:
            indices = np.arange(input_size)

        np.random.shuffle(indices)
        rev_indices = np.zeros_like(indices)
        for i in range(indices.size):
            rev_indices[indices[i]] = i

        indices = torch.from_numpy(indices).long()
        rev_indices = torch.from_numpy(rev_indices).long()
        self.register_buffer('indices', indices)
        self.register_buffer('rev_indices', rev_indices)
        # self.indices, self.rev_indices = indices.cuda(), rev_indices.cuda()

        self._ready = True

    def _inverse(self, y):
        """
        x = inv_shuffle(y)
        """

        if self.mask_size is not None:
            y1 = y[:,:self.mask_size,...]
            y2 = y[:,self.mask_size:,...]
            return torch.cat([y1[:,self.indices,...], y2], dim=1)

        else:
            return y[:,self.indices,...]

    def _call(self, x):
        """
        y = shuffle(x)
        """

        if self.mask_size is not None:
            x1 = x[:,:self.mask_size,...]
            x2 = x[:,self.mask_size:,...]
            return torch.cat([x1[:,self.rev_indices,...], x2], dim=1)

        else:
            return x[:,self.rev_indices,...]

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

class InvertibleConvTransform(TransformModule):
    """
    An invertible convolution transform.

    Args:
        input_size : number of channles in the input
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, input_size, mask_size=None):
        super(InvertibleConvTransform, self).__init__()
        self.input_size = input_size
        self.mask_size = mask_size

        if mask_size is None:
            weight = np.random.randn(input_size, input_size)
        else:
            weight = np.random.randn(mask_size, mask_size)

        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(w_s.abs().log())
        self.w_u = nn.Parameter(w_u)

        self._ready = True

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def _inverse(self, y):
        """
        x = inv_conv(y)
        """
        weight = self.calc_weight()

        if self.mask_size is not None:
            y1 = y[:,:self.mask_size,...]
            y2 = y[:,self.mask_size:,...]
            return torch.cat([F.conv2d(y1, weight), y2], dim=1)

        else:
            return F.conv2d(y, weight)

    def _call(self, x):
        """
        y = conv(x)
        """

        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(2).unsqueeze(3)
        if self.mask_size is not None:
            x1 = x[:, :self.mask_size, ...]
            x2 = x[:, self.mask_size:, ...]
            return torch.cat([F.conv2d(x1, weight_inv), x2], dim=1)
        else:
            return F.conv2d(x, weight_inv)

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
        # return torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)
        # return self.w_s
        # raise NotImplementedError

        output = self.w_s
        non_mask_size = output.size()
        non_mask_size[1] = self.input_size - self.mask_size
        output = torch.cat([output, torch.zeros(non_mask_size)], dim=1).cuda()

        return output

    def reset(self, batch_size):
        """
        Resets the invertible convolution transform.
        """
        pass

