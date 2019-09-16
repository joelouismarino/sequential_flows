import torch
import torch.nn as nn
from torch.distributions import constraints
from .transform_module import TransformModule
from ..networks import get_network, ConvolutionalNetwork
from ..layers import FullyConnectedLayer, ConvolutionalLayer


class AutoregressiveTransform(TransformModule):
    """
    An autoregressive transform.

    Args:
        network_config (dict): configuration for networks
        constant_scale (bool): whether to use a constant scale
        buffer_length (int): number of past images to condition on for convolutions
        cache_size (int): size of cache for storing values
    """
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, network_config, input_size, constant_scale=False, buffer_length=0):
        super(AutoregressiveTransform, self).__init__()
        self.input_size = input_size
        self.constant_scale = constant_scale

        # self.base_network = get_network(base_network_config)
        self.network_type = network_config['type']
        self.network = get_network(network_config)

        self.enc_network = get_network(network_config)

        self.initial_shift = nn.Parameter(torch.zeros([1] + input_size))
        self.initial_scale = nn.Parameter(torch.ones([1] + input_size))

        self._shift = self._scale = None
        self.buffer_length = int(buffer_length)
        self._buffer = []

        if network_config['type'] == 'dcgan_lstm':
            return

        elif network_config['type'] == 'convolutional':
            self.m = ConvolutionalLayer(self.network.n_out, input_size[0],
                                        filter_size=3, padding=1, stride=1)
            # self.m = ConvolutionalNetwork(1, self.network.n_out, input_size[0],
            #                               filter_sizes=3, paddings=1, strides=1)
            # self.g = ConvolutionalNetwork(1, self.network.n_out, input_size[0],
            #                               filter_sizes=3, paddings=1, strides=1)
            if not self.constant_scale:
                self.s = ConvolutionalLayer(self.network.n_out, input_size[0],
                                            filter_size=3, padding=1, stride=1)
        else:
            self.m = FullyConnectedLayer(self.network.n_out, input_size[0])
            if not self.constant_scale:
                self.s = FullyConnectedLayer(self.network.n_out, input_size[0])

        nn.init.constant_(self.m.linear.weight, 0.)
        nn.init.constant_(self.s.linear.weight, 0.)

        # self.shift = get_network(network_config)
        # self.initial_scale = nn.Parameter(torch.ones([1] + input_size))
        # self.constant_scale = constant_scale
        # if not self.constant_scale:
        #     self.log_scale = get_network(network_config)

    def _call(self, x):
        """
        y = scale * x + shift
        """
        return self._scale * x + self._shift

    def _inverse(self, y):
        """
        x = (y - shift) / scale
        """
        return (y - self._shift) / self._scale

    def inv(self, y):
        return self._inverse(y)

    def step(self, x):
        """
        Step the transform to the next time step.
        """
        batch_size = x.size(0)

        if self.network_type == 'dcgan_lstm':
            shift, log_scale = self.network(x)
            self._scale = log_scale.exp().clamp(max=10.)
            self._shift = shift
        else:
            self._buffer.append(x)
            if len(self._buffer) > self.buffer_length:
                self._buffer = self._buffer[-self.buffer_length:]
            input = torch.cat(self._buffer, dim=1) if self.buffer_length > 1 else x
            # input = self.network(input.sigmoid() - 0.5)
            input = self.network(input)
            m = self.m(input)
            if not self.constant_scale:
                self._scale = self.s(input).exp().clamp(max=10.)
            self._shift = m


    @property
    def sign(self):
        return self._scale.sign()

    @property
    def device(self):
        return self.initial_shift.device

    def log_abs_det_jacobian(self, x, y):
        return torch.log(self._scale)

    def reset(self, batch_size):
        """
        Resets the autoregressive transform.
        """
        # self.shift.reset()
        # if not self.constant_scale:
        #     self.log_scale.reset()
        self.network.reset()
        self._shift = self.initial_shift.repeat([batch_size] + [1 for _ in range(len(self.input_size))])
        self._scale = self.initial_scale.repeat([batch_size] + [1 for _ in range(len(self.input_size))])
        # log_scale = self.initial_scale.repeat([batch_size] + [1 for _ in range(len(self.input_size))]).log()
        # self._scale = torch.clamp(log_scale, -15, 5).exp()
        self._buffer = [torch.zeros([batch_size] + self.input_size).to(self.device) for _ in range(self.buffer_length)]

