import torch
import torch.nn as nn
from .network import Network
from ..layers import TransposedConvLayer, ConvolutionalLayer, FullyConnectedLayer


class CustomDecoder(Network):
    """
    Custom transposed convolutional decoder.
    """
    def __init__(self, n_input, inputs=None, non_linearity='elu'):
        super(CustomDecoder, self).__init__(n_layers=12, inputs=inputs)
        self.layers[0] = FullyConnectedLayer(n_input=n_input, n_output=1024)

        self.layers[1] = TransposedConvLayer(n_input=1024, n_output=128, filter_size=6,
                                             padding=0, stride=2, non_linearity=non_linearity)
        self.layers[2] = TransposedConvLayer(n_input=128, n_output=64, filter_size=5,
                                             padding=0, stride=2, non_linearity=non_linearity)
        self.layers[3] = TransposedConvLayer(n_input=64, n_output=32, filter_size=4,
                                             padding=0, stride=2, non_linearity=non_linearity)

        self.layers[4] = ConvolutionalLayer(n_input=1024, n_output=16, filter_size=1, padding=0, stride=1)
        self.layers[5] = ConvolutionalLayer(n_input=128, n_output=16, filter_size=3, padding=0, stride=1)

        # gates
        self.layers[6] = FullyConnectedLayer(n_input=n_input, n_output=1024, non_linearity='sigmoid')

        self.layers[7] = TransposedConvLayer(n_input=1024, n_output=128, filter_size=6,
                                             padding=0, stride=2, non_linearity='sigmoid')
        self.layers[8] = TransposedConvLayer(n_input=128, n_output=64, filter_size=5,
                                             padding=0, stride=2, non_linearity='sigmoid')
        self.layers[9] = TransposedConvLayer(n_input=64, n_output=32, filter_size=4,
                                             padding=0, stride=2, non_linearity='sigmoid')

        self.layers[10] = ConvolutionalLayer(n_input=1024, n_output=16, filter_size=1, padding=0, stride=1, non_linearity='sigmoid')
        self.layers[11] = ConvolutionalLayer(n_input=128, n_output=16, filter_size=3, padding=0, stride=1, non_linearity='sigmoid')

        self.n_out = 64

    def forward(self, input=None, **kwargs):
        if input is None:
            input = torch.cat([kwargs[k] for k in self.inputs], dim=1)
        output = input
        output = self.layers[0](output)
        gate = self.layers[6](input)
        output = gate * output + (1. - gate) * output
        a = output.view(-1, 1024, 1, 1)
        b = self.layers[1](a)
        b_gate = self.layers[7](a)
        b = b_gate * b + (1. - b_gate) * b
        c = self.layers[2](b)
        c_gate = self.layers[8](b)
        c = c_gate * c + (1. - c_gate) * c
        d = self.layers[3](c)
        d_gate = self.layers[9](c)
        d = d_gate * d + (1. - d_gate) * d
        e = self.layers[4](a)
        e_gate = self.layers[10](a)
        e = e_gate * e + (1. - e_gate) * e
        e = nn.functional.interpolate(e, scale_factor=32)
        f = self.layers[5](b)
        f_gate = self.layers[11](b)
        f = f_gate * f + (1. - f_gate) * f
        f = nn.functional.interpolate(f, scale_factor=8)
        output = torch.cat((d, e, f), dim=1)
        return output
