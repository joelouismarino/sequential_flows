import torch
import torch.nn as nn
from .network import Network
from ..layers import ConvolutionalLayer, RecurrentLayer


class CustomEncoder(Network):
    """
    Custom convolutional encoder.
    """
    def __init__(self, n_input=None, inputs=None, non_linearity='elu'):
        super(CustomEncoder, self).__init__(n_layers=12, inputs=inputs)

        self.layers[0] = ConvolutionalLayer(n_input=3, n_output=32, filter_size=4,
                                            padding=0, stride=2, non_linearity=non_linearity)

        self.layers[1] = ConvolutionalLayer(n_input=32, n_output=64, filter_size=4,
                                            padding=0, stride=2, non_linearity=non_linearity)

        self.layers[2] = ConvolutionalLayer(n_input=96, n_output=128, filter_size=4,
                                            padding=0, stride=2, non_linearity=non_linearity)

        self.layers[3] = ConvolutionalLayer(n_input=192, n_output=256, filter_size=4,
                                            padding=0, stride=2, non_linearity=non_linearity)

        self.layers[4] = ConvolutionalLayer(n_input=3, n_output=32, filter_size=9, padding=0, stride=4)
        self.layers[5] = ConvolutionalLayer(n_input=32, n_output=64, filter_size=9, padding=0, stride=4)

        # gates
        self.layers[6] = ConvolutionalLayer(n_input=3, n_output=32, filter_size=4,
                                            padding=0, stride=2, non_linearity='sigmoid')

        self.layers[7] = ConvolutionalLayer(n_input=32, n_output=64, filter_size=4,
                                            padding=0, stride=2, non_linearity='sigmoid')

        self.layers[8] = ConvolutionalLayer(n_input=96, n_output=128, filter_size=4,
                                            padding=0, stride=2, non_linearity='sigmoid')

        self.layers[9] = ConvolutionalLayer(n_input=192, n_output=256, filter_size=4,
                                            padding=0, stride=2, non_linearity='sigmoid')

        self.layers[10] = ConvolutionalLayer(n_input=3, n_output=32, filter_size=9, padding=0, stride=4, non_linearity='sigmoid')
        self.layers[11] = ConvolutionalLayer(n_input=32, n_output=64, filter_size=9, padding=0, stride=4, non_linearity='sigmoid')

        self.n_out = 1024

    def forward(self, input=None, **kwargs):
        input = kwargs['y']

        a = self.layers[0](input)
        a_gate = self.layers[6](input)
        a = a_gate * a + (1. - a_gate) * a

        b = self.layers[1](a)
        b_gate = self.layers[7](a)
        b = b_gate * b + (1. - b_gate) * b

        skip_a = self.layers[4](input)
        skip_a_gate = self.layers[10](input)
        skip_a = skip_a_gate * skip_a + (1. - skip_a_gate) * skip_a

        c = self.layers[2](torch.cat((b, skip_a), dim=1))
        c_gate = self.layers[8](torch.cat((b, skip_a), dim=1))
        c = c_gate * c + (1. - c_gate) * c

        skip_b = self.layers[5](a)
        skip_b_gate = self.layers[11](a)
        skip_b = skip_b_gate * skip_b + (1. - skip_b_gate) * skip_b

        d = self.layers[3](torch.cat((c, skip_b), dim=1))
        d_gate = self.layers[9](torch.cat((c, skip_b), dim=1))
        d = d_gate * d + (1. - d_gate) * d

        return d.view(-1, 1024)
