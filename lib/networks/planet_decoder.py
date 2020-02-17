import torch
import torch.nn as nn
from .network import Network
from ..layers import TransposedConvLayer, FullyConnectedLayer


class PlaNetDecoder(Network):
    """
    Transposed convolutional decoder from PlaNet.
    """
    def __init__(self, n_input=None, inputs=None, non_linearity='elu'):
        super(PlaNetDecoder, self).__init__(n_layers=12, inputs=inputs)
        self.fc = FullyConnectedLayer(n_input, 1024)
        self.conv1 = TransposedConvLayer(n_input=1024, n_output=128, filter_size=5,
                                         stride=2, padding=0, non_linearity=non_linearity)
        self.conv2 = TransposedConvLayer(n_input=128, n_output=64, filter_size=5,
                                         stride=2, padding=0, non_linearity=non_linearity)
        self.conv3 = TransposedConvLayer(n_input=64, n_output=32, filter_size=6,
                                         stride=2, padding=0, non_linearity=non_linearity)
        self.n_out = 32

    def forward(self, input=None, **kwargs):
        output = torch.cat([kwargs['z'], kwargs['s']], dim=1)
        output = self.fc(output)
        output = output.view(-1, 1024, 1, 1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        return output
