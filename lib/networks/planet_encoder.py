import torch
import torch.nn as nn
from .network import Network
from ..layers import ConvolutionalLayer, FullyConnectedLayer


class PlaNetEncoder(Network):
    """
    Convolutional encoder from PlaNet.
    """
    def __init__(self, n_input=None, inputs=None, non_linearity='elu'):
        super(PlaNetEncoder, self).__init__(n_layers=12, inputs=inputs)
        self.conv1 = ConvolutionalLayer(n_input=3, n_output=32, filter_size=4,
                                        stride=2, padding=0, non_linearity=non_linearity)
        self.conv2 = ConvolutionalLayer(n_input=32, n_output=64, filter_size=4,
                                        stride=2, padding=0, non_linearity=non_linearity)
        self.conv3 = ConvolutionalLayer(n_input=64, n_output=128, filter_size=4,
                                        stride=2, padding=0, non_linearity=non_linearity)
        self.conv4 = ConvolutionalLayer(n_input=128, n_output=256, filter_size=4,
                                        stride=2, padding=0, non_linearity=non_linearity)
        self.fc = FullyConnectedLayer(1024 + 256, 256)
        self.n_out = 256

    def forward(self, input=None, **kwargs):
        if 'y' in self.inputs:
            output = kwargs['y']
        else:
            output = kwargs['x']
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = output.view(-1, 1024)
        output = torch.cat([output, kwargs['s']], dim=1)
        output = self.fc(output)
        return output
