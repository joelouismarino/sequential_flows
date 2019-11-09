import torch
from .network import Network
from ..layers import ConvRecurrentLayer


class ConvRecurrentNetwork(Network):
    """
    An LSTM neural network.
    """
    def __init__(self, n_layers, input_size, n_input, output_dim, kernel_size, inputs=None, connectivity='sequential',
                 dropout=None, non_linearity=None, last_linear=True):
        super(ConvRecurrentNetwork, self).__init__(n_layers, inputs, connectivity)

        input_dim = n_input

        if type(output_dim) == int:
            output_dim = [output_dim for _ in range(n_layers)]

        if type(kernel_size) == int:
            kernel_size = [kernel_size for _ in range(n_layers)]

        # n_in = n_input
        all_dims = [input_dim] + output_dim
        for l in range(n_layers):
            if last_linear and l == n_layers-1:
                non_linearity = 'linear'

            self.layers[l] = ConvRecurrentLayer(input_size, all_dims[l], all_dims[l+1], kernel_size[l], non_linearity, dropout)

            assert connectivity == 'sequential'

        self.n_out = output_dim[-1]

    @property
    def state(self):
        return torch.cat([layer.state for layer in self.layers], dim=1)

    def reset(self, batch_size):
        for layer in self.layers:
            layer.reset(batch_size)
