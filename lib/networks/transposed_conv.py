from .network import Network
from ..layers import TransposedConvLayer
from ..layers import FullyConnectedLayer


class TransposedConvNetwork(Network):
    """
    A transposed convolutional neural network.
    """
    def __init__(self, n_layers, n_input, n_units, filter_sizes, paddings=None,
                 strides=1, inputs=None, connectivity='sequential', batch_norm=False,
                 non_linearity='linear', dropout=None, last_linear=False):

        #use an extra fc layer to match the initial filter size
        # n_layers += 1

        super(TransposedConvNetwork, self).__init__(n_layers, inputs, connectivity)

        if type(n_units) == int:
            n_units = [n_units for _ in range(n_layers)]

        if type(filter_sizes) == int:
            filter_sizes = [filter_sizes for _ in range(n_layers)]

        if paddings is None:
            paddings = [int((filter_size - 1.) / 2) for filter_size in filter_sizes]

        if type(paddings) == int:
            paddings = [paddings for _ in range(n_layers)]

        if type(strides) == int:
            strides = [strides for _ in range(n_layers)]

        # n_out_fc = (filter_sizes[0]**2) * n_input
        # self.layers[0] = FullyConnectedLayer(n_input, n_out_fc, batch_norm, non_linearity, dropout)
        # self.trans_conv_init_size = [n_input, filter_sizes[0], filter_sizes[0]]

        # n_layers -= 1
        n_in = n_input
        for l in range(n_layers):
            if last_linear and l == n_layers-1:
                non_linearity = 'linear'
                batch_norm = False
            self.layers[l] = TransposedConvLayer(n_in, n_units[l], filter_sizes[l],
                                                 paddings[l], strides[l],
                                                 batch_norm, non_linearity, dropout)

            if connectivity in ['sequential', 'residual']:
                n_in = n_units[l]
            elif connectivity == 'highway':
                n_in = n_units[l]
                if l > 0:
                    self.gates[l] = TransposedConvLayer(n_in, n_units[l],
                                                        filter_sizes[l],
                                                        paddings[l], strides[l],
                                                        non_linearity='sigmoid')
            elif connectivity == 'concat':
                n_in += n_units[l]
            elif connectivity == 'concat_input':
                n_in = n_units[l] + n_input
            else:
                raise NotImplementedError

        self.n_out = n_in
