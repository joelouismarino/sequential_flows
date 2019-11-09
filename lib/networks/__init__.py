from .fully_connected import FullyConnectedNetwork
from .convolutional import ConvolutionalNetwork
from .transposed_conv import TransposedConvNetwork
from .recurrent import RecurrentNetwork
from .dcgan_lstm import DCGAN_LSTM
from .custom import CustomFlowNetwork
from .rmn import ResidualMultiplicativeNetwork


def get_network(network_args):
    if network_args is None:
        return None
    network_args = network_args.copy()
    network_type = network_args.pop('type')
    network_type = network_type.lower()
    if network_type == 'fully_connected':
        return FullyConnectedNetwork(**network_args)
    elif network_type == 'convolutional':
        return ConvolutionalNetwork(**network_args)
    elif network_type == 'trans_conv':
        return TransposedConvNetwork(**network_args)
    elif network_type == 'recurrent':
        return RecurrentNetwork(**network_args)
    # can include custom network architectures here
    elif network_type == 'dcgan_lstm':
        return DCGAN_LSTM(**network_args)
    elif network_type == 'custom':
        return CustomFlowNetwork(**network_args)
    elif network_type == 'rmn':
        return ResidualMultiplicativeNetwork(**network_args)
    else:
        raise NotImplementedError
