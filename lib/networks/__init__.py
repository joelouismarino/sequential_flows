from .fully_connected import FullyConnectedNetwork
from .convolutional import ConvolutionalNetwork
from .recurrent import RecurrentNetwork


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
    elif network_type == 'recurrent':
        return RecurrentNetwork(**network_args)
    # can include custom network architectures here
    else:
        raise NotImplementedError
