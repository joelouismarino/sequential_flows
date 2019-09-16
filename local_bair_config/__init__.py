from .data_config import data_config
from .exp_config import exp_config
from .model_config import model_config

# TODO: check if network is convolutional when setting sizes

n_channels = 3

def update_model_config():
    """
    Update model_config with variable sizes.
    """
    x_size = data_config['img_size']
    z_size = 0
    if model_config['prior_config'] is not None:
        z_size = model_config['prior_config']['dist_config']['n_variables']

    for config_name, config in model_config.items():
        if config is not None:
            # calculate the network input size
            if config['network_config'] is not None:
                input_size = 0
                for variable in config['network_config']['inputs']:
                    if variable == 'x':
                        input_size += x_size ** 2
                    elif variable == 'z':
                        input_size += z_size
                    else:
                        raise KeyError
                config['network_config']['n_input'] = input_size

            if config_name == 'cond_like_config':
                # calculate the number of variables for the conditional likelihood
                if config['network_config'] is not None:
                    network_type = config['network_config']['type']
                    if network_type in ['fully_connected', 'recurrent']:
                        config['dist_config']['n_variables'] = [x_size ** 2]
                    elif network_type == 'convolutional':
                        config['dist_config']['n_variables'] = [n_channels, x_size, x_size]
                    else:
                        raise KeyError
                else:
                    flow_type = config['dist_config']['flow_config']['network_config']['type']
                    buffer_length = config['dist_config']['flow_config']['buffer_length']
                    if flow_type in ['fully_connected', 'recurrent']:
                        config['dist_config']['n_variables'] = [int(x_size ** 2)]
                        config['dist_config']['flow_config']['input_size'] = [int(x_size ** 2)]
                        config['dist_config']['flow_config']['network_config']['n_input'] = int(buffer_length * (x_size ** 2))
                    elif flow_type == 'convolutional':
                        config['dist_config']['n_variables'] = [n_channels, x_size, x_size]
                        config['dist_config']['flow_config']['input_size'] = [n_channels, x_size, x_size]
                        config['dist_config']['flow_config']['network_config']['n_input'] = int(buffer_length * n_channels)
                    else:
                        raise KeyError


update_model_config()
