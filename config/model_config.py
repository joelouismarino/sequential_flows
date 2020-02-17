
model_type = 'flow'

################################################################################

if model_type == 'flow':

    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'sigmoid_loc': False,
                                        'constant_scale': True,
                                        'flow_config': {'n_transforms': 1,
                                                        'buffer_length': 3,
                                                        'constant_scale': False,
                                                        'network_config': {'type': 'convolutional',
                                                                           'n_layers': 2,
                                                                           'n_units': 32,
                                                                           'filter_sizes': 3,
                                                                           'non_linearity': 'elu',
                                                                           'batch_norm': False,
                                                                           'connectivity': 'highway'}
                                                        }},
                        'spatial_network_config': None,
                        'temporal_network_config': None
                      }

    prior_config = approx_post_config = None

################################################################################
if model_type == 'latent':
    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'sigmoid_loc': True,
                                        'constant_scale': True,
                                        'flow_config': {'n_transforms': 1,
                                                        'type': 'convolutional',
                                                        'n_layers': 1,
                                                        'n_channels': 32,
                                                        'non_linearity': 'elu',
                                                        'connectivity': 'sequential'}},
                         'spatial_network_config': {'inputs': ['z'],
                                            'type': 'fully_connected',
                                            'n_layers': 2,
                                            'n_units': 256,
                                            'non_linearity': 'elu',
                                            'connectivity': 'sequential'},
                         'temporal_network_config': None
                      }

    latent_dim = 256

    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': latent_dim,
                                    'sigmoid_loc': False,
                                    'constant_scale': False},
                    'spatial_network_config': {'inputs': ['z'],
                                       'type': 'fully_connected',
                                       'n_layers': 2,
                                       'n_units': 256,
                                       'non_linearity': 'relu',
                                       'connectivity': 'sequential'},
                    'temporal_network_config': None
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': latent_dim,
                                          'sigmoid_loc': False,
                                          'constant_scale': False},
                          'spatial_network_config': {'inputs': ['x'],
                                             'type': 'fully_connected',
                                             'n_layers': 2,
                                             'n_units': 256,
                                             'non_linearity': 'relu',
                                             'connectivity': 'sequential'},
                          'temporal_network_config': None
                          }

################################################################################

if model_type == 'latent_flow':
    pass

model_config = {'cond_like_config': cond_like_config,
                'prior_config': prior_config,
                'approx_post_config': approx_post_config
}

def get_model_config():
    return model_config
