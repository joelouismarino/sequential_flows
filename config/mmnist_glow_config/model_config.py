
model_type = 'glow'

################################################################################

if model_type == 'glow':

    n_blocks = 1
    cond_like_config = {'dist_config': {'dist_type': 'Glow',
                                        'n_variables': None,
                                        'base_loc_type': {'type': 'convolutional',
                                                          'n_layers': 1,
                                                          'n_units': 4**n_blocks,
                                                          'filter_sizes': 3,
                                                          'strides': 1,
                                                          'paddings': 1},
                                        'base_scale_type': {'type': 'convolutional',
                                                            'n_layers': 1,
                                                            'n_units': 4**n_blocks,
                                                            'filter_sizes': 3,
                                                            'strides': 1,
                                                            'paddings': 1},
                                        'flow_config':{'use_multi_scale': False,
                                                       'axis_transform': 'shuffle',
                                                       'couple_transform': 'additive',
                                                       'n_flows':4,
                                                       'n_blocks':n_blocks,
                                                       'input_size': 1,
                                                       'base_shape': [4**n_blocks, 64//(2**n_blocks), 64//(2**n_blocks)]},
                                        },

                         'spatial_network_config': {'inputs': ['y'],
                                                    'type': 'convolutional',
                                                    'n_input': 4,
                                                    'n_layers': 3,
                                                    'n_units': 512,
                                                    'filter_sizes': 3,
                                                    'strides': 1,
                                                    'paddings': 1,
                                                    'non_linearity': 'relu'},
                        'temporal_network_config': None,
                      }

    prior_config = approx_post_config = None


model_config = {'cond_like_config': cond_like_config,
                'prior_config': prior_config,
                'approx_post_config': approx_post_config
}

def get_model_config():
    return model_config
