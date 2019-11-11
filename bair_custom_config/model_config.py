
model_type = 'latent_conv_recurrent_flow'

################################################################################


if model_type == 'latent_conv_recurrent_flow':
    latent_dim = 256


    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': {'type': 'trans_conv',
                                                          'n_layers': 1,
                                                          'n_units': [3],
                                                          'filter_sizes': 4,
                                                          'strides': 2,
                                                          'paddings': 1,
                                                          'connectivity': 'sequential'},
                                        'base_scale_type': {'type': 'trans_conv',
                                                            'n_layers': 1,
                                                            'n_units': [3],
                                                            'filter_sizes': 4,
                                                            'strides': 2,
                                                            'paddings': 1,
                                                            'connectivity': 'sequential'},
                                        'transform_config': {'sigmoid_last': False,
                                                             'n_transforms': 1},
                                        'flow_config': {'buffer_length': 3,
                                                        'constant_scale': False,
                                                        'network_config': {'type': 'custom',
                                                                           'c_in': 3,}
                                                        }},
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'custom_decoder',
                                                    },
                         'temporal_network_config': None
                      }



    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'base_loc_type': {'type': 'fully_connected',
                                                      'n_layers': 1,
                                                      'n_units': latent_dim},
                                    'base_scale_type': {'type': 'fully_connected',
                                                        'n_layers': 1,
                                                        'n_units': latent_dim}, },
                    'spatial_network_config': {'inputs': ['z'],
                                               'type': 'fully_connected',
                                               'n_layers': 2,
                                               'n_units': 256,
                                               'non_linearity': 'elu',
                                               'connectivity': 'highway'},
                    'temporal_network_config': {'inputs': ['spatial_output'],
                                                'type': 'recurrent',
                                                'n_layers': 1,
                                                'n_units': 256,
                                                'non_linearity': 'tanh',
                                                'connectivity': 'sequential'}
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': [latent_dim],
                                          'base_loc_type': {'type': 'fully_connected',
                                                            'n_layers': 1,
                                                            'n_units': latent_dim},
                                          'base_scale_type': {'type': 'fully_connected',
                                                              'n_layers': 1,
                                                              'n_units': latent_dim}, },
                          'spatial_network_config': {'inputs': ['y'],
                                                     'type': 'custom_encoder',
                                                     },
                          'temporal_network_config': {'inputs': ['spatial_output', 'z'],
                                                      'type': 'recurrent',
                                                      'n_layers': 1,
                                                      'n_units': 256,
                                                      'non_linearity': 'tanh',
                                                      'connectivity': 'sequential'}
                          }


################################################################################


model_config = {'cond_like_config': cond_like_config,
                'prior_config': prior_config,
                'approx_post_config': approx_post_config
}
