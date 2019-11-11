
model_type = 'latent_custom'

if model_type == 'latent_custom':
    latent_dim = 256


    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'base_loc_type': {'type': 'trans_conv',
                                                          'n_layers': 1,
                                                          'n_units': 1,
                                                          'filter_sizes': 4,
                                                          'strides': 2,
                                                          'paddings': 1,
                                                          'connectivity': 'sequential',
                                                          'non_linearity':'sigmoid'},
                                        'base_scale_type': {'type': 'trans_conv',
                                                            'n_layers': 1,
                                                            'n_units': 1,
                                                            'filter_sizes': 4,
                                                            'strides': 2,
                                                            'paddings': 1,
                                                            'connectivity': 'sequential',
                                                            'last_linear': True},
                                        },
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'trans_conv',
                                                    'n_layers': 4,
                                                    'n_units': [512, 256, 128, 64],
                                                    'filter_sizes': 4,
                                                    'strides': [1, 2, 2, 2],
                                                    'paddings': [0, 1, 1, 1],
                                                    'non_linearity': 'leaky_relu',
                                                    'connectivity': 'sequential',
                                                    'batch_norm': True},
                         'temporal_network_config': None
                      }



    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'base_loc_type': {'type': 'fully_connected',
                                                      'n_layers': 1,
                                                      'n_units': latent_dim},
                                    'base_scale_type': {'type': 'fully_connected',
                                                        'n_layers': 1,
                                                        'n_units': latent_dim},},
                    'spatial_network_config': {'inputs': ['z'],
                                               'type': 'fully_connected',
                                               'n_layers': 2,
                                               'n_units': 256,
                                               'non_linearity': 'relu',
                                               'connectivity': 'sequential'},
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
                          'spatial_network_config': {'inputs': ['x'],
                                                     'type': 'convolutional',
                                                     'n_layers': 5,
                                                     'n_units': [64, 128, 256, 512, 128],
                                                     'filter_sizes': 4,
                                                     'strides': [2, 2, 2, 2, 1],
                                                     'paddings': [1, 1, 1, 1, 0],
                                                     'non_linearity': 'leaky_relu',
                                                     'connectivity': 'sequential',
                                                     'batch_norm': True},
                          'temporal_network_config': {'inputs': ['spatial_output'],
                                                      'type': 'recurrent',
                                                      'n_layers': 1,
                                                      'n_units': 256,
                                                      'non_linearity': 'tanh',
                                                      'connectivity': 'sequential'}
                          }

################################################################################
if model_type == 'last_frame_baseline':

    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': 'constant',
                                        'base_scale_type': 'global',
                                        'transform_config':{'sigmoid_last':False,
                                                            'n_transforms':1},
                                        'flow_config': {'buffer_length': 3,
                                                        'constant_scale': False,
                                                        'network_config': {'type': 'last_frame'}
                                                        },
                                        'learnable':True},

                        'spatial_network_config': None,
                        'temporal_network_config': None,
                      }

    prior_config = approx_post_config = None



if model_type == 'latent_conv_fp':

    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': 'global',
                                        'base_scale_type': 'global',
                                        'transform_config':{'sigmoid_last':False,
                                                            'n_transforms':1},
                                        'flow_config': {'buffer_length': 3,
                                                        'constant_scale': False,
                                                        'network_config': {'type': 'convolutional',
                                                                           'n_layers': 6,
                                                                           'n_units': 64,
                                                                           'filter_sizes': 3,
                                                                           'non_linearity': 'elu',
                                                                           'batch_norm': True,
                                                                           'connectivity': 'highway'}
                                                        }},
                        'spatial_network_config': None,
                        'temporal_network_config': None,
                      }

    prior_config = approx_post_config = None


# use conv-lstm for generating latent distiribution for noise
if model_type == 'latent_clstm':
    latent_dim = 64


    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'base_loc_type': {'type': 'trans_conv',
                                                          'n_layers': 1,
                                                          'n_units': 1,
                                                          'filter_sizes': 4,
                                                          'strides': 2,
                                                          'paddings': 1,
                                                          'connectivity': 'sequential',
                                                          'non_linearity':'sigmoid'},
                                        'base_scale_type': {'type': 'trans_conv',
                                                            'n_layers': 1,
                                                            'n_units': 1,
                                                            'filter_sizes': 4,
                                                            'strides': 2,
                                                            'paddings': 1,
                                                            'connectivity': 'sequential',
                                                            'last_linear': True},
                                        },
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'trans_conv',
                                                    'n_layers': 3,
                                                    'n_units': [256, 128, 64],
                                                    'filter_sizes': 4,
                                                    'strides': [2, 2, 2],
                                                    'paddings': [1, 1, 1],
                                                    'non_linearity': 'leaky_relu',
                                                    'connectivity': 'sequential',
                                                    'batch_norm': True},
                         'temporal_network_config': None
                      }



    prior_config = {'latent_size': [latent_dim, 4, 4],
                    'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'base_loc_type': {'type': 'convolutional',
                                                              'n_layers': 1,
                                                              'n_units': latent_dim,
                                                              'filter_sizes': 3,
                                                              'strides': 1,
                                                              'paddings': 1,
                                                              'connectivity': 'sequential',
                                                              'last_linear': True},
                                    'base_scale_type': {'type': 'convolutional',
                                                                'n_layers': 1,
                                                                'n_units': latent_dim,
                                                                'filter_sizes': 3,
                                                                'strides': 1,
                                                                'paddings': 1,
                                                                'connectivity': 'sequential',
                                                                'last_linear': True},},
                    'spatial_network_config': {'inputs': ['z'],
                                               'type': 'convolutional',
                                               'n_input': latent_dim,
                                               'n_layers': 1,
                                               'n_units': 256,
                                               'filter_sizes': 3,
                                               'strides': 1,
                                               'paddings': 1,
                                               'non_linearity': 'elu',
                                               'connectivity': 'sequential'},
                    'temporal_network_config': {'inputs': ['spatial_output'],
                                                'type': 'conv_recurrent',
                                                'n_layers': 1,
                                                'input_size': 4,
                                                # 'n_input' : 512,
                                                'output_dim': latent_dim,
                                                'kernel_size': 3,
                                                'non_linearity': 'tanh',
                                                'connectivity': 'sequential',
                                                'last_linear':True}
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': [latent_dim],
                                          'base_loc_type': {'type': 'convolutional',
                                                                    'n_layers': 1,
                                                                    'n_units': latent_dim,
                                                                    'filter_sizes': 3,
                                                                    'strides': 1,
                                                                    'paddings': 1,
                                                                    'connectivity': 'sequential',
                                                                    'last_linear': True},
                                          'base_scale_type': {'type': 'convolutional',
                                                                      'n_layers': 1,
                                                                      'n_units': latent_dim,
                                                                      'filter_sizes': 3,
                                                                      'strides': 1,
                                                                      'paddings': 1,
                                                                      'connectivity': 'sequential',
                                                                      'last_linear': True},},
                          'spatial_network_config': {'inputs': ['x'],
                                                     'type': 'convolutional',
                                                     'n_layers': 4,
                                                     'n_units': [64, 128, 256, 512],
                                                     'filter_sizes': 4,
                                                     'strides': [2, 2, 2, 2],
                                                     'paddings': [1, 1, 1, 1],
                                                     'non_linearity': 'leaky_relu',
                                                     'connectivity': 'sequential',
                                                     'batch_norm': True},
                          'temporal_network_config': {'inputs': ['spatial_output', 'z'],
                                                      'type': 'conv_recurrent',
                                                      'n_layers': 1,
                                                      'input_size': 4,
                                                      # 'n_input': 512+512,
                                                      'output_dim': latent_dim,
                                                      'kernel_size': 3,
                                                      'non_linearity': 'tanh',
                                                      'connectivity': 'sequential',
                                                      'last_linear': True}
                          }


if model_type == 'custom_flow':

    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': 'global',
                                        'base_scale_type': 'global',
                                        'transform_config':{'sigmoid_last':False,
                                                            'n_transforms':1},
                                        'flow_config': {'buffer_length': 3,
                                                        'constant_scale': False,
                                                        'network_config': {'type': 'custom',
                                                                           'c_in': 1}
                                                        }},
                        'spatial_network_config': None,
                        'temporal_network_config': None,
                      }

    prior_config = approx_post_config = None


if model_type == 'rmn_flow':

    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': 'constant',
                                        'base_scale_type': 'constant',
                                        'transform_config':{'sigmoid_last':False,
                                                            'n_transforms':2},
                                        'flow_config': {'buffer_length': 1,
                                                        'constant_scale': False,
                                                        'init_buffer': False,
                                                        'network_config': {'type': 'rmn'}
                                                        }},
                        'spatial_network_config': None,
                        'temporal_network_config': None,
                      }

    prior_config = approx_post_config = None


################################################################################

if model_type == 'latent':
    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'sigmoid_loc': True,
                                        'constant_loc': False,
                                        'constant_scale': True,
                                        },
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'fully_connected',
                                                    'n_layers': 3,
                                                    'n_units': 512,
                                                    'non_linearity': 'relu',
                                                    'connectivity': 'sequential'},
                         'temporal_network_config': None
                      }

    latent_dim = 128

    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'sigmoid_loc': False,
                                    'constant_loc': False,
                                    'constant_scale': False},
                    'spatial_network_config': {'inputs': ['z'],
                                               'type': 'fully_connected',
                                               'n_layers': 3,
                                               'n_units': 512,
                                               'non_linearity': 'relu',
                                               'connectivity': 'sequential'},
                    'temporal_network_config': None
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': [latent_dim],
                                          'sigmoid_loc': False,
                                          'constant_loc': False,
                                          'constant_scale': False},
                          'spatial_network_config': {'inputs': ['x'],
                                                     'type': 'fully_connected',
                                                     'n_layers': 3,
                                                     'n_units': 512,
                                                     'non_linearity': 'relu',
                                                     'connectivity': 'sequential'},
                          'temporal_network_config': None
                          }


if model_type == 'latent_recurrent':
    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'sigmoid_loc': True,
                                        'constant_loc': False,
                                        'constant_base': True,
                                        },
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'fully_connected',
                                                    'n_layers': 3,
                                                    'n_units': 512,
                                                    'non_linearity': 'relu',
                                                    'connectivity': 'sequential'},
                         'temporal_network_config': None
                      }

    latent_dim = 128

    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'sigmoid_loc': False,
                                    'constant_base': False},
                    'spatial_network_config': {'inputs': ['z'],
                                               'type': 'fully_connected',
                                               'n_layers': 3,
                                               'n_units': 512,
                                               'non_linearity': 'relu',
                                               'connectivity': 'sequential'},
                    'temporal_network_config': {'inputs': ['spatial_output'],
                                                'type': 'recurrent',
                                                'n_layers': 1,
                                                'n_units': 256,
                                                'non_linearity': 'relu',
                                                'connectivity': 'sequential'}
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': [latent_dim],
                                          'sigmoid_loc': False,
                                          'constant_base': False},
                          'spatial_network_config': {'inputs': ['x'],
                                                     'type': 'fully_connected',
                                                     'n_layers': 3,
                                                     'n_units': 512,
                                                     'non_linearity': 'relu',
                                                     'connectivity': 'sequential'},
                          'temporal_network_config': {'inputs': ['spatial_output'],
                                                      'type': 'recurrent',
                                                      'n_layers': 1,
                                                      'n_units': 256,
                                                      'non_linearity': 'relu',
                                                      'connectivity': 'sequential'}
                          }


if model_type == 'latent_conv_fp':
    latent_dim = 20


    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'base_loc_type': {'type': 'trans_conv',
                                                          'n_layers': 1,
                                                          'n_units': 1,
                                                          'filter_sizes': 4,
                                                          'strides': 2,
                                                          'paddings': 1,
                                                          'connectivity': 'sequential',
                                                          'non_linearity':'sigmoid'},
                                        # 'base_scale_type': {'type': 'trans_conv',
                                        #                     'n_layers': 1,
                                        #                     'n_units': 1,
                                        #                     'filter_sizes': 4,
                                        #                     'strides': 2,
                                        #                     'paddings': 1,
                                        #                     'connectivity': 'sequential',
                                        #                     'last_linear': True},
                                        'base_scale_type': 'constant'
                                        },
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'trans_conv',
                                                    'n_layers': 4,
                                                    'n_units': [512, 256, 128, 64],
                                                    'filter_sizes': 4,
                                                    'strides': [1, 2, 2, 2],
                                                    'paddings': [0, 1, 1, 1],
                                                    'non_linearity': 'leaky_relu',
                                                    'connectivity': 'sequential',
                                                    'batch_norm': True},
                         'temporal_network_config': None
                      }



    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'sigmoid_loc': False,
                                    'base_loc_type': 'constant',
                                    'base_scale_type': 'constant'},
                    'spatial_network_config': None,
                    'temporal_network_config': None
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': [latent_dim],
                                          'sigmoid_loc': False,
                                          'base_loc_type': {'type': 'fully_connected',
                                                            'n_layers': 1,
                                                            'n_units': latent_dim},
                                          'base_scale_type': {'type': 'fully_connected',
                                                              'n_layers': 1,
                                                              'n_units': latent_dim},
                                          },
                          'spatial_network_config': {'inputs': ['x'],
                                                     'type': 'convolutional',
                                                     'n_layers': 5,
                                                     'n_units': [64, 128, 256, 512, 128],
                                                     'filter_sizes': 4,
                                                     'strides': [2, 2, 2, 2, 1],
                                                     'paddings': [1, 1, 1, 1, 0],
                                                     'non_linearity': 'leaky_relu',
                                                     'connectivity': 'sequential',
                                                     'batch_norm': True},
                          'temporal_network_config': None
                          }


if model_type == 'latent_conv':
    latent_dim = 256


    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'sigmoid_loc': True,
                                        'constant_loc': False,
                                        'constant_scale': True,
                                        },
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'trans_conv',
                                                    'n_layers': 5,
                                                    'n_units': [256, 128, 64, 32, 16],
                                                    'filter_sizes': 4,
                                                    'strides': [1, 2, 2, 2, 2],
                                                    'paddings': [0, 1, 1, 1, 1],
                                                    'non_linearity': 'relu',
                                                    'connectivity': 'sequential'},
                         'temporal_network_config': None
                      }



    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'sigmoid_loc': False,
                                    'constant_loc': False,
                                    'constant_scale': False,},
                    'spatial_network_config': {'inputs': ['z'],
                                               'type': 'fully_connected',
                                               'n_layers': 2,
                                               'n_units': 256,
                                               'non_linearity': 'relu',
                                               'connectivity': 'sequential'},
                    'temporal_network_config': None
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': [latent_dim],
                                          'sigmoid_loc': False,
                                          'constant_loc': False,
                                          'constant_scale': False},
                          'spatial_network_config': {'inputs': ['x'],
                                                     'type': 'convolutional',
                                                     'n_layers': 5,
                                                     'n_units': [32, 64, 128, 256, 512],
                                                     'filter_sizes': 4,
                                                     'strides': [2, 2, 2, 2, 1],
                                                     'paddings': [1, 1, 1, 1, 0],
                                                     'non_linearity': 'relu',
                                                     'connectivity': 'sequential'},
                          'temporal_network_config': None
                          }


if model_type == 'latent_conv_recurrent':
    latent_dim = 256


    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'base_loc_type': {'type': 'trans_conv',
                                                          'n_layers': 1,
                                                          'n_units': 1,
                                                          'filter_sizes': 4,
                                                          'strides': 2,
                                                          'paddings': 1,
                                                          'connectivity': 'sequential',
                                                          'non_linearity':'sigmoid'},
                                        'base_scale_type': {'type': 'trans_conv',
                                                            'n_layers': 1,
                                                            'n_units': 1,
                                                            'filter_sizes': 4,
                                                            'strides': 2,
                                                            'paddings': 1,
                                                            'connectivity': 'sequential',
                                                            'last_linear': True},
                                        },
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'trans_conv',
                                                    'n_layers': 4,
                                                    'n_units': [512, 256, 128, 64],
                                                    'filter_sizes': 4,
                                                    'strides': [1, 2, 2, 2],
                                                    'paddings': [0, 1, 1, 1],
                                                    'non_linearity': 'leaky_relu',
                                                    'connectivity': 'sequential',
                                                    'batch_norm': True},
                         'temporal_network_config': None
                      }



    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'base_loc_type': {'type': 'fully_connected',
                                                      'n_layers': 1,
                                                      'n_units': latent_dim},
                                    'base_scale_type': {'type': 'fully_connected',
                                                        'n_layers': 1,
                                                        'n_units': latent_dim},},
                    'spatial_network_config': {'inputs': ['z'],
                                               'type': 'fully_connected',
                                               'n_layers': 2,
                                               'n_units': 256,
                                               'non_linearity': 'relu',
                                               'connectivity': 'sequential'},
                    'temporal_network_config': {'inputs': ['spatial_output'],
                                                'type': 'recurrent',
                                                'n_layers': 2,
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
                          'spatial_network_config': {'inputs': ['x'],
                                                     'type': 'convolutional',
                                                     'n_layers': 5,
                                                     'n_units': [64, 128, 256, 512, 128],
                                                     'filter_sizes': 4,
                                                     'strides': [2, 2, 2, 2, 1],
                                                     'paddings': [1, 1, 1, 1, 0],
                                                     'non_linearity': 'leaky_relu',
                                                     'connectivity': 'sequential',
                                                     'batch_norm': True},
                          'temporal_network_config': {'inputs': ['spatial_output', 'z'],
                                                      'type': 'recurrent',
                                                      'n_layers': 2,
                                                      'n_units': 256,
                                                      'non_linearity': 'tanh',
                                                      'connectivity': 'sequential'}
                          }


if model_type == 'latent_conv_recurrent_flow':
    latent_dim = 256


    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': {'type': 'trans_conv',
                                                          'n_layers': 1,
                                                          'n_units': [1],
                                                          'filter_sizes': 4,
                                                          'strides': 2,
                                                          'paddings': 1,
                                                          'connectivity': 'sequential'},
                                        'base_scale_type': {'type': 'trans_conv',
                                                            'n_layers': 1,
                                                            'n_units': [1],
                                                            'filter_sizes': 4,
                                                            'strides': 2,
                                                            'paddings': 1,
                                                            'connectivity': 'sequential'},
                                        'transform_config': {'sigmoid_last': False,
                                                             'n_transforms': 1},
                                        'flow_config': {'buffer_length': 3,
                                                        'constant_scale': False,
                                                        'network_config': {'type': 'custom'}
                                                        }},
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'trans_conv',
                                                    'n_layers': 4,
                                                    'n_units': [512, 256, 128, 64],
                                                    'filter_sizes': 4,
                                                    'strides': [1, 2, 2, 2],
                                                    'paddings': [0, 1, 1, 1],
                                                    'non_linearity': 'leaky_relu',
                                                    'connectivity': 'sequential',
                                                    'batch_norm': True,
                                                    'last_linear': False},
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
                                               'non_linearity': 'relu',
                                               'connectivity': 'sequential'},
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
                                                     'type': 'convolutional',
                                                     'n_layers': 5,
                                                     'n_units': [64, 128, 256, 512, 128],
                                                     'filter_sizes': 4,
                                                     'strides': [2, 2, 2, 2, 1],
                                                     'paddings': [1, 1, 1, 1, 0],
                                                     'non_linearity': 'leaky_relu',
                                                     'connectivity': 'sequential',
                                                     'batch_norm': True},
                          'temporal_network_config': {'inputs': ['spatial_output', 'z'],
                                                      'type': 'recurrent',
                                                      'n_layers': 1,
                                                      'n_units': 256,
                                                      'non_linearity': 'tanh',
                                                      'connectivity': 'sequential'}
                          }


if model_type == 'latent_flow':
    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'sigmoid_loc': True,
                                        'constant_base': False,
                                        'transform_config': {'sigmoid_last': False,
                                                             'n_transforms': 2},
                                        'flow_config': {'buffer_length': 3,
                                                        'constant_scale': False,
                                                        'network_config': {'type': 'convolutional',
                                                                           'n_layers': 2,
                                                                           'n_units': 32,
                                                                           'filter_sizes': 3,
                                                                           'non_linearity': 'elu',
                                                                           'batch_norm': True,
                                                                           'connectivity': 'highway'}
                                                        }},

                         'network_config': {'inputs': ['z'],
                                            'type': 'fully_connected',
                                            'n_layers': 3,
                                            'n_units': 512,
                                            'non_linearity': 'relu',
                                            'connectivity': 'sequential'}
                      }

    latent_dim = 10

    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'sigmoid_loc': False,
                                    'constant_base': False},
                    'network_config': {'inputs': ['z'],
                                       'type': 'fully_connected',
                                       'n_layers': 3,
                                       'n_units': 512,
                                       'non_linearity': 'relu',
                                       'connectivity': 'sequential'}
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': [latent_dim],
                                          'sigmoid_loc': False,
                                          'constant_base': False},
                          'network_config': {'inputs': ['x'],
                                             'type': 'fully_connected',
                                             'n_layers': 3,
                                             'n_units': 512,
                                             'non_linearity': 'relu',
                                             'connectivity': 'sequential'}
                          }


if model_type == 'latent_fp':
    cond_like_config = {'dist_config': {'dist_type': 'Normal',
                                        'n_variables': None,
                                        'sigmoid_loc': True,
                                        'constant_base': False,
                                        },
                         'spatial_network_config': {'inputs': ['z'],
                                                    'type': 'fully_connected',
                                                    'n_layers': 3,
                                                    'n_units': 512,
                                                    'non_linearity': 'relu',
                                                    'connectivity': 'sequential'},
                         'temporal_network_config': None
                      }

    latent_dim = 128

    prior_config = {'dist_config': {'dist_type': 'Normal',
                                    'n_variables': [latent_dim],
                                    'sigmoid_loc': False,
                                    'constant_base': True},
                    'spatial_network_config': None,
                    'temporal_network_config': None
                    }

    approx_post_config = {'dist_config': {'dist_type': 'Normal',
                                          'n_variables': [latent_dim],
                                          'sigmoid_loc': False,
                                          'constant_base': False},
                          'spatial_network_config': {'inputs': ['x'],
                                                     'type': 'fully_connected',
                                                     'n_layers': 3,
                                                     'n_units': 512,
                                                     'non_linearity': 'relu',
                                                     'connectivity': 'sequential'},
                          'temporal_network_config': None
                          }

################################################################################


model_config = {'cond_like_config': cond_like_config,
                'prior_config': prior_config,
                'approx_post_config': approx_post_config
}
