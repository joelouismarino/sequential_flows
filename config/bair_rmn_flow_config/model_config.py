model_type = 'rmn_flow'


if model_type == 'rmn_flow':

    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': 'constant',
                                        'base_scale_type': 'constant',
                                        'transform_config':{'sigmoid_last':False,
                                                            'n_transforms':1},
                                        'flow_config': {'buffer_length': 1,
                                                        'constant_scale': False,
                                                        'init_buffer': False,
                                                        'network_config': {'type': 'rmn',
                                                                           'constant_scale':True,
                                                                           'c_in': 3,
                                                                           'encoder_depth': 8,
                                                                           'decoder_depth': 12,
                                                                           'lstm_hidden': 512}
                                                        }},
                        'spatial_network_config': None,
                        'temporal_network_config': None,
                      }

    prior_config = approx_post_config = None


################################################################################

model_config = {'cond_like_config': cond_like_config,
                'prior_config': prior_config,
                'approx_post_config': approx_post_config
}

def get_model_config():
    return model_config
