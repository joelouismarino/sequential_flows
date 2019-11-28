
model_type = 'custom_2flow'

################################################################################


if model_type == 'custom_2flow':

    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': 'global',
                                        'base_scale_type': 'global',
                                        'transform_config':{'sigmoid_last':False,
                                                            'n_transforms':2},
                                        'flow_config': {'buffer_length': 3,
                                                        'constant_scale': False,
                                                        'network_config': {'type': 'custom',
                                                                           'c_in': 3}
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
