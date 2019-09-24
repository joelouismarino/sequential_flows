
model_type = 'rmn_flow'

################################################################################

if model_type == 'rmn_flow':

    cond_like_config = {'dist_config': {'dist_type': 'AutoregressiveFlow',
                                        'n_variables': None,
                                        'base_loc_type': 'global',
                                        'base_scale_type': 'global',
                                        'transform_config':{'sigmoid_last':False,
                                                            'n_transforms':2},
                                        'flow_config': {'buffer_length': 1,
                                                        'init_buffer': False,
                                                        'network_config': {'type': 'rmn',
                                                                           'constant_scale': False}
                                                        }},
                        'spatial_network_config': None,
                        'temporal_network_config': None,
                      }

    prior_config = approx_post_config = None


model_config = {'cond_like_config': cond_like_config,
                'prior_config': prior_config,
                'approx_post_config': approx_post_config
}
