exp_config = {'n_epochs': 1000,
              'batch_size': 8,
              'lr': 1e-4,
              'grad_clip_value': None,
              # 'grad_clip_norm': None,
              'grad_clip_norm': 1.,
              'device': 1,
              'checkpoint_interval':10,
              'checkpoint_exp_key': None,
              'rest_api_key': None,
              'comet_config': None,
              }

def get_exp_config():
    return exp_config
