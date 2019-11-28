import os, sys
from .update_model_config import update_model_config

def get_config(config_name=None):
    # import the intended config files
    if config_name is not None:
        config_path = os.path.join(os.getcwd(), 'config', config_name)
        assert os.path.exists(config_path), 'Config path does not exist.'
    else:
        config_path = os.path.join(os.getcwd(), 'config')
    sys.path.insert(0, config_path)
    from data_config import get_data_config
    from exp_config import get_exp_config
    from model_config import get_model_config
    data_config = get_data_config()
    exp_config = get_exp_config()
    model_config = get_model_config()
    # update the model config with sizes
    update_model_config(model_config, data_config)
    # enter the comet credentials, data path
    from .local import comet_config, rest_api_key, data_path
    exp_config['comet_config'] = comet_config
    exp_config['rest_api_key'] = rest_api_key
    data_config['data_path'] = data_path
    return model_config, data_config, exp_config
