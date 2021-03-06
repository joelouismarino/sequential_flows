data_config = {'dataset_name': 'kth_actions',
               'data_path': '/path/to/data',
               'add_noise': True,
               'train_epoch_size': 500,
               'val_epoch_size': 0,
               'sequence_length': 16,
               'eval_length': 10,
               'img_hz_flip': False,
               'img_size': 64}

def get_data_config():
    return data_config
