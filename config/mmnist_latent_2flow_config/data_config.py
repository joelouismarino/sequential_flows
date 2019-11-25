data_config = {'dataset_name': 'moving_mnist',
               'data_path': '/path/to/data',
               'add_noise': True,
               # 'num_digits': 2,
               'train_epoch_size': 1000,
               'val_epoch_size': 100,
               'sequence_length': 15,
               'eval_length': 10,
               'img_hz_flip': False,
               'img_size': 64}

def get_data_config():
    return data_config
