import comet_ml, os, torch
# from local_bair_config import exp_config, model_config, data_config
from local_mmnist_config import exp_config, model_config, data_config
from data import load_data
from lib.model import Model
from util import Logger, train, validation, AdamOptimizer

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]=str(exp_config['device'])
# torch.cuda.set_device(0)

# data
train_data, val_data = load_data(data_config, exp_config['batch_size'])

# logger

# model
model = Model(**model_config).to(0)

# optimizer
optimizer = AdamOptimizer(params=model.parameters(), lr=exp_config['lr'],
                          grad_clip_value=exp_config['grad_clip_value'],
                          grad_clip_norm=exp_config['grad_clip_norm'])

logger_on = True

if logger_on:
    logger = Logger(exp_config, model_config, data_config)


# train / val loop
for epoch in range(exp_config['n_epochs']):
    if epoch == 20:
        print()

    print('Epoch:', epoch)
    if logger_on:
        logger.log(train(train_data, model, optimizer), 'train')
        logger.log(validation(val_data, model), 'val')
        logger.save(model)
    else:
        train(train_data, model, optimizer)
        validation(val_data, model)
