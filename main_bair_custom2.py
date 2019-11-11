import comet_ml, os, torch
from bair_custom2_config import exp_config, model_config, data_config
from data import load_data
from lib.model import Model
from util import Logger, train, validation, AdamOptimizer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(exp_config['device'])
torch.cuda.set_device(0)

# data
train_data, val_data = load_data(data_config, exp_config['batch_size'], num_workers=0)
eval_length = data_config['eval_length']
train_epoch_size = data_config['train_epoch_size']
val_epoch_size = data_config['val_epoch_size']

# logger

# model
model = Model(**model_config).to(0)

logger_on = True
if logger_on:
    logger = Logger(exp_config, model_config, data_config)
    if exp_config['checkpoint_exp_key'] is not None:
        logger.load(model)

# optimizer
optimizer = AdamOptimizer(params=model.parameters(), lr=exp_config['lr'],
                          grad_clip_value=exp_config['grad_clip_value'],
                          grad_clip_norm=exp_config['grad_clip_norm'])



# train / val loop
for epoch in range(exp_config['n_epochs']):

    print('Epoch:', epoch)
    if logger_on:
        logger.log(train(train_data, model, optimizer, eval_length, train_epoch_size), 'train')
        logger.log(validation(val_data, model, eval_length, val_epoch_size, use_mean_pred=True), 'val')
        logger.save(model)
    else:
        train(train_data, model, optimizer, eval_length, train_epoch_size)
        validation(val_data, model, eval_length, val_epoch_size)
