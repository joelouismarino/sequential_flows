from comet_ml import API
import numpy as np

def process_metric(raw_data, metric_name):
    n_epoch = raw_data[-1]['step']
    metric_list = [None for _ in range(n_epoch+1)]

    for data in raw_data:
        if data['metricName'] == metric_name:
            step = data['step']
            metric_list[step] = float(data['metricValue'])

    return metric_list

key_dict = {'mmnist_latent': 'd68ff968ef1f4fafabefd89fdf00a689',
            'mmnist_latent_flow': 'ad202b0b168642cd887d18702ee3cfc2',
            'bair_latent': 'f6f69fcbdd0f4092b3b735d82596f88d',
            'bair_latent_flow': '9dcb00e07e4c415fb0895d272dbdff26',
            'kth_latent': '4daf51e39dcd4ec1915a01d25ec9fe7d',
            'kth_latent_flow': 'ab408fa63f70432baa48ecf494f6dd43'}

exp = 'bair_latent'

comet_api = API(rest_api_key='gCy6fGvm32hp4HVYgKW5gQcVf')
raw_data = comet_api.get_experiment_metrics_raw(experiment_key=key_dict[exp])

cll_val_list = process_metric(raw_data, 'cll_val')
kl_val_list = process_metric(raw_data, 'kl_val')

valid_steps = set([i for i,x in enumerate(cll_val_list) if x is not None]).intersection(set([i for i,x in enumerate(kl_val_list) if x is not None]))
valid_steps = list(valid_steps)

cll_val_list = [cll_val_list[x] for x in valid_steps]
kl_val_list = [kl_val_list[x] for x in valid_steps]

metric = np.array(cll_val_list) - np.array(kl_val_list)
best_metric_idx = np.argmax(metric)

print('exp:', exp, 'cll', cll_val_list[best_metric_idx], 'kl', kl_val_list[best_metric_idx], 'cll-kl', metric[best_metric_idx])