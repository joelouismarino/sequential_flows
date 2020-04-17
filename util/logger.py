import comet_ml
from comet_ml import Experiment
import os, torch, io
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class Logger:
    """
    Logs/plots results to comet.

    Args:
        exp_config (dict): experiment configuration hyperparameters
        model_config (dict): model configuration hyperparameters
        data_config (dict): data configuration hyperparameters
    """
    def __init__(self, exp_config, model_config, data_config):
        self.exp_config = exp_config
        self.experiment = Experiment(**exp_config['comet_config'])
        self.experiment.disable_mp()
        self._log_hyper_params(exp_config, model_config, data_config)
        self._epoch = 0

    def _log_hyper_params(self, exp_config, model_config, data_config):
        """
        Log the hyper-parameters for the experiment.

        Args:
            exp_config (dict): experiment configuration hyperparameters
            model_config (dict): model configuration hyperparameters
            data_config (dict): data configuration hyperparameters
        """
        def flatten_arg_dict(arg_dict):
            flat_dict = {}
            for k, v in arg_dict.items():
                if type(v) == dict:
                    flat_v = flatten_arg_dict(v)
                    for kk, vv in flat_v.items():
                        flat_dict[k + '_' + kk] = vv
                else:
                    flat_dict[k] = v
            return flat_dict

        self.experiment.log_parameters(flatten_arg_dict(exp_config))
        self.experiment.log_parameters(flatten_arg_dict(model_config))
        self.experiment.log_parameters(flatten_arg_dict(data_config))

    def log(self, results, train_val):
        """
        Plot the results in comet.

        Args:
            results (dict): dictionary of metrics to plot
            train_val (str): either 'train' or 'val'
        """
        objectives, grads, params, images, metrics = results
        objectives['fe_sep'] = [-1*x for x in objectives['fe_sep']]
        for metric_name, metric in objectives.items():
            if 'sep' not in metric_name:
                self.experiment.log_metric(metric_name + '_' + train_val, metric, self._epoch)
                print(metric_name, ':', metric.item())

        if train_val == 'train':
            self.train_obj = objectives
            for grad_metric_name, grad_metric in grads.items():
                self.experiment.log_metric('grads_' + grad_metric_name, grad_metric, self._epoch)
        for param_name, param in params.items():
            self.experiment.log_metric(param_name + '_' + train_val, param, self._epoch)
        for image_name, imgs in images.items():
            self.plot_images(imgs, image_name, train_val)
        for metric_name, metric in metrics.items():
            self.experiment.log_metric(metric_name + '_' + train_val, metric, self._epoch)
        if train_val == 'val':
            self._epoch += 1

            for metric in ['cll', 'fe']:
                self.plot_dist_hist(self.train_obj, objectives, metric)

    def plot_dist_hist(self, train_obj, val_obj, metric):

        max_val = max(max(train_obj['{}_sep'.format(metric)]), max(val_obj['{}_sep'.format(metric)]))
        min_val = min(min(train_obj['{}_sep'.format(metric)]), min(val_obj['{}_sep'.format(metric)]))

        bins = np.linspace(min_val, max_val, 50)
        plt.hist(train_obj['{}_sep'.format(metric)], bins=bins, color='r', alpha=0.5, label='{}_train'.format(metric), density=True)
        plt.hist(val_obj['{}_sep'.format(metric)], bins=bins, color='b', alpha=0.5, label='{}_val'.format(metric), density=True)

        leg = plt.legend(loc='upper right')

        plt.draw()
        # plt.show()
        self.experiment.log_figure(figure=plt, figure_name='{}_hist'.format(metric))
        plt.close()


    def plot_images(self, images, title, train_val):
        """
        Plot a tensor of images.

        Args:
            images (torch.Tensor): a tensor of shape [steps, b, c, h, w]
            title (str): title for the images, e.g. reconstructions
            train_val (str): either 'train' or 'val'
        """
        # add a channel dimension if necessary
        if len(images.shape) == 4:
            s, b, h, w = images.shape
            images = images.view(s, b, 1, h, w)
        s, b, c, h, w = images.shape
        if b > 10:
            images = images[:, :10]
        # swap the steps and batch dimensions
        images = images.transpose(0, 1).contiguous()
        images = images.view(-1, c, h, w)
        # grid = make_grid(images.clamp(0, 1), nrow=s).numpy()
        grid = make_grid(images, nrow=s).numpy()
        if c == 1:
            grid = grid[0]
            cmap = 'gray'
        else:
            grid = np.transpose(grid, (1, 2, 0))
            cmap = None
        plt.imshow(grid, cmap=cmap)
        plt.axis('off')
        self.experiment.log_figure(figure=plt, figure_name=title + '_' + train_val)
        plt.close()

    def save(self, model):
        """
        Save the model weights in comet.

        Args:
            model (nn.Module): the model to be saved
        """
        if self._epoch % self.exp_config['checkpoint_interval'] == 0:
            print('Checkpointing the model...')
            state_dict = model.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            # save the state dictionary
            ckpt_path = os.path.join('./ckpt_epoch_'+ str(self._epoch) + '.ckpt')
            torch.save(cpu_state_dict, ckpt_path)
            self.experiment.log_asset(ckpt_path)
            os.remove(ckpt_path)
            print('Done.')

    def load(self, model):
        """
        Load the model weights.
        """
        assert self.exp_config['checkpoint_exp_key'] is not None, 'Checkpoint experiment key must be set.'
        print('Loading checkpoint from ' + self.exp_config['checkpoint_exp_key'] + '...')
        comet_api = comet_ml.papi.API(rest_api_key=self.exp_config['rest_api_key'])
        exp = comet_api.get_experiment(workspace=self.exp_config['comet_config']['workspace'],
                                       project_name=self.exp_config['comet_config']['project_name'],
                                       experiment=self.exp_config['checkpoint_exp_key'])
        # asset_list = comet_api.get_experiment_asset_list(self.exp_config['checkpoint_exp_key'])
        asset_list = exp.get_asset_list()
        # get most recent checkpoint
        ckpt_assets = [asset for asset in asset_list if 'ckpt' in asset['fileName']]
        asset_times = [asset['createdAt'] for asset in ckpt_assets]
        asset = asset_list[asset_times.index(max(asset_times))]
        print('Checkpoint Name:', asset['fileName'])
        ckpt = exp.get_asset(asset['assetId'])
        state_dict = torch.load(io.BytesIO(ckpt))
        model.load(state_dict)
        print('Done.')
