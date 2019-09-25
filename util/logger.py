from comet_ml import Experiment
import numpy as np
import matplotlib
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
        for metric_name, metric in objectives.items():
            self.experiment.log_metric(metric_name + '_' + train_val, metric, self._epoch)
            print(metric_name, ':', metric.item())
        if train_val == 'train':
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
        grid = make_grid(images.clamp(0, 1), nrow=s).numpy()
        if c == 1:
            grid = grid[0]
            cmap = 'gray'
        else:
            grid = np.transpose(grid, (1, 2, 0))
            cmap = None
        plt.imshow(grid, cmap=cmap)
        self.experiment.log_figure(figure=plt, figure_name=title + '_' + train_val)
        plt.close()

    def save(self, model):
        """
        Save the model weights in comet.

        Args:
            model (nn.Module): the model to be saved
        """
        pass
