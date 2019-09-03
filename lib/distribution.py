import torch
import torch.nn as nn
from .networks import get_network
from .layers import FullyConnectedLayer, ConvolutionalLayer
from .flows import AutoregressiveFlow, AutoregressiveTransform


class Distribution(nn.Module):
    """
    A (conditional) probability distribution.

    Args:
        dist_config (dict): configuration for the distribution
        network_config (dict, optional): configuration for the network
    """
    def __init__(self, dist_config, network_config=None):
        super(Distribution, self).__init__()
        # network
        self.inputs = self.network = None
        if network_config:
            self.network = get_network(network_config)

        # distribution
        self.dist = None
        self.transforms = None
        self.param_layers = None
        self.n_variables = dist_config['n_variables']
        if dist_config['dist_type'] == 'AutoregressiveFlow':
            self.dist_type = AutoregressiveFlow
            n_transforms = dist_config['flow_config'].pop('n_transforms')
            self.transforms = nn.ModuleList([AutoregressiveTransform(**dist_config['flow_config']) for _ in range(n_transforms)])
        else:
            self.dist_type = getattr(torch.distributions, dist_config['dist_type'])

        param_names = self.dist_type.arg_constraints.keys()

        self.log_scale = None
        if 'scale' in param_names:
            self._log_scale_lim = [-15, 0]
            if dist_config['constant_scale']:
                self.log_scale = nn.Parameter(torch.ones([1] + self.n_variables))
                param_names = ['loc']

        # self.initial_params = nn.ParameterDict({name: None for name in param_names})
        self.initial_params = {name: None for name in param_names}
        if network_config:
            self.param_layers = nn.ModuleDict({name: None for name in param_names})
        for param_name in param_names:
            if self.param_layers:
                non_linearity = None
                if param_name == 'loc' and dist_config['sigmoid_loc']:
                    non_linearity = 'sigmoid'
                if len(self.n_variables) == 1:
                    self.param_layers[param_name] = FullyConnectedLayer(self.network.n_out,
                                                                        self.n_variables,
                                                                        non_linearity=non_linearity)
                else:
                    raise NotImplementedError
            if param_name == 'scale':
                self.initial_params[param_name] = nn.Parameter(torch.ones([1] + self.n_variables))
            else:
                self.initial_params[param_name] = nn.Parameter(torch.zeros([1] + self.n_variables))

        self.reset(1)

    def step(self, x):
        """
        Step the distribution forward in time.
        """
        if self.network:
            self.network.step(x)
        if self.transforms:
            self.dist.step(x)

    def forward(self, **kwargs):
        """
        Calculate the distribution.
        """
        if self.network:
            dist_input = self.network(**kwargs)
            parameters = {}
            for param_name, param_layer in self.param_layers.items():
                param = param_layer(dist_input)
                if param_name == 'scale':
                    param = torch.exp(torch.clamp(param, -15, 5))
                parameters[param_name] = param
            if self.log_scale is not None:
                log_scale = self.log_scale.repeat(dist_input.shape[0], 1)
                scale = torch.exp(torch.clamp(log_scale, -15, 5))
                parameters['scale'] = scale
            if self.transforms:
                parameters['transforms'] = self.transforms
            self.dist = self.dist_type(**parameters)

    def sample(self):
        """
        Sample from the distribution.
        """
        return self.dist.rsample() if self.dist.has_rsample else self.dist.sample()

    def log_prob(self, value):
        """
        Calculate the log probability at the given value.
        """
        return self.dist.log_prob(value)

    def reset(self, batch_size):
        if self.network:
            self.network.reset(batch_size)
        params = {k: v.repeat([batch_size] + [1 for _ in range(len(self.n_variables))]) for k, v in self.initial_params.items()}
        if self.log_scale is not None:
            log_scale = self.log_scale.repeat([batch_size] + [1 for _ in range(len(self.n_variables))])
            scale = torch.exp(torch.clamp(log_scale, -15, 5))
            params['scale'] = scale
        if self.transforms:
            for transform in self.transforms:
                if 'reset' in dir(transform):
                    transform.reset(batch_size)
            params['transforms'] = [t for t in self.transforms]
        self.dist = self.dist_type(**params)
