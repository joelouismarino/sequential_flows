import torch
import torch.nn as nn
from .networks import get_network
from .layers import FullyConnectedLayer, ConvolutionalLayer, Layer
from .flows import AutoregressiveFlow, AutoregressiveTransform


class Distribution(nn.Module):
    """
    A (conditional) probability distribution.

    Args:
        dist_config (dict): configuration for the distribution
        network_config (dict, optional): configuration for the network
    """
    def __init__(self, dist_config, spatial_network_config=None, temporal_network_config=None):
        super(Distribution, self).__init__()
        # network
        self.inputs = self.spatial_network = self.temporal_network = None
        n_out = None
        if spatial_network_config:
            self.spatial_network = get_network(spatial_network_config)
            n_out = self.spatial_network.n_out
        if temporal_network_config:
            if not spatial_network_config:
                print('spatial network not specified')
                raise NotImplementedError
            temporal_network_config['n_input'] += n_out
            self.temporal_network = get_network(temporal_network_config)
            n_out = self.temporal_network.n_out

        self.n_out = n_out

        # distribution
        self.dist = None
        self.transforms = None
        self.param_layers = None
        self.n_variables = dist_config['n_variables']
        self._ready = True

        if dist_config['dist_type'] == 'AutoregressiveFlow':
            self.dist_type = AutoregressiveFlow
            self.sigmoid_last = dist_config['transform_config']['sigmoid_last']
            # n_transforms = dist_config['flow_config'].pop('n_transforms')
            n_transforms = dist_config['transform_config']['n_transforms']
            self.transforms = nn.ModuleList([AutoregressiveTransform(**dist_config['flow_config']) for _ in range(n_transforms)])
            self._ready = all([t.ready() for t in self.transforms])
        else:
            self.dist_type = getattr(torch.distributions, dist_config['dist_type'])

        param_names = list(self.dist_type.arg_constraints.keys())

        self.log_scale = None
        self.loc = None
        param_configs = {}
        if 'scale' in param_names:
            # use constant scale
            if dist_config['base_scale_type'] == 'constant':
                self.log_scale = torch.zeros([1] + self.n_variables).cuda()
                param_names.remove('scale')

            # use global learnable scale
            elif dist_config['base_scale_type'] == 'global':
                self.log_scale = torch.nn.Parameter(torch.zeros([1] + self.n_variables)).cuda()
                param_names.remove('scale')

        if 'loc' in param_names:
            # use constant scale
            if dist_config['base_loc_type'] == 'constant':
                self.loc = torch.zeros([1] + self.n_variables).cuda()
                param_names.remove('loc')

            # use global learnable scale
            elif dist_config['base_loc_type'] == 'global':
                self.loc = torch.nn.Parameter(torch.zeros([1] + self.n_variables)).cuda()
                param_names.remove('loc')


        self.initial_params = nn.ParameterDict({name: None for name in param_names})
        if spatial_network_config:
            self.param_layers = nn.ModuleDict({name: None for name in param_names})
        for param_name in param_names:
            if self.param_layers:
                param_layer_config = dist_config['base_{}_type'.format(param_name)]
                param_layer_config['n_input'] = self.n_out
                self.param_layers[param_name] = get_network(param_layer_config)
                # non_linearity = None
                # if param_name == 'loc' and dist_config['sigmoid_loc']:
                #     non_linearity = 'sigmoid'
                #
                # if 'trans_conv' in spatial_network_config['type']:
                #     if param_name == 'loc':
                #         self.param_layers[param_name] = Layer()
                #         if dist_config['sigmoid_loc']:
                #             self.param_layers[param_name].non_linearity = torch.nn.Sigmoid()
                #     else:
                #         raise NotImplementedError
                #
                # else:
                #     if len(self.n_variables) == 1:
                #         self.param_layers[param_name] = FullyConnectedLayer(self.n_out,
                #                                                             self.n_variables[0],
                #                                                             non_linearity=non_linearity)
                #     else:
                #         n_out = 1
                #         for var in self.n_variables:
                #             n_out *= var
                #         self.param_layers[param_name] = FullyConnectedLayer(self.n_out,
                #                                                             n_out,
                #                                                             non_linearity=non_linearity)
                        # raise NotImplementedError
            if param_name == 'scale':
                self.initial_params[param_name] = nn.Parameter(torch.ones([1] + self.n_variables))
            else:
                self.initial_params[param_name] = nn.Parameter(torch.zeros([1] + self.n_variables))

        self.reset(1)

    def step(self, x):
        """
        Step the distribution forward in time.
        """
        if self.spatial_network:
            self.spatial_network.step(x)
        if self.temporal_network:
            self.temporal_network.step(x)
        if self.transforms:
            self.dist.step(x)
            self._ready = all([t.ready() for t in self.transforms])

    def forward(self, **kwargs):
        """
        Calculate the distribution.
        """
        if self.spatial_network:
            dist_input = self.spatial_network(**kwargs)
            if self.temporal_network:
                if 'Conv' not in type(self.temporal_network.layers[0]).__name__:
                    kwargs['spatial_output'] = dist_input.view(dist_input.size(0), -1)
                else:
                    kwargs['spatial_output'] = dist_input

                dist_input = self.temporal_network(**kwargs)
            parameters = {}
            for param_name, param_layer in self.param_layers.items():
                # if 'FullyConnected' in type(param_layer).__name__:
                #     dist_input = dist_input.view(dist_input.size(0), -1)
                param = param_layer(dist_input)

                if param_name == 'scale':
                    param = torch.exp(torch.clamp(param, -15, 5))
                    # param = torch.exp(param)

                parameters[param_name] = param

            if self.log_scale is not None:
                log_scale = self.log_scale.repeat([dist_input.size(0)] + [1 for _ in range(len(self.n_variables))])
                scale = torch.exp(torch.clamp(log_scale, -15, 5))
                parameters['scale'] = scale
            if self.loc is not None:
                parameters['loc'] = self.loc.repeat([dist_input.size(0)] + [1 for _ in range(len(self.n_variables))])
            if self.transforms:
                parameters['transforms'] = [t for t in self.transforms]
                parameters['sigmoid_last'] = self.sigmoid_last
            self.dist = self.dist_type(**parameters)


    def ready(self):
        return self._ready


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
        if self.spatial_network:
            self.spatial_network.reset(batch_size)
        if self.temporal_network:
            self.temporal_network.reset(batch_size)

        params = {k: v.repeat([batch_size] + [1 for _ in range(len(self.n_variables))]) for k, v in self.initial_params.items()}
        # params = {k: v.repeat([batch_size] + [1 for _ in v.size()[1:]]) for k, v in self.initial_params.items()}
        if self.log_scale is not None:
            # log_scale = self.log_scale.repeat([batch_size] + [1 for _ in range(len(self.n_variables))])
            log_scale = self.log_scale.repeat([batch_size] + [1 for _ in self.log_scale.size()[1:]])
            scale = torch.exp(torch.clamp(log_scale, -15, 5))
            params['scale'] = scale
        if self.loc is not None:
            # params['loc'] = self.loc.repeat([batch_size] + [1 for _ in range(len(self.n_variables))])
            params['loc'] = self.loc.repeat([batch_size] + [1 for _ in self.loc.size()[1:]])
        if self.transforms:
            for transform in self.transforms:
                if 'reset' in dir(transform):
                    transform.reset(batch_size)
            params['transforms'] = [t for t in self.transforms]
            params['sigmoid_last'] = self.sigmoid_last
            self._ready = all([t.ready() for t in self.transforms])

        self.dist = self.dist_type(**params)