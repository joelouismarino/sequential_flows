import torch
import torch.nn as nn
from .distribution import Distribution


class Model(nn.Module):
    """
    An autoregressive model.

    Args:
        cond_like_config (dict): configuration for the conditional likelihood
        prior_config (dict, optional): configuration for the prior
        approx_post_config (dict, optional): configuration for the approximate posterior
    """
    def __init__(self, cond_like_config, prior_config=None, approx_post_config=None):
        super(Model, self).__init__()

        self.cond_like_config = cond_like_config
        self.latent_size = None
        if prior_config is not None and 'latent_size' in prior_config:
            self.latent_size = prior_config.pop('latent_size')

        self.cond_like = Distribution(**cond_like_config)
        self.prior = Distribution(**prior_config) if prior_config else None
        self.approx_post = Distribution(**approx_post_config) if approx_post_config else None
        self._prev_x = None
        self._prev_z = None
        self._prev_y = None
        self._batch_size = None

        self.with_flow = (cond_like_config['dist_config']['dist_type'] in ['AutoregressiveFlow', 'Glow'])
        self.flow_type = None if not self.with_flow else cond_like_config['dist_config']['dist_type']
        self._ready = self.cond_like.ready()

        # import ipdb; ipdb.set_trace()
    # def forward(self, x, generate=False):
    #     """
    #     Calculates distributions.
    #     """
    #     # x = x.view(self._batch_size, -1)
    #
    #     y = None
    #     if self.prior is not None:
    #         if self.with_flow:
    #             y = self.cond_like.dist.inverse(x)
    #         if self._prev_z is not None:
    #             self.prior(z=self._prev_z, x=self._prev_x, y=self._prev_y)
    #         if generate:
    #             z = self.prior.sample()
    #         else:
    #             self.approx_post(z=self._prev_z, x=x-0.5, y=y)
    #             # self.approx_post(z=self._prev_z, x=x, y=y)
    #             z = self.approx_post.sample()
    #         self.cond_like(z=z, x=self._prev_x)
    #         self._prev_z = z
    #         self._prev_y = y
    #     else:
    #         self.cond_like(x=self._prev_x)
    #     self._prev_x = x

    def ready(self):
        return self._ready

    def forward(self, x, generate=False):
        """
        Calculates distributions.
        """
        if self._ready:
            y = None
            s = None
            if self.prior is not None:
                if self.with_flow:
                    y = self.cond_like.dist.inverse(x)
                if self._prev_z is not None:
                    self.prior(z=self._prev_z, x=self._prev_x, y=self._prev_y)
                    s = self.prior.state

                self.approx_post(z=self._prev_z, x=x, y=y, s=s)

                if generate:
                    z = self.prior.sample()
                else:
                    z = self.approx_post.sample()

                self.cond_like(z=z, x=self._prev_x, s=s)
                self._prev_z = z
                self._prev_y = y

            elif self.flow_type == 'Glow':
                if self._prev_y is None:
                    y_shape = [x.size(0)] + self.cond_like_config['dist_config']['flow_config']['base_shape']
                    self._prev_y = torch.zeros(y_shape).cuda()

                self.cond_like(x=self._prev_x, y=self._prev_y)
                y = self.cond_like.dist.inverse(x)
                self._prev_y = y

            else:
                self.cond_like(x=self._prev_x)

        self._prev_x = x

    def predict(self, use_mean_pred=False):
        """
        predict next time step
        """
        # x = x.view(self._batch_size, -1)
        y = None
        if self.prior is not None:
            if self._prev_z is not None:
                self.prior(z=self._prev_z, x=self._prev_x, y=self._prev_y)
            z = self.prior.sample()
            s = self.prior.state

            self.cond_like(z=z, x=self._prev_x, s=s)
            if use_mean_pred:
                pred = self.cond_like.dist.mean.view(self._prev_x.size())
            else:
                pred = self.cond_like.dist.sample().view(self._prev_x.size())

            self._prev_z = z
            self._prev_x = pred

            if self.with_flow:
                y = self.cond_like.dist.inverse(pred)
            self._prev_y = y

        elif self.flow_type == 'Glow':
            self.cond_like(x=self._prev_x, y=self._prev_y)
            if use_mean_pred:
                pred = self.cond_like.dist.mean.view(self._prev_x.size())
            else:
                pred = self.cond_like.dist.sample().view(self._prev_x.size())

            self._prev_x = pred
            y = self.cond_like.dist.inverse(pred)
            self._prev_y = y

        else:
            self.cond_like(x=self._prev_x)
            if use_mean_pred:
                pred = self.cond_like.dist.mean.view(self._prev_x.size())
            else:
                pred = self.cond_like.dist.sample().view(self._prev_x.size())

            self._prev_x = pred


    def step(self):
        """
        Step the model forward in time.
        """
        self.cond_like.step(self._prev_x)
        if self.prior is not None:
            self.prior.step(self._prev_z)
            self.approx_post.step(self._prev_z)

        self._ready = self.cond_like.ready()

    def evaluate(self, x):
        """
        Evaluates the objective at x.
        """
        # x = x.view(self._batch_size, -1)

        cond_log_like = self.cond_like.log_prob(x).view(self._batch_size, -1).sum(dim=1)
        if self.prior is not None:
            try:
                kl = torch.distributions.kl_divergence(self.approx_post.dist, self.prior.dist)
            except NotImplementedError:
                z = approx_post.sample()
                kl = self.approx_post.log_prob(z) - self.prior.log_prob(z)
            kl = kl.view(self._batch_size, -1).sum(dim=1)
        else:
            kl = torch.zeros(self._batch_size).to(self.device)
        return {'cll': cond_log_like, 'kl': kl}

    def reset(self, batch_size):
        """
        Reset the model's hidden states.
        """
        self._prev_x = None
        self._batch_size = batch_size
        self.cond_like.reset(batch_size)
        if self.prior is not None:
            self.prior.reset(batch_size)
            self.approx_post.reset(batch_size)

            if self.latent_size is None:
                self._prev_z = torch.zeros(batch_size, self.approx_post.n_variables[0]).to(self.device)
            else:
                self._prev_z = torch.zeros([batch_size]+self.latent_size).to(self.device)

        self._ready = self.cond_like.ready()

    @property
    def device(self):
        return list(self.parameters())[0].device
