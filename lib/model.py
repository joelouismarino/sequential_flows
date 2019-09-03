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
        self.cond_like = Distribution(**cond_like_config)
        self.prior = Distribution(**prior_config) if prior_config else None
        self.approx_post = Distribution(**approx_post_config) if approx_post_config else None
        self._prev_x = None
        self._prev_z = None
        self._batch_size = None

    def forward(self, x, generate=False):
        """
        Calculates distributions.
        """
        # x = x.view(self._batch_size, -1)
        if self.prior is not None:
            if self._prev_z is not None:
                self.prior(z=self._prev_z, x=self._prev_x)
            if generate:
                z = self.prior.sample()
            else:
                self.approx_post(z=self._prev_z, x=x-0.5)
                z = self.approx_post.sample()
            self.cond_like(z=z, x=self._prev_x)
            self._prev_z = z
        else:
            self.cond_like(x=self._prev_x)
        self._prev_x = x

    def step(self):
        """
        Step the model forward in time.
        """
        self.cond_like.step(self._prev_x)
        if self.prior is not None:
            self.prior.step(self._prev_z)
            self.approx_post.step(self._prev_z)

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
            self._prev_z = torch.zeros(batch_size, self.approx_post.n_variables).to(self.device)

    @property
    def device(self):
        return list(self.parameters())[0].device
