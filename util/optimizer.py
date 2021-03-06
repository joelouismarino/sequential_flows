from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import Adam
import sys


def count_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)

class AdamOptimizer(object):

    def __init__(self, params, lr, grad_clip_value=None, grad_clip_norm=None, **kwargs):
        # print('optimizing {} paramters'.format(count_parameters(params)))
        # sys.exit()

        self.optimizer = Adam(params=params, lr=lr)
        self.params = params
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm


    def step(self):
        if self.grad_clip_value is not None:
            clip_grad_value_(self.params, self.grad_clip_value)
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self.params, self.grad_clip_norm)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

