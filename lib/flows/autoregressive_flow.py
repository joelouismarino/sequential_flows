from torch.distributions import TransformedDistribution, constraints, Normal
from torch.distributions.transforms import SigmoidTransform
from .autoregressive_transform import AutoregressiveTransform


class AutoregressiveFlow(TransformedDistribution):
    """
    Autoregressive flow. Transforms a Normal distribution using autoregressive
    transforms.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, transforms, sigmoid_last=True, validate_args=None):
        if sigmoid_last:
            transforms.append(SigmoidTransform())
        super(AutoregressiveFlow, self).__init__(Normal(loc, scale), transforms,
                                                 validate_args=validate_args)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        loc = self.base_dist.loc
        for transform in self.transforms:
            loc = transform(loc)

        return loc


    def inverse(self, x):
        """
        Inverts the transforms: x --> y.
        """
        for transform in reversed(self.transforms):
            x = transform.inv(x)
        return x

    def step(self, x):
        """
        Steps the transforms forward in time.
        """
        subtracted = False
        for transform in reversed(self.transforms):
            if 'step' in dir(transform):
                # if not subtracted:
                #     x = x - 0.5
                #     subtracted = True
                transform.step(x)
            x = transform.inv(x)

    def get_affine_params(self):
        """
        Collects the affine parameters from each transform.
        """
        params = {'scales': [], 'shifts': []}
        for transform in reversed(self.transforms):
            if '_scale' in dir(transform):
                params['scales'].append(transform._scale)
                params['shifts'].append(transform._shift)
        return params

    def reset(self, batch_size):
        """
        Resets each of the transforms.
        """
        for transform in self.transforms:
            if 'reset' in dir(transform):
                transform.reset(batch_size)
