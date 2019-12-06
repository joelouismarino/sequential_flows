from torch.distributions import TransformedDistribution, constraints, Normal
from torch.distributions.transforms import SigmoidTransform
from .autoregressive_transform import AutoregressiveTransform


class Glow(TransformedDistribution):
    """
    Glow model.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, transforms, validate_args=None):

        super(Glow, self).__init__(Normal(loc, scale), transforms,
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
        for transform in reversed(self.transforms):
            if 'step' in dir(transform):

                if transform.ready():
                    x_inv = transform.inv(x)
                else:
                    x_inv = None

                transform.step(x)

                if x_inv is not None:
                    x = x_inv
                else:
                    break

            else:
                x = transform.inv(x)


    # def ready(self):
    #     return all([t.ready() for t in self.transforms])

    # def get_affine_params(self):
    #     """
    #     Collects the affine parameters from each transform.
    #     """
    #     params = {'scales': [], 'shifts': []}
    #     for transform in reversed(self.transforms):
    #         if '_scale' in dir(transform):
    #             params['scales'].append(transform._scale)
    #             params['shifts'].append(transform._shift)
    #     return params

    def reset(self, batch_size):
        """
        Resets each of the transforms.
        """
        for transform in self.transforms:
            if 'reset' in dir(transform):
                transform.reset(batch_size)
