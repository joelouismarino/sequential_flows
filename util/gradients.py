from torch._six import inf


def grad_max(params):
    return grad_norm(params, norm_type=inf)

def grad_norm(params, norm_type=2):
    gradients = [p.grad for p in params]
    grads = list(filter(lambda g: g is not None, gradients))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in grads).item()
    else:
        total_norm = 0
        for g in grads:
            norm = g.data.norm(norm_type)
            total_norm += norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm
