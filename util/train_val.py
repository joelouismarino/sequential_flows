import torch
import time
import numpy as np
from .gradients import grad_norm, grad_max


def train_val(data, model, optimizer=None):
    """
    Train the model on the train data.
    """
    total_objective = {'cll': [], 'kl': []}
    gradients = {'max': [], 'norm': []}
    total_params = {'scales': [], 'shifts': []}
    images = {'data': [], 'recon': [], 'noise': [], 'shift': []}

    for batch_ind, batch in enumerate(data):
        batch = batch.to(model.device).clamp(1e-6, 1-1e-6)
        model.reset(batch.shape[1])
        objective = {'cll': [], 'kl': []}
        params = {'scales': [], 'shifts': []}

        for step_ind, step_data in enumerate(batch):
            # run the model on the data
            model(step_data)

            # get the objective terms
            results = model.evaluate(step_data)
            for result_name, result in results.items():
                objective[result_name].append(result)

            # get various parameters for monitoring
            affine_params = model.cond_like.dist.get_affine_params()
            for param_name, param in affine_params.items():
                params[param_name].append(param[0].detach().cpu())

            # get images for visualization
            if batch_ind == 0:
                images['data'].append(step_data.cpu())
                recon = model.cond_like.sample().detach().cpu()
                recon = recon.view(step_data.shape)
                images['recon'].append(recon)
                noise = model.cond_like.dist.inverse(step_data).detach().cpu()
                images['noise'].append(noise)
                images['shift'].append(affine_params['shifts'][0].detach().cpu())

            # step the model forward
            model.step()

        cll = torch.stack(objective['cll'])
        kl = torch.stack(objective['kl'])
        fe = - cll + kl

        if optimizer:
            optimizer.zero_grad()
            fe.mean().backward()
            gradients['max'].append(grad_max(model.parameters()))
            gradients['norm'].append(grad_norm(model.parameters()))
            optimizer.step()

        total_objective['cll'].append(cll.mean().detach().cpu())
        total_objective['kl'].append(kl.mean().detach().cpu())

        for param_name, param in params.items():
            total_params[param_name].append(torch.stack(param).mean().item())

    objectives = {k: torch.stack(v).mean() for k, v in total_objective.items()}
    grads = {k: np.mean(v) for k, v, in gradients.items()} if optimizer else None
    parameters = {k: np.mean(v) for k, v in total_params.items()}
    imgs = {k: torch.stack(v) for k, v in images.items()}

    return objectives, grads, parameters, imgs

def train(data, model, optimizer):
    print('Training...')
    t_start = time.time()
    model.train()
    results = train_val(data, model, optimizer)
    print('Duration: ' + '{:.2f}'.format(time.time() - t_start) + ' s.')
    return results

def validation(data, model):
    print('Validation...')
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        results = train_val(data, model)
    print('Duration: ' + '{:.2f}'.format(time.time() - t_start) + ' s.')
    return results
