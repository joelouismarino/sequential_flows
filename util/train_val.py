import torch
import time
import numpy as np
from .gradients import grad_norm, grad_max
from .generate import generate


def train_val(data, model, optimizer=None, predict=False, eval_length=0):
    """
    Train the model on the train data.
    """
    total_objective = {'cll': [], 'kl': []}
    gradients = {'max': [], 'norm': []}
    # total_params = {'scales': [], 'shifts': [], 'base_scale': [], 'base_loc': []}
    total_params = {'scales': [], 'shifts': []}

    images = {'data': [], 'recon_mean': [], 'recon_sample': []}
    if 'get_affine_params' in dir(model.cond_like.dist):
        images['noise'] = []
        affine_params = model.cond_like.dist.get_affine_params()
        n_flows = len(affine_params['shifts'])
        for i in range(n_flows):
            images['shift{}'.format(i)] = []
            images['scale{}'.format(i)] = []

    recon_mse = []
    pred_mse = []
    pred_psnr = []
    pred_ssim = []

    for batch_ind, batch in enumerate(data):
        # batch = batch.to(model.device).clamp(1e-6, 1-1e-6)
        batch = batch.to(model.device)

        model.reset(batch.shape[1])
        objective = {'cll': [], 'kl': []}
        # params = {'scales': [], 'shifts': [], 'base_scale': [], 'base_loc': []}
        params = {'scales': [], 'shifts': []}

        if predict:
            preds, batch_pred_mse, batch_pred_psnr = generate(batch, model, 5)
            if batch_ind == 0:
                images['pred'] = preds['pred']

            pred_mse.append(batch_pred_mse)
            pred_psnr.append(batch_pred_psnr)
            # pred_ssim.append(batch_pred_ssim)
            model.reset(batch.shape[1])

        # for step_ind in range(batch.size(0)):
        #     step_data = batch[step_ind,...].contiguous()

        for step_ind, step_data in enumerate(batch):
            # run the model on the data

            # check if all transform buffers are filled

            model(step_data)

            if model.ready():
                # get the objective terms
                results = model.evaluate(step_data)
                for result_name, result in results.items():
                    objective[result_name].append(result)

                # get various parameters for monitoring
                if 'get_affine_params' in dir(model.cond_like.dist):
                    affine_params = model.cond_like.dist.get_affine_params()
                    for param_name, param in affine_params.items():
                        params[param_name].append(param[0].detach().cpu())

                # params['base_loc'].append(model.cond_like.dist.base_dist.loc.detach().cpu())
                # params['base_scale'].append(model.cond_like.dist.base_dist.scale.detach().cpu())

                # get images for visualization
                recon_mean = model.cond_like.dist.mean.detach().cpu().view(step_data.shape)
                recon_sample = model.cond_like.dist.sample().detach().cpu().view(step_data.shape)
                if batch_ind == 0:
                    images['data'].append(step_data.cpu())
                    # recon = model.cond_like.sample().detach().cpu()
                    images['recon_mean'].append(recon_mean)
                    images['recon_sample'].append(recon_sample)

                    if 'get_affine_params' in dir(model.cond_like.dist):
                        noise = model.cond_like.dist.inverse(step_data).detach().cpu()
                        # images['noise'].append(noise)
                        images['noise'].append((noise - noise.min()) / (noise.max() - noise.min()))
                        for i_flow in range(n_flows):
                            shift = affine_params['shifts'][i_flow].detach().cpu()
                            images['shift{}'.format(i_flow)].append(shift)
                            scale = affine_params['scales'][i_flow].detach().cpu()
                            images['scale{}'.format(i_flow)].append(scale)


                # accumulate metrics
                recon_mse.append(torch.sum((step_data.cpu() - recon_mean.detach().cpu())**2).item())
                # sample_mse.append(torch.sum((step_data.cpu() - model.cond_like.sample().detach().cpu())**2).item())

            # else:
            #     print('skip step', step_ind)

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

        total_objective['cll'].append(cll[-eval_length:].mean().detach().cpu())
        total_objective['kl'].append(kl[-eval_length:].mean().detach().cpu())

        if 'get_affine_params' in dir(model.cond_like.dist):
            for param_name, param in params.items():
                total_params[param_name].append(torch.stack(param).mean().item())


    objectives = {k: torch.stack(v).mean() for k, v in total_objective.items()}
    grads = {k: np.mean(v) for k, v, in gradients.items()} if optimizer else None
    imgs = {k: torch.stack(v) for k, v in images.items()}

    if 'get_affine_params' in dir(model.cond_like.dist):
        parameters = {k: np.mean(v) for k, v in total_params.items()}
        metrics = {'recon_mean_mse': sum(recon_mse)/len(recon_mse)}
    else:
        parameters = {'dummy parama':[]}
        metrics = {'dummy_metric': []}

    if predict:
        metrics['pred_mse'] = sum(pred_mse)/len(pred_mse)
        metrics['pred_psnr'] = sum(pred_psnr)/len(pred_psnr)
        # metrics['pred_ssim'] = sum(pred_ssim)/len(pred_ssim)

    return objectives, grads, parameters, imgs, metrics

def train(data, model, optimizer, eval_length):
    print('Training...')
    t_start = time.time()
    model.train()
    results = train_val(data, model, optimizer, eval_length=eval_length)
    print('Duration: ' + '{:.2f}'.format(time.time() - t_start) + ' s.')
    return results

def validation(data, model, eval_length):
    print('Validation...')
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        results = train_val(data, model, predict=True, eval_length=eval_length)
    print('Duration: ' + '{:.2f}'.format(time.time() - t_start) + ' s.')
    return results
