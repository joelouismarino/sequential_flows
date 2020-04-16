import torch
import time
import numpy as np
from .gradients import grad_norm, grad_max
from .generate import generate
from misc import estimate_correlation


def train_val(data, model, optimizer=None, predict=False, eval_length=0, epoch_size=0, use_mean_pred=False):
    """
    Train the model on the train data.
    """
    total_objective = {'cll': [], 'kl': [], 'fe': [], 'cll_sep': [], 'kl_sep': [], 'fe_sep': []}
    gradients = {'max': [], 'norm': []}
    # total_params = {'scales': [], 'shifts': [], 'base_scale': [], 'base_loc': []}
    total_params = {'scales': [], 'shifts': []}

    images = {'data': [], 'recon_mean': [], 'recon_sample': []}
    if 'get_affine_params' in dir(model.cond_like.dist):
        noise_correlation = []
        images['noise'] = []
        images['base_loc'] = []
        images['base_scale'] = []
        affine_params = model.cond_like.dist.get_affine_params()
        n_flows = len(affine_params['shifts'])
        for i in range(n_flows):
            images['shift{}'.format(i)] = []
            images['scale{}'.format(i)] = []

    recon_mse = []
    pred_mse = []
    # pred_psnr = []
    # pred_ssim = []

    cur_epoch_size = 0
    for batch_ind, batch in enumerate(data):
        # print('iter', batch_ind)
        # batch = batch.to(model.device).clamp(1e-6, 1-1e-6)
        batch = batch.to(model.device)

        model.reset(batch.shape[1])
        objective = {'cll': [], 'kl': []}
        # params = {'scales': [], 'shifts': [], 'base_scale': [], 'base_loc': []}
        params = {'scales': [], 'shifts': []}

        if predict:
            preds, batch_pred_mse = generate(batch, model, 5, use_mean_pred)
            if batch_ind == 0:
                images['pred'] = preds['pred']
                if 'get_affine_params' in dir(model.cond_like.dist):
                    images['pred_base_loc'] = preds['base_loc']
                    images['pred_base_scale'] = preds['base_scale']
                    images['pred_noise'] = preds['noise']

            pred_mse.append(batch_pred_mse)
            # pred_psnr.append(batch_pred_psnr)
            # pred_ssim.append(batch_pred_ssim)
            model.reset(batch.shape[1])

        # for step_ind in range(batch.size(0)):
        #     step_data = batch[step_ind,...].contiguous()

        if 'get_affine_params' in dir(model.cond_like.dist):
            noises = batch.new_zeros(batch.shape)

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
                    images['data'].append(step_data.cpu().clamp(0, 1))
                    # recon = model.cond_like.sample().detach().cpu()
                    images['recon_mean'].append(recon_mean.clamp(0, 1))
                    images['recon_sample'].append(recon_sample.clamp(0, 1))

                    if 'get_affine_params' in dir(model.cond_like.dist):
                        base_loc = model.cond_like.dist.base_dist.loc.detach().cpu().view(step_data.shape)
                        base_scale = model.cond_like.dist.base_dist.scale.detach().cpu().view(step_data.shape)
                        images['base_loc'].append((base_loc - base_loc.min()) / (base_loc.max() - base_loc.min()))
                        images['base_scale'].append((base_scale - base_scale.min()) / (base_scale.max() - base_scale.min()))

                        noise = model.cond_like.dist.inverse(step_data).detach().cpu()
                        # images['noise'].append(noise)
                        images['noise'].append((noise - noise.min()) / (noise.max() - noise.min()))
                        for i_flow in range(n_flows):
                            shift = affine_params['shifts'][i_flow].detach().cpu()
                            images['shift{}'.format(i_flow)].append((shift - shift.min()) / (shift.max() - shift.min()))
                            scale = affine_params['scales'][i_flow].detach().cpu()
                            images['scale{}'.format(i_flow)].append((scale - scale.min()) / (scale.max() - scale.min()))

                if 'get_affine_params' in dir(model.cond_like.dist):
                    noise = model.cond_like.dist.inverse(step_data).detach().cpu()
                    noises[step_ind] = noise

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
        total_objective['fe'].append(fe[-eval_length:].mean().detach().cpu())

        total_objective['cll_sep'].extend(cll[-eval_length:].mean(dim=0).detach().cpu().tolist())
        total_objective['kl_sep'].extend(kl[-eval_length:].mean(dim=0).detach().cpu().tolist())
        total_objective['fe_sep'].extend(-fe[-eval_length:].mean(dim=0).detach().cpu().tolist())

        if 'get_affine_params' in dir(model.cond_like.dist):
            noise_correlation.append(estimate_correlation(noises).mean().detach().cpu())
            for param_name, param in params.items():
                total_params[param_name].append(torch.stack(param).mean().item())

        cur_epoch_size += batch.size(0)
        if epoch_size > 0 and cur_epoch_size >= epoch_size:
            break


    objectives = {k: torch.stack(v).mean() for k, v in total_objective.items() if 'sep' not in k}
    objectives['cll_sep'] = total_objective['cll_sep']
    objectives['kl_sep'] = total_objective['kl_sep']
    objectives['fe_sep'] = total_objective['fe_sep']

    grads = {k: np.mean(v) for k, v, in gradients.items()} if optimizer else None
    imgs = {k: torch.stack(v) for k, v in images.items()}

    if 'get_affine_params' in dir(model.cond_like.dist):
        parameters = {k: np.mean(v) for k, v in total_params.items()}
        metrics = {'recon_mean_mse': sum(recon_mse) / len(recon_mse),
                   'noise_correlation': sum(noise_correlation) / len(noise_correlation)}
    else:
        parameters = {'dummy parama':[]}
        metrics = {'dummy_metric': []}

    if predict:
        metrics['pred_mse'] = sum(pred_mse) / len(pred_mse)
        # metrics['pred_psnr'] = sum(pred_psnr)/len(pred_psnr)
        # metrics['pred_ssim'] = sum(pred_ssim)/len(pred_ssim)

    return objectives, grads, parameters, imgs, metrics

def train(data, model, optimizer, eval_length, epoch_size=0):
    print('Training...')
    t_start = time.time()
    model.train()
    results = train_val(data, model, optimizer, eval_length=eval_length, epoch_size=epoch_size)
    print('Duration: ' + '{:.2f}'.format(time.time() - t_start) + ' s.')
    return results

def validation(data, model, eval_length, epoch_size=0, use_mean_pred=False):
    print('Validation...')
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        results = train_val(data, model, predict=True, eval_length=eval_length, epoch_size=epoch_size, use_mean_pred=use_mean_pred)
    print('Duration: ' + '{:.2f}'.format(time.time() - t_start) + ' s.')
    return results
