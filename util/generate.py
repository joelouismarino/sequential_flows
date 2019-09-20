import torch
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

def generate(batch, model, cond_len=5, use_mean_pred=False):
    """
    Conditional generation of frames.
    """
    preds = {'data': [], 'pred': []}

    batch_pred_mse = 0
    batch_pred_psnr = 0
    batch_pred_ssim = 0

    factor = (batch.size(0)-cond_len) * batch.size(1)
    batch_size = batch.size(1)

    batch = batch.to(model.device).clamp(1e-6, 1-1e-6)
    model.reset(batch.shape[1])

    for step_ind, step_data in enumerate(batch):
        # run the model on the data

        preds['data'].append(step_data)

        if step_ind <= cond_len-1:
            # data = step_data
            model(step_data)
            if model.ready():
                pred_data = model.cond_like.dist.mean.view(step_data.size())

        else:
            assert model.ready()
            model.predict(use_mean_pred)
            pred_data = model._prev_x

        if model.ready():
            preds['pred'].append(pred_data.view(step_data.shape).detach().cpu())

            # accumulate metrics
            if step_ind >= cond_len:
                step_np = step_data.cpu().numpy()
                pred_np = pred_data.detach().cpu().numpy()
                batch_pred_mse += torch.sum((step_data.cpu() - pred_data.detach().cpu()) ** 2) / factor
                batch_pred_psnr += sum([peak_signal_noise_ratio(step_np[x], pred_np[x]) for x in range(batch_size)]) / factor
                # batch_pred_ssim += sum([structural_similarity(step_np[x], recon_np[x], multichannel=True) for x in range(batch_size)])/factor

        model.step()

    # preds = {k: torch.stack(v) for k, v in preds.items()}
    # metrics = {'pred_mse': batch_pred_mse, 'pred_psnr': batch_pred_psnr, 'pred_ssim': batch_pred_ssim}

    return preds, batch_pred_mse, batch_pred_psnr
