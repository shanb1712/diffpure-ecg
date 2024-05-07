# ---------------------------------------------------------------
# This file has been modified from DiffPure.
#
# Source:
# https://github.com/NVlabs/DiffPure/blob/master/runners/diffpure_ddpm.py
#
# ---------------------------------------------------------------
import os
import random

import torch
from diffusion_model import DDPM
from denoising_model_small import ConditionalModel
from adv_utils import save_signal


class DeScoDECG(torch.nn.Module):
    def __init__(self, args, config, model_path, noise_type=1, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        # load model
        if config.data.dataset == 'tnmg':
            base_model = ConditionalModel(config.diffusion_train.feats).to(self.device)
            model = DDPM(base_model, config, self.device)

            model_path = f"{model_path}noise_type_{noise_type}/"
            model.load_state_dict(torch.load(f'{model_path}/final.pth', map_location='cpu'))
            model.requires_grad_(False).eval().to(self.device)

        else:
            raise NotImplementedError(f'Unknown dataset {config.data.dataset}!')

        self.model = model
        self.betas = model.betas.cpu().float().to(self.device)

    def signal_editing_sample(self, sig=None, bs_id=0, tag=None):
        with torch.no_grad():
            assert isinstance(sig, torch.Tensor)
            r_batch_size = 4
            if tag is None:
                tag = 'rnd' + str(random.randint(0, 10000))
            out_dir = os.path.join(self.args.log_dir, 'bs' + str(bs_id) + '_' + tag)

            assert sig.ndim == 3, sig.ndim
            sig = sig.to(self.device)
            x0 = sig

            if bs_id < 2:
                os.makedirs(out_dir, exist_ok=True)
                save_signal(x0[:r_batch_size], os.path.join(out_dir, f'original_input.png'), fs=self.config.data.fs)
                save_signal(x0[:r_batch_size], os.path.join(out_dir, f'original_input_zoom_in.png'),
                            fs=self.config.data.fs, xlim_range=[4, 7])

            xs = []
            # for it in range(self.args.sample_step):
                # e = torch.randn_like(x0)
                # total_noise_levels = self.args.t
                # a = (1 - self.betas).cumprod(dim=0)
                # x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                # if bs_id < 2:
                #     save_signal(x[:r_batch_size], os.path.join(out_dir, f'init_{it}.png'), fs=self.config.data.fs)
                #     save_signal(x[:r_batch_size], os.path.join(out_dir, f'init_{it}_zoom_in.png'),
                #                 fs=self.config.data.fs, xlim_range=[4, 7])
            if self.args.sample_step > 1:
                x = 0
                for i in range(self.args.sample_step):
                    x += self.model.denoising(x0)
                x /= self.args.sample_step
            else:
                x = self.model.denoising(x0)  # B,1,L
            x = x - torch.Tensor.mean(x, axis=2)[:, None]

            x0 = x

            if bs_id < 2:
                save_signal(x0[:r_batch_size], os.path.join(out_dir, f'denoised_sample.png'), fs=self.config.data.fs)
                save_signal(x0[:r_batch_size], os.path.join(out_dir, f'denoised_sample_zoom_in.png'),
                            fs=self.config.data.fs, xlim_range=[4, 7])
                torch.save(x0[:r_batch_size], os.path.join(out_dir, f'samples_{it}.pth'))

                xs.append(x0)

            return torch.cat(xs, dim=0)
