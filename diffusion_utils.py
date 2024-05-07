# ---------------------------------------------------------------
# This file has been modified from Score-based-ECG-Denoising.
#
# Source:
# https://github.com/HuayuLiArizona/Score-based-ECG-Denoising/blob/main/utils.py
#
# ---------------------------------------------------------------

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import metrics
import pandas as pd


def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername="", min_lr=1e-7):
    optimizer = Adam(model.parameters(), lr=config["lr"])
    # ema = EMA(0.9)
    # ema.register(model)

    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=150, gamma=.1, verbose=True
    )

    best_valid_loss = np.Inf
    history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr'])

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        train_desc = "Epoch {:2d}: train - Loss: {:.6f}"
        train_bar = tqdm(initial=0, leave=True, total=len(train_loader),
                         desc=train_desc.format(epoch_no, 0, 0), position=0)
        for batch_no, (noisy_batch, clean_batch) in enumerate(train_loader, 1):
            noisy_batch = noisy_batch.reshape(noisy_batch.shape[0], 1, noisy_batch.shape[1])
            clean_batch = clean_batch.reshape(clean_batch.shape[0], 1, clean_batch.shape[1])
            noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)
            optimizer.zero_grad()

            loss = model(clean_batch, noisy_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            avg_loss += loss.item()

            # ema.update(model)
            train_loss = avg_loss / batch_no
            train_bar.desc = train_desc.format(epoch_no, train_loss)
            train_bar.update(1)
        train_bar.close()
        # lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            valid_desc = "Epoch {:2d}: valid - Loss: {:.6f}"
            valid_bar = tqdm(initial=0, leave=True, total=len(valid_loader),
                             desc=valid_desc.format(epoch_no, 0, 0), position=0)
            with torch.no_grad():
                for batch_no, (noisy_batch, clean_batch) in enumerate(valid_loader, 1):
                    noisy_batch = noisy_batch.reshape(noisy_batch.shape[0], 1, noisy_batch.shape[1])
                    clean_batch = clean_batch.reshape(clean_batch.shape[0], 1, clean_batch.shape[1])
                    noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)
                    loss = model(clean_batch, noisy_batch)
                    avg_loss_valid += loss.item()
                    valid_loss = avg_loss_valid / batch_no

                    valid_bar.desc = valid_desc.format(epoch_no, valid_loss)
                    valid_bar.update(1)
                valid_bar.close()
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                print("\n best loss is updated to ", valid_loss, "at", epoch_no, )

                if foldername != "":
                    torch.save({'epoch': epoch_no,
                                'model': model.state_dict(),
                                'valid_loss': valid_loss,
                                'optimizer': optimizer.state_dict()},
                               output_path)

            # Get learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        if learning_rate < min_lr:
            break
        # Print message
        tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                   '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                   .format(epoch_no, train_loss, valid_loss, learning_rate))
        # Save history
        history = history.append({"epoch": epoch_no, "train_loss": train_loss,
                                  "valid_loss": valid_loss, "lr": learning_rate}, ignore_index=True)
        history.to_csv(foldername + 'history.csv', index=False)
        # Update learning rate
        lr_scheduler.step()

    torch.save(model.state_dict(), final_path)


def evaluate(model, test_loader, shots, device, foldername):
    ssd_total = []
    mad_total = []
    prd_total = []
    cos_sim_total = []
    snr_noise = []
    snr_recon = []
    snr_improvement = []
    performance = pd.DataFrame(columns=['shots', 'ssd', 'mad', 'prd', 'cos_sim', 'snr_in', 'snr_out', 'snr_improve'])

    restored_sig = []
    for batch_no, (noisy_batch, clean_batch) in enumerate(test_loader, 1):
        noisy_batch = noisy_batch.reshape(noisy_batch.shape[0], 1, noisy_batch.shape[1])
        clean_batch = clean_batch.reshape(clean_batch.shape[0], 1, clean_batch.shape[1])
        noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)
        if shots > 1:
            output = 0
            for i in range(shots):
                output += model.denoising(noisy_batch)
            output /= shots
        else:
            output = model.denoising(noisy_batch)  # B,1,L
        output = output - torch.Tensor.mean(output, axis=2)[:, None]
        clean_batch = clean_batch.permute(0, 2, 1)
        noisy_batch = noisy_batch.permute(0, 2, 1)
        output = output.permute(0, 2, 1)  # B,L,1
        out_numpy = output.cpu().detach().numpy()
        clean_numpy = clean_batch.cpu().detach().numpy()
        noisy_numpy = noisy_batch.cpu().detach().numpy()

        ssd_total.append(metrics.SSD(clean_numpy, out_numpy))
        mad_total.append(metrics.MAD(clean_numpy, out_numpy))
        prd_total.append(metrics.PRD(clean_numpy, out_numpy))
        cos_sim_total.append(metrics.COS_SIM(clean_numpy, out_numpy))
        snr_noise.append(metrics.SNR(clean_numpy, noisy_numpy))
        snr_recon.append(metrics.SNR(clean_numpy, out_numpy))
        snr_improvement.append(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
        restored_sig.append(out_numpy)

    ssd_total = np.concatenate(ssd_total, axis=0)
    mad_total = np.concatenate(mad_total, axis=0)
    prd_total = np.concatenate(prd_total, axis=0)
    cos_sim_total = np.concatenate(cos_sim_total, axis=0)
    snr_noise = np.concatenate(snr_noise, axis=0)
    snr_recon = np.concatenate(snr_recon, axis=0)
    snr_improvement = np.concatenate(snr_improvement, axis=0)
    restored_sig = np.concatenate(restored_sig)

    # np.save(foldername / 'denoised.npy', restored_sig)
    performance = pd.concat([performance, pd.DataFrame({"shots": shots, "ssd": ssd_total.mean(),
                                                        "mad": mad_total.mean(), "prd": prd_total.mean(),
                                                        'cos_sim': cos_sim_total.mean(),
                                                        'snr_in': snr_noise.mean(),
                                                        'snr_out': snr_recon.mean(),
                                                        'snr_improve': snr_improvement.mean()}, index=['mean'])])
    performance = pd.concat([performance, pd.DataFrame({"shots": shots, "ssd": ssd_total.std(),
                                                        "mad": mad_total.std(), "prd": prd_total.std(),
                                                        'cos_sim': cos_sim_total.std(),
                                                        'snr_in': snr_noise.std(),
                                                        'snr_out': snr_recon.std(),
                                                        'snr_improve': snr_improvement.std()}, index=['std'])])

    print('******************' + str(shots) + '-shots' + '******************')
    print('******************ALL******************')
    print("ssd: ", ssd_total.mean(), '$\pm$', ssd_total.std(), )
    print("mad: ", mad_total.mean(), '$\pm$', mad_total.std(), )
    print("prd: ", prd_total.mean(), '$\pm$', prd_total.std(), )
    print("cos_sim: ", cos_sim_total.mean(), '$\pm$', cos_sim_total.std(), )
    print("snr_in: ", snr_noise.mean(), '$\pm$', snr_noise.std(), )
    print("snr_out: ", snr_recon.mean(), '$\pm$', snr_recon.std(), )
    print("snr_improve: ", snr_improvement.mean(), '$\pm$', snr_improvement.std(), )
    performance.T.to_csv(foldername / 'performance.csv', mode='a', header=False)
