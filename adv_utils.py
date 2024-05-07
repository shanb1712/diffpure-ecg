# ---------------------------------------------------------------
# This file has been modified from DiffPure.
#
# Source:
# https://github.com/NVlabs/DiffPure/utils.py
#
# ---------------------------------------------------------------

import sys
import argparse
from typing import Any
import pathlib
import h5py
import numpy as np
import os
import pandas as pd

import torch
import torch.nn as nn
import models as models
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchmetrics import AveragePrecision
from classifier_utils import score_fun, load_annotators, get_scores


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def update_state_dict(state_dict, idx_start=9):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name] = v

    return new_state_dict


# ------------------------------------------------------------------------
def run_inference(model, x_orig, bs=64, device=torch.device('cuda')):
    n_batches = x_orig.shape[0] // bs
    y_pred_all = []
    outputs_all = []
    for counter in range(n_batches+1):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        y_pred, outputs = model(x, mode='classify')
        y_pred_all.append(y_pred)
        outputs_all.append(outputs)
    return torch.cat(y_pred_all, dim=0), torch.cat(outputs_all, dim=0)


def report_performance(y_orig, y_pred, outputs, output_path, tag, print_precision=False):
    diagnosis = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
    nclasses = y_orig.shape[-1]
    acc = y_orig.eq(y_pred).all(1).sum()
    # Get micro average precision
    print('Accuracy: {:.2%}'.format((acc / y_orig.shape[0]).item()))
    if print_precision:
        average_precision = AveragePrecision(task="multilabel", num_labels=nclasses, average='micro')
        micro_avg_precision = average_precision(outputs.detach().cpu(), y_orig.detach().cpu().int())
        print('Micro average precision:  {:.2%}'.format(micro_avg_precision.item()))

    scores = get_scores(y_orig.detach().cpu(), y_pred.detach().cpu(), score_fun)
    scores_df = pd.DataFrame(scores, index=diagnosis, columns=score_fun.keys())
    scores_df.to_csv(f"{output_path}/{tag}_scores.csv", float_format='%.3f')
    return (acc / y_orig.shape[0]).item()


def get_signal_classifier(classifier_name, config, classifier_path, device=None):
    """
    This function was modified from get_image_classifier()
    """

    class _Wrapper_ResNet(nn.Module):
        def __init__(self, resnet, device):
            super().__init__()
            self.resnet = resnet
            if device is None:
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.device = device
            thrsh_path = f'{classifier_path}thresholds.npy'
            if not os.path.exists(thrsh_path):
                raise FileNotFoundError(f'{thrsh_path} does not exist\n'
                                        f'Evaluate classifier with run_clasifier.py first to get thresholds.')

            else:
                self.thresholds = torch.from_numpy(np.load(thrsh_path)).float().to(device)

        def get_logit(self, output):
            mask = output > self.thresholds
            y = torch.zeros_like(output, dtype=int)
            y[mask] = 1
            # get y labels as categorical
            # temp = 1 * (y.sum(dim=1) == 0)[:, None]
            # y = torch.cat((y, temp), dim=1)
            # for those with multiple arrhythmias, get first
            return y  # torch.argmax(y, dim=1)[:, None]

        def forward(self, x):
            if len(x.shape) > 3:
                x = x.squeeze(dim=0)
            output = self.resnet(x)
            y = self.get_logit(output)
            return y, output

    if 'resnetEcg' in classifier_name:
        print('using resnetEcg...')
        classifier_path = f"{classifier_path}resnet_model/"
        model = models.load_resnetEcg(config, classifier_path).eval()
    wrapper_resnet = _Wrapper_ResNet(model, device)
    return wrapper_resnet


def load_data(args):
    if 'tnmg' in args.domain:
        path_to_database = pathlib.PurePath(args.path_to_database)
        path_to_test_files = path_to_database / 'data'
        path_to_annotators = path_to_test_files / 'csv_files'
        with h5py.File(path_to_test_files / 'ecg_tracings.hdf5', "r") as f:
            x = np.array(f['tracings'])
            x = x[:, :, 5].reshape(x.shape[0], 1, x.shape[1])
        y_true, _, _, _, _, _ = load_annotators(path_to_annotators)
        x = torch.FloatTensor(x)

        y = torch.FloatTensor(y_true)

        n_samples = len(y_true)
        val_set = TensorDataset(x, y)
        val_loader = DataLoader(val_set, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    else:
        raise NotImplementedError(f'Unknown domain: {args.domain}!')

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return x_val, y_val


@torch.no_grad()
def save_signal(tensor, fp, format=None, fs=1, xlim_range=None) -> None:
    if len(tensor.size()) > 2:
        tensor = tensor.squeeze(1)
    t_shape = tensor.size()
    ndarr = tensor.to("cpu", torch.float).numpy()
    tt = np.array(list(range(t_shape[-1]))) / fs
    fig, axs = plt.subplots(t_shape[0], sharex=True)
    if t_shape[0] > 1:
        for ax, sig in zip(axs, ndarr):
            ax.plot(tt, sig)
            if xlim_range is not None:
                ax.set_xlim(xlim_range)
    else:
        axs.plot(tt, ndarr[0])
        if xlim_range is not None:
            axs.set_xlim(xlim_range)
    fig.text(0.5, 0.04, 'Time [sec]', ha='center')
    fig.text(0.04, 0.5, 'Amplitude [mv]', va='center', rotation='vertical')
    fig.savefig(fp, format=format)
