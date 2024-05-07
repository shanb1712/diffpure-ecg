# ---------------------------------------------------------------
# This file has been modified from DiffPure.
#
# Source:
# https://github.com/NVlabs/DiffPure/eval_sde_adv_bpda.py
#
# ---------------------------------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import logging
import yaml
import time
import random
import numpy as np
import torch
import torch.nn as nn

import adv_utils
from bpda_eot.bpda_eot_attack import BPDA_EOT_Attack
from adv_utils import str2bool, get_accuracy, get_signal_classifier, load_data
from runners.diffpure_ecg import DeScoDECG


class ResNet_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        # signal classifier
        self.args = args
        self.resnet = get_signal_classifier(classifier_name=args.classifier_name, config=config.classifier,
                                            classifier_path=args.model_path).to(config.device)

    def purify(self, x):
        return x

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            out = self.purify(x)
        elif mode == 'classify':
            out = self.resnet(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            x = self.purify(x)
            out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        self.resnet = get_signal_classifier(classifier_name=args.classifier_name, config=config.classifier,
                                            classifier_path=args.model_path).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}, noise_type: {args.noise_type}')
        if args.diffusion_type == 'DeScoDECG':
            self.runner = DeScoDECG(args, config, model_path=args.model_path,
                                    noise_type=args.noise_type, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
    # and you may want to change the freq)
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def purify(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        start_time = time.time()
        x_re = self.runner.signal_editing_sample(x, bs_id=counter, tag=self.tag)
        minutes, seconds = divmod(time.time() - start_time, 60)

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before resnet): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        self.counter += 1

        return x_re

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            out = self.purify(x)
        elif mode == 'classify':
            out = self.resnet(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            x = self.purify(x)
            out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


def eval_bpda(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module

    x_val, y_val = x_val.to(config.device), y_val.to(config.device)

    # ------------------ apply the attack to resnet ------------------
    print(f'apply the bpda attack to resnet...')
    resnet_bpda = ResNet_Adv_Model(args, config)
    if ngpus > 1:
        resnet_bpda = torch.nn.DataParallel(resnet_bpda)

    start_time = time.time()
    init_acc = get_accuracy(resnet_bpda, x_val, y_val, bs=adv_batch_size, device=config.device)
    print('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))

    adversary_resnet = BPDA_EOT_Attack(resnet_bpda, adv_eps=args.adv_eps, eot_defense_reps=args.eot_defense_reps,
                                       eot_attack_reps=args.eot_attack_reps)

    start_time = time.time()
    class_batch, ims_adv_batch = adversary_resnet.attack_all(x_val, y_val, batch_size=adv_batch_size)
    init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
    robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]

    print('init acc: {:.2%}, robust acc: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, robust_acc,
                                                                               time.time() - start_time))

    print(f'x_adv_resnet shape: {ims_adv_batch.shape}')
    torch.save([ims_adv_batch, y_val], f'{log_dir}/x_adv_resnet_sd{args.seed}.pt')

    # ------------------ apply the attack to sde_adv ------------------
    print(f'apply the bpda attack to sde_adv...')

    start_time = time.time()
    model_.reset_counter()
    model_.set_tag('no_adv')
    init_acc = get_accuracy(model, x_val, y_val, bs=adv_batch_size)
    print('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))

    adversary_sde = BPDA_EOT_Attack(model, adv_eps=args.adv_eps, eot_defense_reps=args.eot_defense_reps,
                                    eot_attack_reps=args.eot_attack_reps)

    start_time = time.time()
    model_.reset_counter()
    model_.set_tag()
    class_batch, ims_adv_batch = adversary_sde.attack_all(x_val, y_val, batch_size=adv_batch_size)
    init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
    robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]

    print('init acc: {:.2%}, robust acc: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, robust_acc,
                                                                               time.time() - start_time))

    print(f'x_adv_sde shape: {ims_adv_batch.shape}')
    torch.save([ims_adv_batch, y_val], f'{log_dir}/x_adv_sde_sd{args.seed}.pt')


def robustness_eval(args, config):
    middle_name = '_'.join([args.diffusion_type, 'bpda'])
    log_dir = os.path.join(args.signal_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = adv_utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(config.device)

    # load data
    x_val, y_val = load_data(args)

    # eval classifier and sde_adv against bpda attack
    eval_bpda(args, config, model, x_val, y_val, adv_batch_size, log_dir)

    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-s', '--signal_folder', type=str, default='signals', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--model_path', type=str, default='./check_points/', help='pretrained models')

    # data
    parser.add_argument('--path_to_database', default='/MLAIM/databases/tnmg/', type=str,
                        help='path to folder containing tnmg database')
    parser.add_argument('--domain', type=str, default='tnmg', help='which domain: tnmg')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--data_seed', type=int, default=2, help='Random seed')

    # diffusion models
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=50, help='Sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='DeScoDECG', help='[DeScoDECG]')
    parser.add_argument('--score_type', type=str, default='DeScoDECG', help='[DeScoDECG]')
    parser.add_argument('--noise_type', type=int, default=1, help='noise type on which diffusion model was trained on')

    # adv
    parser.add_argument('--eot_defense_reps', type=int, default=1)
    parser.add_argument('--eot_attack_reps', type=int, default=1)
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--adv_eps', type=float, default=0.07)

    # classifier
    parser.add_argument('--classifier_name', type=str, default='resnetEcg', help='which classifier to use')

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('config', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = adv_utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.signal_folder = os.path.join(args.exp, args.signal_folder)
    os.makedirs(args.signal_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    args, config = parse_args_and_config()
    robustness_eval(args, config)
