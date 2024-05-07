# ---------------------------------------------------------------
# This file has been modified from automatic-ecg-diagnosis.
#
# Source:
# https://github.com/antonior92/automatic-ecg-diagnosis/blob/tensorflow-v1/train.py
#
# ---------------------------------------------------------------

import json
import pathlib
import torch
import os
from tqdm import tqdm
from resnetEcg import ResNet1d
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import h5py
from dataloaders import Dataset
from Data_Preparation.data_preparation import Load_Data
from classifier_utils import train, evaluate, report_performance


if __name__ == "__main__":
    import pandas as pd
    import argparse
    from warnings import warn

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict arrhythmias from the raw ecg tracing.')
    parser.add_argument("--config", type=str, default="resnet_base.json",
                        help="model hyperparameters")
    parser.add_argument('--device', default='cuda:1', help='Device')
    parser.add_argument('--path_to_database', type=str,
                        help='path to folder containing tnmg database')
    parser.add_argument('--train', action='store_true',
                        help='train the classifier from scratch (default: False)')
    parser.add_argument('--n_leads', type=int, default=1,
                        help='how many leads to train on, choose between [1,12] (default: lead 6)')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--sample_freq', type=int, default=400,
                        help='sample frequency (in Hz) in which all traces will be resampled at (default: 400)')
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='size (in # of samples) for all traces. If needed traces will be zeropadded'
                             'to fit into the given size. (default: 4096)')
    parser.add_argument('--scale_multiplier', type=int, default=10,
                        help='multiplicative factor used to rescale inputs.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (default: 64).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument("--patience", type=int, default=12,
                        help='maximum number of epochs without reducing the learning rate (default: 12)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    parser.add_argument('--dropout_rate', type=float, default=0.8,
                        help='dropout rate (default: 0.8).')
    parser.add_argument('--kernel_size', type=int, default=17,
                        help='kernel size in convolutional layers (default: 17).')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    torch.manual_seed(args.seed)
    print(args)
    # Set device
    device = torch.device(args.device)

    # Save config file
    config_path = "./config/" + args.config
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent='\t')

    foldername = pathlib.PurePath("./check_points/resnet_model")
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    # Database paths
    path_to_database = pathlib.PurePath(args.path_to_database)
    path_to_hdf5 = path_to_database / "ecg-traces/ecg-traces/preprocessed"
    path_to_csv = path_to_database / "ecg-traces/ecg-traces"
    path_to_test_files = path_to_database / 'data'

    tqdm.write("Building data loaders...")
    x, y, idx_train, idx_val = Load_Data(path_to_hdf5, path_to_csv)

    if args.train:
        params = {'batch_size': args.batch_size,
                  'shuffle': True}
        training_set = Dataset(x, y, idx_train)
        training_generator = DataLoader(training_set, **params)

        validation_set = Dataset(x, y, idx_val)
        validation_generator = DataLoader(validation_set, **params)

        tqdm.write("Define model...")
        N_LEADS = args.n_leads
        N_CLASSES = 6
        model = ResNet1d(input_dim=(N_LEADS, args.seq_length),
                         blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                         n_classes=N_CLASSES,
                         kernel_size=args.kernel_size,
                         dropout_rate=args.dropout_rate)
        model.to(device=device)

        tqdm.write("Define loss...")
        criterion = nn.BCELoss()

        tqdm.write("Define optimizer...")
        optimizer = optim.Adam(model.parameters(), args.lr)

        tqdm.write("Define scheduler...")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience,
                                                         min_lr=args.lr_factor * args.min_lr,
                                                         factor=args.lr_factor)

        tqdm.write("Training...")
        start_epoch = 0
        best_loss = np.Inf
        history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr'])
        for ep in range(start_epoch, args.epochs):
            train_loss = train(model, ep, training_generator, device, criterion, optimizer)
            valid_loss = evaluate(model, ep, validation_generator, device, criterion)
            # Save best model
            if valid_loss < best_loss:
                # Save model
                torch.save({'epoch': ep,
                            'model': model.state_dict(),
                            'valid_loss': valid_loss,
                            'optimizer': optimizer.state_dict()},
                           str(foldername / 'model.pth'))
                # Update best validation loss
                best_loss = valid_loss
            # Get learning rate
            for param_group in optimizer.param_groups:
                learning_rate = param_group["lr"]
            # Interrupt for minimum learning rate
            if learning_rate < args.min_lr:
                break
            # Print message
            tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                       '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                       .format(ep, train_loss, valid_loss, learning_rate))
            # Save history
            history = pd.concat([history, pd.DataFrame({"epoch": ep, "train_loss": train_loss,
                                                        "valid_loss": valid_loss, "lr": learning_rate}, index=[0])],
                                ignore_index=True)
            history.to_csv(foldername / 'history.csv', index=False)
            # Update learning rate
            scheduler.step(valid_loss)
        tqdm.write("Done!")
    else:
        # Get checkpoint
        ckpt = torch.load(str(foldername / "model.pth"), map_location=lambda storage, loc: storage)
        # Get config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        # Get model
        N_LEADS = args.n_leads
        N_CLASSES = 6
        model = ResNet1d(input_dim=(N_LEADS, config_dict['seq_length']),
                         blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
                         n_classes=N_CLASSES,
                         kernel_size=config_dict['kernel_size'],
                         dropout_rate=config_dict['dropout_rate'])
        # load model checkpoint
        model.load_state_dict(ckpt["model"])
        model = model.to(device)

        # get data
        # Import data
        with h5py.File(path_to_test_files / 'ecg_tracings.hdf5', "r") as f:
            x = np.array(f['tracings'])[:, :, 5]

        # Evaluate on test data
        model.eval()

        # Compute gradients
        predicted_y = np.empty((len(x), N_CLASSES))
        end = 0

        print('Evaluating...')
        for batch_no in range(len(x) // config_dict['batch_size'] + 1):
            batch_x = x[batch_no * config_dict['batch_size']: (batch_no + 1) * config_dict['batch_size']]
            batch_x = batch_x.reshape(batch_x.shape[0], 1, batch_x.shape[1])
            batch_x = torch.FloatTensor(batch_x)
            with torch.no_grad():
                batch_x = batch_x.to(device, dtype=torch.float32)
                y_pred = model(batch_x)
            predicted_y[batch_no * config_dict['batch_size']:(batch_no + 1) * config_dict[
                'batch_size']] = y_pred.detach().cpu().numpy()

        # Report and save performance

        report_performance(output_path=pathlib.PurePath('./performance/resnet'),
                           model_path=foldername,
                           path_to_annotators=path_to_test_files / 'csv_files',
                           y_pred=predicted_y)