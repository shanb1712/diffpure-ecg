import argparse
import torch
import yaml
import pathlib
from dataloaders import Diffusion_Dataset
import os
import pandas as pd
from Data_Preparation.data_preparation import Load_Data, Load_Noise
from tqdm import tqdm
import h5py
import numpy as np
from diffusion_model import DDPM
from denoising_model_small import ConditionalModel
from diffusion_utils import train, evaluate

from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM for ECG")
    parser.add_argument("--config", type=str, default="diffusion_base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')
    parser.add_argument('--data_portion', type=int, default=1000, help='amount of data to fetch')
    parser.add_argument('--path_to_database', type=str,
                        help='path to folder containing tnmg database')
    parser.add_argument('--train', action='store_true',
                        help='train the classifier from scratch (default: False)')
    args = parser.parse_args()
    print(args)
    
    config_path = "./config/" + args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    foldername = pathlib.PurePath(f"./check_points/noise_type_{args.n_type}/")
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    # Database paths
    path_to_database = pathlib.PurePath(args.path_to_database)
    path_to_hdf5 = path_to_database / "ecg-traces/ecg-traces/preprocessed"
    path_to_csv = path_to_database / "ecg-traces/ecg-traces"
    path_to_test_files = path_to_database / 'data'

    tqdm.write("Define model...")
    base_model = ConditionalModel(config['train']['feats']).to(args.device)
    model = DDPM(base_model, config, args.device)

    if args.train:
        tqdm.write("Building data loaders...")
        params = {'batch_size': config['train']['batch_size'],
                  'shuffle': True,
                  'num_workers': 16}
        x, y, idx_train, idx_val = Load_Data(path_to_hdf5, path_to_csv, portion=args.data_portion)
        noise_train, noise_test = Load_Noise(noise_version=args.n_type)

        training_set = Diffusion_Dataset(x, noise_train, idx_train)
        training_generator = DataLoader(training_set, **params)

        validation_set = Diffusion_Dataset(x, noise_test, idx_val)
        validation_generator = DataLoader(validation_set, **params)

        tqdm.write("Training...")
        train(model, config['train'], train_loader=training_generator, device=args.device,
              valid_loader=validation_generator, valid_epoch_interval=1, foldername=foldername)
    else:
        shot_sg = [1, 3, 5, 10]
        #eval final
        # load model checkpoint
        output_path = foldername / "model.pth"
        model.load_state_dict(torch.load(str(output_path)))

        # get data
        # Import data
        with h5py.File(path_to_test_files / 'ecg_tracings.hdf5', "r") as f:
            x = np.array(f['tracings'])
        params = {'batch_size': 50,
                  'num_workers': 0,
                  'shuffle': False}
        _, noise_test = Load_Noise(noise_test_len=len(x), noise_version=args.n_type)
        test_set = Diffusion_Dataset(x, noise_test, list(range(len(x))), test=True)
        test_generator = DataLoader(test_set, **params)

        print(f'Noise level: {args.n_type}')
        performance_foldername = pathlib.PurePath(f"./performance/noise_type_{args.n_type}/")
        print('Saving scores to folder:', foldername)
        os.makedirs(performance_foldername, exist_ok=True)
        with open(performance_foldername / 'performance.csv', 'w') as f:
            f.write('\t'.join([' , mean, std']))

        for shots in shot_sg:
            print(f'Evaluating for {shots}-shot...')
            evaluate(model, test_generator, shots, args.device, foldername=performance_foldername)




    
    
    
    
    
    
    
