# ---------------------------------------------------------------
# This file has been modified from Score-based-ECG-Denoising.
#
# Source:
# https://github.com/HuayuLiArizona/Score-based-ECG-Denoising/blob/main/Data_Preparation/Data_preparation.py
#
# ---------------------------------------------------------------
import numpy as np
import _pickle as pickle
import pathlib
import h5py
import pandas as pd


# from Data_Preparation import Prepare_NSTDB
from classifier_utils import get_data


def Load_Data(path_to_hdf5, path_to_csv, portion=None):
    # Get tracings
    path_to_hdf5 = pathlib.PurePath(path_to_hdf5)
    path_to_csv = pathlib.PurePath(path_to_csv)

    f = h5py.File(path_to_hdf5 / 'traces.hdf5', "r")
    x = f['signal']

    traces_ids = np.array(f['id_exam'])[:portion]

    # Get annotations
    y_csv = pd.read_csv(path_to_csv / 'annotations.csv')
    y = get_data(y_csv, traces_ids)
    # Get ids that are used for training the classifier
    idx = np.arange(len(traces_ids))
    partition = {'validation': idx[-round(0.02 * len(idx)):],
                 'train': idx[:len(idx) - round(0.02 * len(idx))]}
    idx_val = np.sort(np.asarray(partition['validation']))
    idx_train = np.sort(np.asarray(partition['train']))
    return x, y, idx_train, idx_val


def Load_Noise(noise_test_len=0, noise_version=1, path_to_data='./data/', path_to_save='./check_points/', prepare=False,):
    path_to_data = pathlib.PurePath(path_to_data)
    print('Getting the Data ready ... ')

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)
    # if prepare:
    #     Prepare_NSTDB.prepare(path_to_data)

    # Load NSTDB
    with open(path_to_data / 'NoiseBWL.pkl', 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    [bw_signals, _, _] = nstdb
    # [_, em_signals, _ ] = nstdb
    # [_, _, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)

    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0] / 2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0] / 2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0] / 2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0] / 2):-1, 1]

    #####################################
    # Data split
    #####################################
    if noise_version == 1:
        noise_test = bw_noise_channel2_b
        noise_train = bw_noise_channel1_a
    elif noise_version == 2:
        noise_test = bw_noise_channel1_b
        noise_train = bw_noise_channel2_a
    else:
        raise Exception("Sorry, noise_version should be 1 or 2")
    if noise_test_len > 0:
        rnd_test = np.random.randint(low=20, high=200, size=noise_test_len) / 100
        # Saving the random array so we can use it on the amplitude segmentation tables
        path_to_save = pathlib.PurePath(path_to_save)
        np.save(path_to_save / f'noise_type_{noise_version}' / 'rnd_test.npy', rnd_test)
        print('rnd_test shape: ' + str(rnd_test.shape))

    return noise_train, noise_test
