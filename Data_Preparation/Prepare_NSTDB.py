# ---------------------------------------------------------------
# This file has been modified from Score-based-ECG-Denoising.
#
# Source:
# https://github.com/HuayuLiArizona/Score-based-ECG-Denoising/blob/main/Data_Preparation/Prepare_NSTDB.py
#
# ---------------------------------------------------------------

import wfdb
import _pickle as pickle
from scipy.signal import resample_poly
import math
import pathlib


def resample_noise(signal, orig_fs, actual_fs):
    L = math.ceil(len(signal) * actual_fs / orig_fs)
    # Padding data to avoid edge effects caused by resample
    normBeat = list(reversed(signal)) + list(signal) + list(reversed(signal))

    # resample
    res = resample_poly(normBeat, actual_fs, orig_fs)
    res = res[L - 1:2 * L - 1]

    return res


def prepare(path_to_data):
    path_to_data = pathlib.PurePath(path_to_data)
    bw_signals, bw_fields = wfdb.rdsamp(path_to_data / 'mit-bih-noise-stress-test-database-1.0.0/' / 'bw')
    em_signals, em_fields = wfdb.rdsamp(path_to_data / 'mit-bih-noise-stress-test-database-1.0.0/' / 'em')
    ma_signals, ma_fields = wfdb.rdsamp(path_to_data / 'mit-bih-noise-stress-test-database-1.0.0/' / 'ma')

    orig_fs = 360
    actual_fs = 400

    bw_signals_re = resample_noise(bw_signals, orig_fs, actual_fs)
    em_signals_re = resample_noise(em_signals, orig_fs, actual_fs)
    ma_signals_re = resample_noise(ma_signals, orig_fs, actual_fs)

    for key in bw_fields:
        bw_fields['fs'] = actual_fs
        bw_fields['sig_len'] = len(bw_signals_re)
        print(key, bw_fields[key])

    for key in em_fields:
        em_fields['fs'] = actual_fs
        em_fields['sig_len'] = len(em_signals_re)
        print(key, em_fields[key])

    for key in ma_fields:
        ma_fields['fs'] = actual_fs
        ma_fields['sig_len'] = len(ma_signals_re)
        print(key, ma_fields[key])

    # Save Data
    with open(path_to_data / 'NoiseBWL.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump([bw_signals_re, em_signals_re, ma_signals_re], output)
    print('=========================================================')
    print('MIT BIH data noise stress test database (NSTDB) saved as pickle')
