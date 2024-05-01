import numpy as np
import torch


class Diffusion_Dataset(torch.utils.data.Dataset):
    def __init__(self, x_set, noise, partition, test=False):
        self.test = test
        self.x_clean = x_set
        self.noise = noise
        self.indices = partition
        if test:
            self.rnd_noise = dict(zip(partition, np.load('rnd_test.npy')))
        self.noise_index = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # x - signal with movement
        # y - clean signal
        trace = self.indices[idx]
        y = self.x_clean[trace, :, 5]
        y = y - (y[0] + y[-1]) / 2
        x = self.add_noise(y, trace)  # noise signal
        return torch.FloatTensor(x), torch.FloatTensor(y)

    def add_noise(self, y, trace):
        # Adding noise to data
        eps = 0.001
        samples = len(y)
        if self.test:
            rnd_noise_level = self.rnd_noise[trace]
        else:
            rnd_noise_level = np.random.randint(low=20, high=200) / 100
        noise = self.noise[self.noise_index:self.noise_index + samples]
        lead_max_value = np.max(y, axis=0) - np.min(y, axis=0) + eps
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / lead_max_value
        alpha = rnd_noise_level / Ase
        signal_noise = y + alpha * noise
        x = signal_noise
        self.noise_index += samples

        if self.noise_index > (len(self.noise) - samples):
            self.noise_index = 0
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_set, y_set, partition):
        self.x, self.y = x_set, y_set
        self.indices = partition

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trace = self.indices[idx]
        x = self.x[trace, :, 5]
        y = self.y[trace]
        return x, np.array(y)
