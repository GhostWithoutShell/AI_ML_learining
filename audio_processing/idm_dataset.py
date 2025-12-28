import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import pywt
import os

class IDMDataset(Dataset):
    def __init__(self, folder_path, slice_len=3, sr=22050):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp3') or f.endswith('.wav')]
        self.slice_len = slice_len 
        self.sr = sr
        
        self.wavelet = 'cmor1.5-1.0'
        self.freqs = np.geomspace(20, sr/2, num=64) 
        self.scales = pywt.frequency2scale(self.wavelet, self.freqs / sr)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        path = self.files[idx]
        y, _ = librosa.load(path, sr=self.sr)
        
        
        target_len = self.slice_len * self.sr
        if len(y) > target_len:
            start = np.random.randint(0, len(y) - target_len)
            y_chunk = y[start : start + target_len]
        else:
        
            y_chunk = np.pad(y, (0, target_len - len(y)))

        # wavelet transformation
        coefs, _ = pywt.cwt(y_chunk, self.scales, self.wavelet, sampling_period=1/self.sr)
        hop_length = 128
        coefs_reduced = coefs[:, ::hop_length]
        
        power = np.abs(coefs_reduced)**2
        log_power = librosa.power_to_db(power, top_db=80) 
        
        
        norm_spec = (log_power - log_power.min()) / (log_power.max() - log_power.min() + 1e-6)
        
        
        
        tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0) 
        width = tensor.shape[-1]

        new_width = width - (width % 4)
        tensor = tensor[..., :new_width]
        return tensor
