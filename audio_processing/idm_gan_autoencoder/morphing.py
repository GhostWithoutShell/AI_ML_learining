import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import math


class IDMTensorDataset(Dataset):
    def __init__(self, tensor_folder, slice_len=3, sr=22050, hop_length=128):
        self.files = glob.glob(os.path.join(tensor_folder, '*.pt'))

        if len(self.files) == 0:
             self.files = glob.glob(os.path.join(tensor_folder, '**', '*.pt'), recursive=True)
             
        self.slice_pixels = int((slice_len * sr) / hop_length)
        self.slice_pixels = self.slice_pixels - (self.slice_pixels % 4)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        full_tensor = torch.load(path)
        _, _, width = full_tensor.shape
        target_width = self.slice_pixels
        
        if width > target_width:
            start = np.random.randint(0, width - target_width)
            crop = full_tensor[:, :, start : start + target_width]
        else:
            crop = torch.zeros((1, 64, target_width))
            crop[:, :, :width] = full_tensor
        return crop

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class IDMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 64 * 16
        self.pos_emb = PositionalEncoding(d_model=self.embed_dim, max_len=2000)
        
        # ENCODER (Strided Conv)
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # TRANSFORMER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # DECODER (PixelShuffle)
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16 * 4, kernel_size=3, padding=1), 
            nn.PixelShuffle(upscale_factor=2), 
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        pass 



def run_morphing(model, dataset, device):
    print("Генерирую морфинг...")
    model.eval()
    
    
    idx1 = np.random.randint(0, len(dataset))
    idx2 = np.random.randint(0, len(dataset))
    while idx1 == idx2: idx2 = np.random.randint(0, len(dataset))
    
    track_A = dataset[idx1].unsqueeze(0).to(device)
    track_B = dataset[idx2].unsqueeze(0).to(device)
    
    with torch.no_grad():

        

        def get_latent(x):
            
            feat = model.encoder_cnn(x)
            b, c, h, w = feat.shape
            
            flat = feat.permute(0, 3, 1, 2).reshape(b, w, c*h)
            
            flat = model.pos_emb(flat)
            latent = model.transformer(flat)
            return latent, (b, c, h, w)

        latent_A, shape_info = get_latent(track_A)
        latent_B, _ = get_latent(track_B)
        
        b, c, h, w = shape_info

        
        
        
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = []
        
        for alpha in alphas:
    
            latent_mix = (1 - alpha) * latent_A + alpha * latent_B
            
    
    
            latent_reshaped = latent_mix.reshape(b, w, c, h).permute(0, 2, 3, 1)
    
            reconstructed = model.decoder_cnn(latent_reshaped)
            
    
            img = reconstructed.squeeze().cpu().numpy()
            results.append(img)

    
    plt.figure(figsize=(20, 6))
    titles = ["Track A (Source)", "25%", "50% (Hybrid IDM)", "75%", "Track B (Target)"]
    
    for i, img in enumerate(results):
        plt.subplot(1, 5, i+1)
        plt.title(titles[i])
        plt.imshow(img, aspect='auto', origin='lower', cmap='magma')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('morphing_result_gan.png')
    print("Готово! Результат сохранен в 'morphing_result_gan.png'")
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    

    model = IDMAutoencoder().to(device)
    
    
    weights_path = 'idm_generator_gan.pth' 
    
    if os.path.exists(weights_path):
        print(f"Загружаю веса из {weights_path}...")
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("Веса успешно загружены!")
        except Exception as e:
            print(f"Ошибка загрузки весов: {e}")
            print("Убедись, что архитектура в этом скрипте совпадает с той, на которой обучали!")
            exit()
    else:
        print(f"Файл {weights_path} не найден! Проверь имя файла.")
        exit()

    
    dataset = IDMTensorDataset(tensor_folder='tensors/valid', slice_len=3)
    
    if len(dataset) > 0:
        run_morphing(model, dataset, device)
    else:
        print("Датасет пуст или не найден.")