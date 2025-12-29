import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
from n_layer_descrim import IDMDiscriminator
import math
METRIC = 'mel'  # 'cwt' или 'mel'
class IDMTensorDataset(Dataset):
    def __init__(self, tensor_folder, slice_len=3, sr=22050, hop_length=128):
        
        self.files = glob.glob(os.path.join(tensor_folder, '*.pt'))
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
        
        # Positional Encoding оставляем (он важен)
        self.pos_emb = PositionalEncoding(d_model=self.embed_dim, max_len=2000)
        
        # --- 1. ENCODER (Strided Conv вместо MaxPool) ---
        self.encoder_cnn = nn.Sequential(
            # Вход: [1, 64, W]
            # Сжимаем: stride=(2, 2) уменьшает размер в 2 раза
            nn.Conv2d(1, 32, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(), # GELU работает мягче и лучше для трансформеров
            
            # Вход: [32, 32, W/2]
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.GELU()
            # Выход: [64, 16, W/4]
        )
        
        # --- 2. TRANSFORMER ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=8, # Увеличим кол-во голов для лучшего внимания
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3) # +1 слой для глубины
        
        # --- 3. DECODER (PixelShuffle) ---
        self.decoder_cnn = nn.Sequential(
            # Этап 1: Восстанавливаем из [64, 16, W/4]
            # PixelShuffle(2) уменьшает каналы в 4 раза (r^2), поэтому на входе нужно много каналов
            # Нам нужно на выходе 32 канала. Значит вход PixelShuffle должен быть 32 * 4 = 128.
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2), # [128, H, W] -> [32, 2H, 2W]
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # Этап 2: Восстанавливаем из [32, 32, W/2]
            # Хотим на выходе 1 канал (картинку). 
            # PixelShuffle съест каналы. Сделаем промежуточный слой.
            
            nn.Conv2d(32, 16 * 4, kernel_size=3, padding=1), # Готовим каналы для шафла
            nn.PixelShuffle(upscale_factor=2), # [64, H, W] -> [16, 2H, 2W]
            nn.GELU(),
            
            # Финальная доводка (резкость)
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder_cnn(x) 
        b, c, h, w = features.shape
        
        # [Batch, Channels, Freq, Time] -> [Batch, Time, Channels*Freq]
        features_flat = features.permute(0, 3, 1, 2).reshape(b, w, c*h) 
        
        features_flat = self.pos_emb(features_flat)
        latent = self.transformer(features_flat)
        
        # Обратно
        latent_reshaped = latent.reshape(b, w, c, h).permute(0, 2, 3, 1)
        reconstructed = self.decoder_cnn(latent_reshaped)
        
        # Иногда при Strided Conv размеры могут гулять на 1 пиксель
        if reconstructed.shape != x.shape:
             reconstructed = torch.nn.functional.interpolate(reconstructed, size=x.shape[2:], mode='bilinear')
             
        return reconstructed

def visualize_results(model, dataset, device, epoch):
    model.eval()
    idx = np.random.randint(0, len(dataset))
    
    input_tensor = dataset[idx].unsqueeze(0).to(device) 
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    inp_img = input_tensor.squeeze().cpu().numpy()
    out_img = output_tensor.squeeze().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original IDM")
    plt.imshow(inp_img, aspect='auto', origin='lower', cmap='magma')
    plt.subplot(1, 2, 2)
    plt.title(f"Reconstructed (Epoch {epoch})")
    plt.imshow(out_img, aspect='auto', origin='lower', cmap='magma')
    plt.savefig(f"results_gan/epoch_{epoch}.png")
    plt.close()
    model.train()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training GAN on: {device}")
    root_folder = 'tensors' if METRIC == 'cwt' else 'idm_mels'
    weights = torch.ones(1, 1, 64, 1).to(device)
    train_dataset = IDMTensorDataset(tensor_folder=f'{root_folder}/train', slice_len=3)
    val_dataset = IDMTensorDataset(tensor_folder=f'{root_folder}/valid', slice_len=3)
    for i in range(64):
        weights[:, :, i, :] = 1.0 + (i / 64) * 10.0  # Чем выше частота, тем больше штраф
    
    
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    
    
    generator = IDMAutoencoder().to(device) # Твой AE (желательно PixelShuffle версия)
    discriminator = IDMDiscriminator().to(device)
    criterion_GAN = nn.BCEWithLogitsLoss() # Для критика (Real vs Fake)
    criterion_L1 = nn.L1Loss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #scheduler = ReduceLROnPlateau(
    #    optim, 
    #    mode='min', 
    #    factor=0.5,     
    #    patience=3,     
    #    threshold=1e-3, 
    #    verbose=True
    #)
    epochs = 200
    lambda_l1 = 150.0 
    for epoch in range(epochs):
        train_loss = 0
        generator.train() # Это теперь generator
        discriminator.train()
        
        for i, real_batch in enumerate(dataloader):
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            
            optimizer_D.zero_grad()
            
        
            output_real = discriminator(real_batch)
            label_real = torch.tensor(0.9).expand_as(output_real).to(device)
            loss_D_real = criterion_GAN(output_real, label_real)
            
            
            fake_batch = generator(real_batch) 
            
            output_fake = discriminator(fake_batch.detach())
            label_fake = torch.zeros_like(output_fake).to(device)
            loss_D_fake = criterion_GAN(output_fake, label_fake)
            
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            
            
            output_fake_for_G = discriminator(fake_batch)
            label_real_for_G = torch.ones_like(output_fake_for_G).to(device)
            
            
            loss_G_GAN = criterion_GAN(output_fake_for_G, label_real_for_G)
            
            
            loss_G_L1 = criterion_L1(fake_batch, real_batch) * lambda_l1
            
            
            loss_G = loss_G_GAN + loss_G_L1
            
            loss_G.backward()
            optimizer_G.step()
            
            
            if i % 50 == 0:
                print(f"\rEpoch {epoch+1} [Batch {i}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]", end="")

        print(f"\nEpoch {epoch+1} Finished.")
        
        
        if (epoch + 1) % 5 == 0:
            visualize_results(generator, train_dataset, device, epoch+1)
            torch.save(generator.state_dict(), f'idm_generator_gan_{METRIC}.pth')
            torch.save(discriminator.state_dict(), f'idm_discriminator_{METRIC}.pth')
            