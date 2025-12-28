import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

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
        self.pos_emb = PositionalEncoding(d_model=self.embed_dim, max_len=1000)
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(7, 3), stride=1, padding=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(7, 3), stride=1, padding=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        features = self.encoder_cnn(x) 
        b, c, h, w = features.shape
        features_flat = features.permute(0, 3, 1, 2).reshape(b, w, c*h) 
        features_flat = self.pos_emb(features_flat)
        latent = self.transformer(features_flat)
        latent_reshaped = latent.reshape(b, w, c, h).permute(0, 2, 3, 1)
        reconstructed = self.decoder_cnn(latent_reshaped)
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
    plt.savefig(f"results/epoch_{epoch}.png")
    plt.close()
    model.train()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    train_dataset = IDMTensorDataset(tensor_folder='tensors/train', slice_len=3)
    val_dataset = IDMTensorDataset(tensor_folder='tensors/valid', slice_len=3)
    
    
    
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    model = IDMAutoencoder().to(device)
    criterion = torch.nn.L1Loss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=6, verbose=True)
    epochs = 100

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            output = model(batch)
            loss = criterion(output, batch)
            
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            train_loss += loss.item()
            
        # Validation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)
        avg_train = train_loss / len(dataloader)
        avg_val = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        if (epoch + 1) % 5 == 0:
            visualize_results(model, train_dataset, device, epoch+1)
            torch.save(model.state_dict(), 'idm_autoencoder_fast.pth')