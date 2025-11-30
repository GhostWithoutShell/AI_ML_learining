import torch
import torch.nn as nn
import numpy as np
import librosa as lr

def generate_pink_noise(shape):
    """
    Генерирует розовый шум (1/f), похожий на звук воды/дождя.
    """
    
    white = np.random.randn(*shape)
    
    
    fft_vals = np.fft.rfft(white)
    
    
    freqs = np.fft.rfftfreq(shape[-1])
    
    scale = 1.0 / (np.where(freqs == 0, float('inf'), freqs) ** 0.5)
    scale[0] = 0
    
    
    pink_fft = fft_vals * scale
    pink_noise = np.fft.irfft(pink_fft, n=shape[-1])
    
    
    pink_noise = (pink_noise - np.mean(pink_noise)) / np.std(pink_noise)
    return torch.tensor(pink_noise, dtype=torch.float32)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, intensity=0.1):
        super().__init__()
        self.intensity = intensity

    def forward(self, x):
        if self.training:
            
            noise = torch.randn_like(x) 
            return x + (noise * self.intensity)
        return x

class WaterNoiseLayer(nn.Module):
    def __init__(self, intensity=0.1):
        super().__init__()
        self.intensity = intensity

    def forward(self, x):
        
        if self.training:
            
            
            noise = generate_pink_noise(x.shape).to(x.device)
            
            
            return x + (noise * self.intensity)
        return x



class AphexNoiseLayer(nn.Module):
    def __init__(self, audio_path, intensity=0.1, device='cpu'):
        super().__init__()
        self.intensity = intensity
        self.device = device
        
        
        print(f"Loading Aphex Twin buffer from {audio_path}...")
        y, sr = lr.load(audio_path, sr=None, mono=True)
        
        
        y = (y - np.mean(y)) / np.std(y)
        
        
        self.register_buffer('audio_buffer', torch.from_numpy(y).float().to(device))
        
        print(f"Buffer loaded. Length: {len(self.audio_buffer)} samples.")

    def forward(self, x):
        
        if self.training:
            
            num_elements = x.numel()
            
            
            if num_elements > len(self.audio_buffer):
                noise = self.audio_buffer.repeat((num_elements // len(self.audio_buffer)) + 1)
                noise = noise[:num_elements]
            else:
                
                
                max_start = len(self.audio_buffer) - num_elements
                start_idx = np.random.randint(0, max_start)
                noise = self.audio_buffer[start_idx : start_idx + num_elements]
            
            
            noise = noise.view(x.shape).to(self.device)
            
            
            return x + (noise * self.intensity)
            
        return x

# --- Пример использования ---
#input_tensor = torch.randn(2,5)
#audio_file = "water.mp3" 
##audio_file = "aphex_twin_-_Formula.mp3" 
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#input_tensor = input_tensor.to(device)
#layer = AphexNoiseLayer(audio_file, intensity=0.05, device='cuda' if torch.cuda.is_available() else 'cpu')
#output = layer(input_tensor)
#layer_pink = WaterNoiseLayer(intensity=0.05)
#output_p = layer_pink(input_tensor)
#
#print(output)
#print(input_tensor)
#print(output_p)    

#input_data = torch.randn(10, 100) # Батч 10, 100 признаков
#output = layer(input_data)