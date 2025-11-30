import torch
import torch.nn as nn
import numpy as np
import librosa as lr

def generate_pink_noise(shape):
    """
    Генерирует розовый шум (1/f), похожий на звук воды/дождя.
    """
    # Генерируем белый шум в частотной области
    white = np.random.randn(*shape)
    
    # Преобразуем Фурье
    fft_vals = np.fft.rfft(white)
    
    # Создаем фильтр 1/f
    freqs = np.fft.rfftfreq(shape[-1])
    # Избегаем деления на 0 на нулевой частоте
    scale = 1.0 / (np.where(freqs == 0, float('inf'), freqs) ** 0.5)
    scale[0] = 0 # Убираем DC component (постоянное смещение)
    
    # Применяем фильтр и делаем обратное преобразование
    pink_fft = fft_vals * scale
    pink_noise = np.fft.irfft(pink_fft, n=shape[-1])
    
    # Нормализуем, чтобы шум не "взрывал" веса
    pink_noise = (pink_noise - np.mean(pink_noise)) / np.std(pink_noise)
    return torch.tensor(pink_noise, dtype=torch.float32)
 
class WaterNoiseLayer(nn.Module):
    def __init__(self, intensity=0.1):
        super().__init__()
        self.intensity = intensity

    def forward(self, x):
        # Применяем только во время обучения!
        if self.training:
            # Генерируем шум той же формы, что и входные данные
            # (Можно оптимизировать, генерируя батчами на GPU)
            noise = generate_pink_noise(x.shape).to(x.device)
            
            # Вариант: добавляем шум
            return x + (noise * self.intensity)
        return x



class AphexNoiseLayer(nn.Module):
    def __init__(self, audio_path, intensity=0.1, device='cpu'):
        super().__init__()
        self.intensity = intensity
        self.device = device
        
        # 1. Загружаем трек
        # sr=None сохраняет оригинальную частоту дискретизации
        # mono=True смешивает стерео в один канал (так проще для весов)
        print(f"Loading Aphex Twin buffer from {audio_path}...")
        y, sr = lr.load(audio_path, sr=None, mono=True)
        
        # 2. Нормализация (критически важно!)
        # Звук должен быть около 0 с разбросом 1, иначе веса "взорвутся"
        y = (y - np.mean(y)) / np.std(y)
        
        # Превращаем в тензор и сохраняем в буфер (не как обучаемый параметр!)
        # register_buffer гарантирует, что это сохранится вместе с моделью, 
        # но градиенты по этому тензору считаться не будут.
        self.register_buffer('audio_buffer', torch.from_numpy(y).float().to(device))
        
        print(f"Buffer loaded. Length: {len(self.audio_buffer)} samples.")

    def forward(self, x):
        # Применяем только на этапе обучения
        if self.training:
            # Нам нужно столько же точек шума, сколько элементов во входном тензоре
            num_elements = x.numel()
            
            # Если трек короче, чем данные (вряд ли, но вдруг), зацикливаем
            if num_elements > len(self.audio_buffer):
                noise = self.audio_buffer.repeat((num_elements // len(self.audio_buffer)) + 1)
                noise = noise[:num_elements]
            else:
                # Выбираем случайную позицию старта в треке
                # Это "DJ-метод": берем случайный семпл из песни
                max_start = len(self.audio_buffer) - num_elements
                start_idx = np.random.randint(0, max_start)
                noise = self.audio_buffer[start_idx : start_idx + num_elements]
            
            # Придаем шуму форму входных данных (batch_size, features...)
            noise = noise.view(x.shape).to(self.device)
            
            # Добавляем к сигналу (Additive Noise)
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
# Пример использования

#input_data = torch.randn(10, 100) # Батч 10, 100 признаков
#output = layer(input_data)