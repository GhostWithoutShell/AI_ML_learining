import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import math
import librosa
import soundfile as sf
from tqdm import tqdm

# ==========================================
# 1. АРХИТЕКТУРА (Твоя последняя версия)
# ==========================================
# (Вставляем классы, чтобы скрипт был автономным)

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
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
            nn.BatchNorm2d(64), nn.GELU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 16 * 4, kernel_size=3, padding=1), 
            nn.PixelShuffle(upscale_factor=2), nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

class IDMTensorDataset(Dataset):
    def __init__(self, tensor_folder, slice_len=3, sr=22050, hop_length=256):
        self.files = glob.glob(os.path.join(tensor_folder, '*.pt'))
        if len(self.files) == 0: self.files = glob.glob(os.path.join(tensor_folder, '**', '*.pt'), recursive=True)
        self.slice_pixels = int((slice_len * sr) / hop_length)
        self.slice_pixels = self.slice_pixels - (self.slice_pixels % 4)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        full_tensor = torch.load(self.files[idx])
        _, _, width = full_tensor.shape
        if width > self.slice_pixels:
            start = np.random.randint(0, width - self.slice_pixels)
            return full_tensor[:, :, start : start + self.slice_pixels]
        return full_tensor[:, :, :self.slice_pixels] # Simplification for demo

# ==========================================
# 2. ГЕНЕРАТОР ИСКУССТВА (Gradient Visualizer)
# ==========================================

def tensor_to_audio(tensor, sr=22050, n_fft=1024, hop_length=256):
    spec = tensor.squeeze().cpu().detach().numpy()
    spec_db = (spec * 80) - 80
    spec_power = librosa.db_to_power(spec_db)
    stft_spec = librosa.feature.inverse.mel_to_stft(spec_power, sr=sr, n_fft=n_fft, power=2.0)
    audio = librosa.griffinlim(stft_spec, n_iter=32, hop_length=hop_length)
    return audio

def generate_av_art(model, dataset, device, frames=120):
    print("🎨 Начинаю создание аудио-визуального арта...")
    os.makedirs("art_frames", exist_ok=True)
    
    # 1. Выбираем треки
    idx1, idx2 = np.random.randint(0, len(dataset)), np.random.randint(0, len(dataset))
    while idx1 == idx2: idx2 = np.random.randint(0, len(dataset))
    
    track_A = dataset[idx1].unsqueeze(0).to(device)
    track_B = dataset[idx2].unsqueeze(0).to(device)
    
    model.eval()
    
    # 2. Получаем начальный и конечный латент
    # Нам нужно вытащить Z (латент) ДО трансформера, чтобы градиенты текли через него
    def get_latent_pre_transformer(x):
        feat = model.encoder_cnn(x)
        b, c, h, w = feat.shape
        flat = feat.permute(0, 3, 1, 2).reshape(b, w, c*h)
        flat = model.pos_emb(flat)
        return flat, (b, c, h, w)

    with torch.no_grad():
        latent_A, shape = get_latent_pre_transformer(track_A)
        latent_B, _ = get_latent_pre_transformer(track_B)
    
    b, c, h, w = shape
    full_audio = []

    # 3. ЦИКЛ ГЕНЕРАЦИИ КАДРОВ
    # Мы будем идти от 0% до 100% морфинга
    alphas = np.linspace(0, 1, frames)
    
    print("🚀 Генерация кадров и градиентов...")
    
    for i, alpha in enumerate(tqdm(alphas)):
        # --- А. СМЕШИВАНИЕ ---
        # Создаем копию латента, для которой будем считать градиенты
        z_mix = (1 - alpha) * latent_A + alpha * latent_B
        
        # ! МАГИЯ ЗДЕСЬ !
        # Мы разрешаем PyTorch считать производные для этого вектора
        z_mix = z_mix.detach().requires_grad_(True)
        
        # --- Б. ПРЯМОЙ ПРОХОД (FORWARD) ---
        # Прогоняем через Трансформер и Декодер
        z_transformed = model.transformer(z_mix)
        latent_reshaped = z_transformed.reshape(b, w, c, h).permute(0, 2, 3, 1)
        generated_spec = model.decoder_cnn(latent_reshaped)
        
        # --- В. ОБРАТНЫЙ ПРОХОД (BACKWARD) ---
        # Мы хотим узнать: какие части латента сильнее всего влияют на "громкость" картинки?
        # Или можно использовать дисперсию (variance), чтобы искать контрастные места
        target = generated_spec.sum() # Простая сумма всех пикселей
        
        # Обнуляем старые градиенты (на всякий случай)
        model.zero_grad()
        if z_mix.grad is not None:
            z_mix.grad.zero_()
            
        # Запускаем волну назад
        target.backward()
        
        # --- Г. ВИЗУАЛИЗАЦИЯ ГРАДИЕНТОВ ---
        # z_mix.grad - это тензор размера [1, Time, 1024]
        grads = z_mix.grad.abs().squeeze().cpu().numpy()
        
        # Нормализуем для красоты (чтобы вспышки были яркими)
        # Отрезаем экстремальные выбросы для контраста
        grads = np.clip(grads, 0, np.percentile(grads, 99))
        grads = grads / (grads.max() + 1e-8)
        
        # Транспонируем для отрисовки [1024, Time]
        grads = grads.T 
        
        # --- Д. ОТРИСОВКА КАДРА ---
        # Сверху: Сгенерированная Спектрограмма (Звук)
        # Снизу: Карта Градиентов (Мозг сети)
        
        gen_img = generated_spec.detach().squeeze().cpu().numpy()
        
        plt.figure(figsize=(10, 10), facecolor='black')
        
        # Верхняя часть: Спектрограмма
        plt.subplot(2, 1, 1)
        plt.imshow(gen_img, aspect='auto', origin='lower', cmap='magma')
        plt.axis('off')
        plt.title("Generated Sound (Decoder Output)", color='white', fontsize=10)
        
        # Нижняя часть: ГРАДИЕНТЫ (Neural Activity)
        plt.subplot(2, 1, 2)
        # Используем cmap='inferno' или 'plasma' для "магического" вида
        plt.imshow(grads, aspect='auto', origin='lower', cmap='inferno') 
        plt.axis('off')
        plt.title("Neural Gradients (Sensitivity Map)", color='white', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"art_frames/frame_{i:04d}.png", facecolor='black')
        plt.close()
        
        # --- Е. СБОР АУДИО ---
        # Восстанавливаем звук только для текущего кадра
        # ВНИМАНИЕ: Это медленно. Для демо берем центральную часть
        current_audio = tensor_to_audio(generated_spec.detach())
        
        # Чтобы сделать плавный переход звука, мы берем небольшой кусочек из центра
        # (Это упрощенная логика, для идеального морфинга звука нужны кроссфейды)
        samples_per_frame = len(current_audio) // frames
        start_sample = 0 
        # Просто накапливаем весь кусок (будет наложение, но для IDM сойдет как текстура)
        # Для чистоты эксперимента лучше сохранить просто последний сгенерированный кадр целиком
        # Но давай сохраним морфинг:
        if i == 0:
            full_audio = current_audio
        else:
            # Простой кроссфейд не делаем, просто склеиваем (для эксперимента)
            # Лучше: просто сохраним аудио центрального кадра (50%) как "звук этого видео"
            # Или сгенерируем один длинный трек отдельно.
            pass

    # Генерируем финальный длинный аудио трек морфинга (правильно)
    print("🎹 Рендеринг итогового аудио...")
    final_audio_list = []
    # Генерируем аудио кусками по 10 кадров чтобы не забивать память, но здесь просто возьмем
    # аудио из середины (50%) как пример звучания, 
    # либо (лучше) сгенерируем 4 ключевых точки и склеим.
    
    # Для простоты: сохраним аудио ПОСЛЕДНЕГО кадра (track B) и СРЕДНЕГО (Morph)
    # Чтобы ты мог наложить их в видеоредакторе.
    
    mid_idx = frames // 2
    
    # Регенерируем средний кадр для звука
    z_mid = (0.5 * latent_A + 0.5 * latent_B)
    z_mid = model.transformer(z_mid)
    latent_reshaped = z_mid.reshape(b, w, c, h).permute(0, 2, 3, 1)
    spec_mid = model.decoder_cnn(latent_reshaped)
    audio_mid = tensor_to_audio(spec_mid.detach())
    
    sf.write('art_audio_mid.wav', audio_mid, 22050)
    print("Готово! Кадры в папке 'art_frames', звук 'art_audio_mid.wav'")

# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IDMAutoencoder().to(device)
    
    # УКАЖИ ПУТЬ К ВЕСАМ (MEL-SPECTROGRAM VERSION)
    weights_path = 'idm_generator_gan.pth' 
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Веса не найдены!")
        exit()

    dataset = IDMTensorDataset(tensor_folder='idm_mels/train', slice_len=3, hop_length=256)
    
    # Генерируем 100 кадров (примерно 3-4 секунды видео при 30fps)
    generate_av_art(model, dataset, device, frames=100)