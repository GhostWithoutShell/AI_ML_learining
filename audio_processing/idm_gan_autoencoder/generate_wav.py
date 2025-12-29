import torch
import numpy as np
import librosa
import soundfile as sf # pip install soundfile
import os
# Импортируй свои классы архитектуры (IDMAutoencoder, PositionalEncoding, etc.)
# Чтобы не копипастить, предположим, они в файле architecture.py
# Или скопируй их сюда полностью, как в прошлый раз.
from idm_gan_processing import IDMAutoencoder, IDMTensorDataset 

# --- НАСТРОЙКИ ---
SR = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64

def tensor_to_audio(tensor):
    """Превращает тензор [1, 64, Time] обратно в звук"""
    # 1. Тензор -> Numpy
    spec = tensor.squeeze().cpu().detach().numpy()
    
    # 2. Денормализация (из [0, 1] обратно в [-80, 0] dB)
    spec_db = (spec * 80) - 80
    
    # 3. dB -> Power
    spec_power = librosa.db_to_power(spec_db)
    
    # 4. Mel -> Linear STFT (Приближенно)
    # Griffin-Lim работает с линейным STFT, а не Mel.
    # Librosa умеет восстанавливать приближенно.
    stft_spec = librosa.feature.inverse.mel_to_stft(
        spec_power, sr=SR, n_fft=N_FFT, power=2.0
    )
    
    # 5. Griffin-Lim (Восстановление фазы)
    audio = librosa.griffinlim(stft_spec, n_iter=32, hop_length=HOP_LENGTH)
    
    return audio

def generate_morphing_track(model, dataset, device):
    print("🎧 Генерирую новый IDM трек...")
    model.eval()
    
    # Берем два случайных трека
    idx1 = np.random.randint(0, len(dataset))
    idx2 = np.random.randint(0, len(dataset))
    while idx1 == idx2: idx2 = np.random.randint(0, len(dataset))
    
    track_A = dataset[idx1].unsqueeze(0).to(device)
    track_B = dataset[idx2].unsqueeze(0).to(device)
    
    # === ПОЛУЧАЕМ ЛАТЕНТЫ ===
    with torch.no_grad():
        # Encoder A
        feat_A = model.encoder_cnn(track_A)
        b, c, h, w = feat_A.shape
        flat_A = model.pos_emb(feat_A.permute(0, 3, 1, 2).reshape(b, w, c*h))
        latent_A = model.transformer(flat_A)
        
        # Encoder B
        feat_B = model.encoder_cnn(track_B)
        flat_B = model.pos_emb(feat_B.permute(0, 3, 1, 2).reshape(b, w, c*h))
        latent_B = model.transformer(flat_B)
        
        # === МОРФИНГ (50%) ===
        # Смешиваем латентные пространства
        latent_mix = (latent_A + latent_B) / 2
        
        # Decoder
        latent_reshaped = latent_mix.reshape(b, w, c, h).permute(0, 2, 3, 1)
        reconstructed_mix = model.decoder_cnn(latent_reshaped)

    # === СОХРАНЕНИЕ ===
    # 1. Оригинал А
    audio_A = tensor_to_audio(track_A)
    sf.write('output_track_A.wav', audio_A, SR)
    
    # 2. Оригинал Б
    audio_B = tensor_to_audio(track_B)
    sf.write('output_track_B.wav', audio_B, SR)
    
    # 3. НАШ ГЕНЕРАТИВНЫЙ ТРЕК
    audio_mix = tensor_to_audio(reconstructed_mix)
    sf.write('generated_IDM_hybrid.wav', audio_mix, SR)
    
    print("Готово! Слушай файл 'generated_IDM_hybrid.wav' 🎹")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем модель
    model = IDMAutoencoder().to(device)
    
    # ВАЖНО: Тут должен быть файл весов, обученный на MEL-спектрограммах!
    weights_path = 'idm_generator_gan.pth' 
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Веса загружены.")
    else:
        print("Веса не найдены! Сначала обучи модель на данных idm_mels.")
        exit()

    # Датасет
    dataset = IDMTensorDataset(tensor_folder='idm_mels/train', slice_len=3, hop_length=256)
    
    generate_morphing_track(model, dataset, device)