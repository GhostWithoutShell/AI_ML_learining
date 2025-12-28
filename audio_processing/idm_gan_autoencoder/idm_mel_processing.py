import os
import torch
import numpy as np
import librosa
from joblib import Parallel, delayed
import multiprocessing


SOURCE_FOLDER = 'D:\\MyFiles\\MLLEarning\\ansi_gen\\genre_classif\\Dataset\\IDM'       
OUTPUT_FOLDER = 'idm_mels'  
SR = 22050
N_MELS = 64                 
N_FFT = 1024
HOP_LENGTH = 256            

def process_one_file(file_info):
    root, file = file_info
    file_path = os.path.join(root, file)
    
    try:
        # 1. Загрузка
        y, _ = librosa.load(file_path, sr=SR)
        
        
        
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        
        
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        
        mel_db = np.clip(mel_db, -80, 0)
        norm_spec = (mel_db + 80) / 80
        
        
        tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0)
        
        # 6. Сохранение
        filename_without_ext = os.path.splitext(file)[0]
        save_name = filename_without_ext + ".pt"

        if 'valid' in root.lower() or 'val' in root.lower():
            save_dir = os.path.join(OUTPUT_FOLDER, 'valid')
        else:
            save_dir = os.path.join(OUTPUT_FOLDER, 'train')
            
        os.makedirs(save_dir, exist_ok=True)
        torch.save(tensor, os.path.join(save_dir, save_name))
        
        return f"✅ {file}"
    
    except Exception as e:
        return f"❌ {file}: {e}"

def main():
    tasks = []
    print("📂 Сканирую папку для Mel-спектрограмм...")
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                tasks.append((root, file))

    print(f"Запускаю обработку {len(tasks)} файлов...")
    
    Parallel(n_jobs=-1, verbose=5)(
        delayed(process_one_file)(task) for task in tasks
    )
    print("\nГотово! Датасет idm_mels создан.")

if __name__ == '__main__':
    main()