import os
import torch
import numpy as np
import librosa
import pywt
from joblib import Parallel, delayed
import multiprocessing

SOURCE_FOLDER = 'D:\\MyFiles\\MLLEarning\\ansi_gen\\genre_classif\\Dataset\\IDM'       
OUTPUT_FOLDER = 'tensors' 
SR = 22050
HOP_LENGTH = 128            


def process_one_file(file_info):
    root, file, scales, wavelet = file_info
    
    file_path = os.path.join(root, file)
    
    try:
        y, _ = librosa.load(file_path, sr=SR, duration=300) 
        
        coefs, _ = pywt.cwt(y, scales, wavelet, sampling_period=1/SR)
        
        coefs = coefs[:, ::HOP_LENGTH]
        
        power = np.abs(coefs)**2
        log_power = librosa.power_to_db(power, top_db=80)
        norm_spec = (log_power - log_power.min()) / (log_power.max() - log_power.min() + 1e-6)
        
        tensor = torch.tensor(norm_spec, dtype=torch.float32).unsqueeze(0)
        
        filename_without_ext = os.path.splitext(file)[0]
        save_name = filename_without_ext + ".pt"
        if 'valid' in root.lower() or 'val' in root.lower():
            save_dir = os.path.join(OUTPUT_FOLDER, 'valid')
        else:
            save_dir = os.path.join(OUTPUT_FOLDER, 'train')
            
        save_path = os.path.join(save_dir, save_name)
        torch.save(tensor, save_path)
        
        return f"✅ {file}"
    
    except Exception as e:
        return f"❌ {file}: {e}"

def main():
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'valid'), exist_ok=True)

    wavelet = 'cmor1.5-1.0'
    freqs = np.geomspace(20, SR/2, num=64) 
    scales = pywt.frequency2scale(wavelet, freqs / SR)

    tasks = []
    
    print("📂 Сканирую папку...")
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                tasks.append((root, file, scales, wavelet))

    print(f"Найдено треков: {len(tasks)}")
    print(f"Запускаю обработку на {multiprocessing.cpu_count()} ядрах...")

    results = Parallel(n_jobs=2, verbose=10)(
        delayed(process_one_file)(task) for task in tasks
    )

    print("\nГотово!")

if __name__ == '__main__':
    main()