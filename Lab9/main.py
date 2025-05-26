import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import wiener

# # === 1. ЗАПИСЬ ЗВУКА ===
#
# DURATION = 5  # секунд
# SR = 44100  # частота дискретизации
#
# print("🔴 Запись звука 5 секунд... Говорите или играйте на инструменте!")
# recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype='float32')
# sd.wait()
# wav.write("original.wav", SR, (recording * 32767).astype(np.int16))
# print("✅ Запись завершена и сохранена в original.wav")
#
#
#
# y, sr = librosa.load("original.wav", sr=None, mono=True)
# S = librosa.stft(y, window='hann')
# S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
#
# plt.figure(figsize=(10, 6))
# librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
# plt.colorbar(format="%+2.0f dB")
# plt.title("Спектрограмма до фильтрации")
# plt.savefig("spectrogram_original.png")
# plt.close()
#
#
#
# denoised = wiener(y)  # винер
# wav.write("denoised.wav", sr, (denoised * 32767).astype(np.int16))
#
# S_denoised = librosa.stft(denoised, window='hann')
# S_denoised_db = librosa.amplitude_to_db(np.abs(S_denoised), ref=np.max)
#
# plt.figure(figsize=(10, 6))
# librosa.display.specshow(S_denoised_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
# plt.colorbar(format="%+2.0f dB")
# plt.title("Спектрограмма после фильтрации Винера")
# plt.savefig("spectrogram_denoised.png")
# plt.close()
#
#
# S_power = np.abs(S) ** 2
# hop_length = 512
# frame_duration = 0.1  # секунд
# frames_per_segment = int(sr * frame_duration / hop_length)
#
# energy_per_frame = np.sum(S_power, axis=0)
# conv = np.convolve(energy_per_frame, np.ones(frames_per_segment), mode='valid')
# max_idx = np.argmax(conv)
# time_of_max_energy = max_idx * hop_length / sr

def plot_diff_spectrogram(orig_file, filtered_file):
    y1, sr = librosa.load(orig_file, sr=None)
    y2, _ = librosa.load(filtered_file, sr=sr)

    S1 = librosa.stft(y1, window='hann', n_fft=2048, hop_length=512)
    S2 = librosa.stft(y2, window='hann', n_fft=2048, hop_length=512)

    # Амплитуда -> дБ
    S1_db = librosa.amplitude_to_db(np.abs(S1), ref=np.max)
    S2_db = librosa.amplitude_to_db(np.abs(S2), ref=np.max)

    # Разность
    diff = S1_db - S2_db

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(diff, sr=sr, x_axis='time', y_axis='log', cmap='bwr', vmin=-30, vmax=30)
    plt.colorbar(label='Разность амплитуды (дБ)')
    plt.title("Разностная спектрограмма: Оригинал − Фильтрованный")
    plt.tight_layout()
    plt.savefig("diff_spectrogram.png")
    plt.show()

plot_diff_spectrogram("original.wav", "denoised.wav")

# print(f"⚡ Максимальная энергия во временном окне: {time_of_max_energy:.2f} сек")

# === ГОТОВО ===
print("📂 Сохранены файлы:")
print(" - original.wav")
print(" - denoised.wav")
print(" - spectrogram_original.png")
print(" - spectrogram_denoised.png")
