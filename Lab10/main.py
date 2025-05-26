import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import soundfile as sf


SR = 44100  # Частота дискретизации
DURATION = 8  # Длительность записи, сек

def record_and_save(label):
    print(f"🔴 Запись звука: '{label}'...")
    audio = sd.rec(int(SR * DURATION), samplerate=SR, channels=1)
    sd.wait()
    sf.write(f"{label}.wav", audio.flatten(), SR)
    print(f"✅ Сохранено в {label}.wav")


def plot_spectrogram(file, label):
    y, sr = librosa.load(file, sr=None)
    S = librosa.stft(y, window='hann', n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Спектрограмма: {label}")
    plt.savefig(f"{label}_spectrogram.png")
    plt.close()
    return y, sr, S

def find_freq_range(S, sr):
    S_mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr)
    mean_energy = np.mean(S_mag, axis=1)
    threshold = np.max(mean_energy) * 0.05
    idx = np.where(mean_energy > threshold)[0]
    f_min, f_max = freqs[idx[0]], freqs[idx[-1]]
    return f_min, f_max

def find_fundamental_with_overtones(S, sr):
    power_spectrum = np.mean(np.abs(S) ** 2, axis=1)
    freqs = librosa.fft_frequencies(sr=sr)
    peaks, _ = find_peaks(power_spectrum, height=np.max(power_spectrum)*0.05)
    overtone_counts = {}
    for f0 in freqs[peaks]:
        count = 0
        for n in range(2, 10):
            overtone = f0 * n
            if np.any(np.isclose(freqs[peaks], overtone, atol=20)):
                count += 1
        overtone_counts[f0] = count
    best = max(overtone_counts, key=overtone_counts.get)
    return best

def find_formants(S, sr, hop_length=512):
    freqs = librosa.fft_frequencies(sr=sr)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    S_power = np.abs(S)**2
    avg_power = np.mean(S_power, axis=1)
    peaks, _ = find_peaks(avg_power, distance=4)
    formant_freqs = freqs[peaks]
    sorted_formants = formant_freqs[np.argsort(avg_power[peaks])][-3:]
    return sorted(sorted_formants)


for label in ["A", "I", "dog_cat_or_tarzan"]:
    record_and_save(label)



for label in ["A", "I", "dog_cat_or_tarzan"]:
    y, sr, S = plot_spectrogram(f"{label}.wav", label)

    f_min, f_max = find_freq_range(S, sr)
    print(f"🔎 [{label}] Частотный диапазон: от {f_min:.1f} Гц до {f_max:.1f} Гц")

    fundamental = find_fundamental_with_overtones(S, sr)
    print(f"🎵 [{label}] Основной тембральный тон: {fundamental:.1f} Гц")

    formants = find_formants(S, sr)
    print(f"🎯 [{label}] Форманты: {', '.join(f'{f:.1f}' for f in formants)} Гц")
    print()
