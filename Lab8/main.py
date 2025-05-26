import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def glrlm(img: np.array, direction: str = '0'):
    max_gray = 256
    max_run_length = max(img.shape)
    matrix = np.zeros((max_gray, max_run_length), dtype=int)

    if direction == '0':
        # Горизонтально
        for row in img:
            run_val = row[0]
            run_len = 1
            for val in row[1:]:
                if val == run_val:
                    run_len += 1
                else:
                    matrix[run_val, run_len - 1] += 1
                    run_val = val
                    run_len = 1
            matrix[run_val, run_len - 1] += 1

    elif direction == '90':
        # Вертикально
        for col in img.T:
            run_val = col[0]
            run_len = 1
            for val in col[1:]:
                if val == run_val:
                    run_len += 1
                else:
                    matrix[run_val, run_len - 1] += 1
                    run_val = val
                    run_len = 1
            matrix[run_val, run_len - 1] += 1

    elif direction == '45':
        # Диагональ ↗ (снизу вверх, слева направо)
        h, w = img.shape
        for offset in range(-h + 1, w):
            diag = np.diagonal(np.fliplr(img), offset=offset)
            if diag.size == 0:
                continue
            run_val = diag[0]
            run_len = 1
            for val in diag[1:]:
                if val == run_val:
                    run_len += 1
                else:
                    matrix[run_val, run_len - 1] += 1
                    run_val = val
                    run_len = 1
            matrix[run_val, run_len - 1] += 1

    elif direction == '135':
        # Диагональ ↙ (сверху вниз, слева направо)
        h, w = img.shape
        for offset in range(-h + 1, w):
            diag = np.diagonal(img, offset=offset)
            if diag.size == 0:
                continue
            run_val = diag[0]
            run_len = 1
            for val in diag[1:]:
                if val == run_val:
                    run_len += 1
                else:
                    matrix[run_val, run_len - 1] += 1
                    run_val = val
                    run_len = 1
            matrix[run_val, run_len - 1] += 1

    return matrix

def GLNU(matrix: np.array):
    return np.sum(np.square(np.sum(matrix, axis=1))) / np.sum(matrix)

def RLNU(matrix: np.array):
    return np.sum(np.square(np.sum(matrix, axis=0))) / np.sum(matrix)


# === Загрузка и подготовка изображения ===
img_path = '/Users/darina_samoylenko/Lab1/Lab8/tree/tree.jpg'  # Замени на свой файл
base_dir = os.path.dirname(img_path)

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
H, L, S = cv2.split(img_hls)

# Сохраняем исходное полутоновое изображение
cv2.imwrite(os.path.join(base_dir, 'lightness_original.jpg'), L)

# === Построение GLRLM до преобразования яркости ===
glrlm_before = glrlm(L, direction='0')
glnu_before = GLNU(glrlm_before)
rlnu_before = RLNU(glrlm_before)

# === Яркостное линейное преобразование ===
alpha = 1.2  # контраст
beta = -30   # яркость (в диапазоне -255..255)
L_adj = np.clip(alpha * L.astype(np.float32) + beta, 0, 255).astype(np.uint8)

# Сохраняем новое изображение
hls_new = cv2.merge([H, L_adj, S])
img_contrast = cv2.cvtColor(hls_new, cv2.COLOR_HLS2BGR)
cv2.imwrite(os.path.join(base_dir, 'lightness_adjusted.jpg'), L_adj)
cv2.imwrite(os.path.join(base_dir, 'contrast_image.jpg'), img_contrast)

# === GLRLM после преобразования ===
glrlm_after = glrlm(L_adj, direction='0')
glnu_after = GLNU(glrlm_after)
rlnu_after = RLNU(glrlm_after)

# === Визуализация GLRLM через matplotlib ===
def show_glrlm(matrix, filename):
    matrix_log = np.log1p(matrix.astype(np.float32))  # логарифмическое масштабирование
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix_log, cmap='hot', aspect='auto')
    plt.colorbar(label='log(1 + value)')
    plt.title(filename.replace('_', ' ').capitalize())
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, filename))
    plt.close()

show_glrlm(glrlm_before, 'glrlm_before_log.png')
show_glrlm(glrlm_after, 'glrlm_after_log.png')

# === Гистограммы яркости ===
def save_hist(img, name):
    plt.figure()
    plt.hist(img.ravel(), bins=256, color='gray')
    plt.title(f'Histogram: {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'hist_{name}.png'))
    plt.close()

save_hist(L, 'before')
save_hist(L_adj, 'after')

# === Сохранение признаков ===
with open(os.path.join(base_dir, 'features.txt'), 'w') as f:
    f.write(f'GLNU before: {glnu_before:.4f}\n')
    f.write(f'RLNU before: {rlnu_before:.4f}\n')
    f.write(f'GLNU after: {glnu_after:.4f}\n')
    f.write(f'RLNU after: {rlnu_after:.4f}\n')

print("✅ Готово! Изображения, гистограммы и признаки сохранены.")
