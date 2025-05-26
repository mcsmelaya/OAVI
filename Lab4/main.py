import cv2
import numpy as np
import matplotlib.pyplot as plt


# Функция для загрузки и преобразования изображения в оттенки серого
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


# Функция для свёртки изображения вручную с заданной матрицей (ядром)
def apply_filter(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2  # Размер отступа для свёртки
    output = np.zeros((h, w), dtype=np.float64)

    # Дополняем границы
    padded = np.pad(image, pad, mode='edge')

    for i in range(h):
        for j in range(w):
            region = padded[i:i + kh, j:j + kw]  # Выделяем подматрицу
            output[i, j] = np.sum(region * kernel)  # Свёртка

    return output


# Функция нормализации матрицы в диапазон [0, 255]
def normalize_image(image):
    min_val, max_val = np.min(image), np.max(image)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)


# Функция для бинаризации
def binarize_image(image, threshold=50):
    return np.where(image >= threshold, 255, 0).astype(np.uint8)


# Морфологическое расширение вручную
def morphological_dilation(image, kernel_size=3):
    h, w = image.shape
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            if np.max(padded[i:i + kernel_size, j:j + kernel_size]) > 0:
                output[i, j] = 255

    return output


# Основная программа
def main(image_path):

    image, gray = load_image(image_path)

    # Определяем операторы Собеля вручную
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)

    # Вычисляем градиенты Gx и Gy
    Gx = apply_filter(gray, sobel_x)
    Gy = apply_filter(gray, sobel_y)

    # Вычисляем полную градиентную матрицу G
    G = np.abs(Gx) + np.abs(Gy)

    # Нормализация
    Gx_norm = normalize_image(np.abs(Gx))
    Gy_norm = normalize_image(np.abs(Gy))
    G_norm = normalize_image(G)

    # Бинаризация градиентной матрицы
    G_bin = binarize_image(G_norm, threshold=50)

    # Морфологическое расширение
    G_dilated = morphological_dilation(G_bin)

    # Вычисляем разность между серым изображением и расширенной матрицей
    black_gradient = cv2.absdiff(gray, G_dilated)

    # Отображение результатов
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Исходное изображение")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(gray, cmap="gray")
    axs[0, 1].set_title("Полутоновое изображение")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(Gx_norm, cmap="gray")
    axs[0, 2].set_title("Градиентная матрица Gx")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(Gy_norm, cmap="gray")
    axs[1, 0].set_title("Градиентная матрица Gy")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(G_bin, cmap="gray")
    axs[1, 1].set_title("Бинаризованная градиентная матрица G")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(black_gradient, cmap="gray")
    axs[1, 2].set_title("Разность с морфологически расширенным")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()



image_path = "/Users/darina_samoylenko/Lab1/Lab4/src_img/strawberry.png"
main(image_path)
