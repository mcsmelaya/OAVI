import cv2
import numpy as np
import matplotlib.pyplot as plt


def custom_rank_filter(image, size=5, rank=19):
    pad = size // 2
    padded_image = np.pad(image, pad, mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + size, j:j + size].flatten()
            sorted_window = np.sort(window)
            filtered_image[i, j] = sorted_window[rank]

    return filtered_image


def compute_difference(original, filtered, grayscale=True):
    if grayscale:
        return np.abs(original.astype(np.int16) - filtered.astype(np.int16)).astype(np.uint8)
    else:
        return np.bitwise_xor(original, filtered)


def process_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    filtered_image = custom_rank_filter(image, size=5, rank=19)


    diff_image = compute_difference(image, filtered_image, grayscale=True)


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Исходное изображение')

    plt.subplot(1, 3, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Отфильтрованное изображение')

    plt.subplot(1, 3, 3)
    plt.imshow(diff_image, cmap='gray')
    plt.title('Разностное изображение')

    plt.show()


# Пример использования
image_paths = ["/Users/darina_samoylenko/Lab1/Lab3/src_img/sp1.png", "/Users/darina_samoylenko/Lab1/Lab3/src_img/sp2.png", "/Users/darina_samoylenko/Lab1/Lab3/src_img/sp3.png", "/Users/darina_samoylenko/Lab1/Lab3/src_img/sp4.png", "/Users/darina_samoylenko/Lab1/Lab3/src_img/sp5.png", "/Users/darina_samoylenko/Lab1/Lab3/src_img/sp6.png"]  # Укажите пути к изображениям
for img_path in image_paths:
    process_image(img_path)
