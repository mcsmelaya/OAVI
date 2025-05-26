from pickletools import uint8

from PIL import Image
import numpy as np

print("Enter image name: ")
filename = input()

with Image.open(f'/Users/darina_samoylenko/Lab1/Lab2/src_img/{filename}.png') as img:
    img.load()


img = img.convert("RGB")


def semitone(src_img):
    """Переводит цветное изображение в оттенки серого"""
    w, h = src_img.size
    semitone = np.zeros((h, w, 3), dtype=np.uint8)
    pixels = np.array(src_img)

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[y, x][:3]
            Y = int(0.3 * r + 0.59 * g + 0.11 * b)
            semitone[y, x] = [Y, Y, Y]

    return Image.fromarray(semitone)

print("semiton...")
semitone_img = semitone(img)
semitone_img.save(f'/Users/darina_samoylenko/Lab1/Lab2/rslt_img/{filename}_gray.png')


def integral_image(img):
    """Вычисляет интегральное изображение"""
    h, w = img.shape
    integral_img = np.zeros((h, w), dtype=np.int32)

    integral_img[0, 0] = img[0, 0]

    for x in range(1, h):
        integral_img[x, 0] = img[x, 0] + integral_img[x - 1, 0]

    for y in range(1, w):
        integral_img[0, y] = img[0, y] + integral_img[0, y - 1]

    for x in range(1, h):
        for y in range(1, w):
            integral_img[x, y] = img[x, y] + integral_img[x - 1, y] + integral_img[x, y - 1] - integral_img[x - 1, y - 1]


    return integral_img


def get_region_sum(integral_image, x1, y1, x2, y2):
    """Вычисляет сумму пикселей в заданном окне"""
    sum_region = integral_image[y2, x2]
    if x1 > 0:
        sum_region -= integral_image[y2, x1 - 1]
    if y1 > 0:
        sum_region -= integral_image[y1 - 1, x2]
    if x1 > 0 and y1 > 0:
        sum_region += integral_image[y1 - 1, x1 - 1]

    return sum_region


def compute_wan_threshold(img, window_size=7, k=0.2, R=128):
    """Вычисляет порог по методу WAN и применяет бинаризацию."""
    img_array = np.array(img.convert("L"))
    h, w = img_array.shape

    half_w = window_size // 2

    integral_img = integral_image(img_array)
    integral_sq_img = integral_image(img_array ** 2)

    binary_img = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            x1, y1 = max(0, x - half_w), max(0, y - half_w)
            x2, y2 = min(w - 1, x + half_w), min(h - 1, y + half_w)

            area = (x2 - x1 + 1) * (y2 - y1 + 1)

            sum_I = get_region_sum(integral_img, x1, y1, x2, y2)
            mean_I = sum_I / area

            sum_I2 = get_region_sum(integral_sq_img, x1, y1, x2, y2)
            variance = (sum_I2 / area) - (mean_I ** 2)
            std_dev = np.sqrt(max(variance, 0))

            max_I = np.max(img_array[y1:y2 + 1, x1:x2 + 1])

            m_max = (max_I + mean_I) / 2
            T = m_max * (1 - k * (1 - std_dev / R))

            binary_img[y, x] = 255 if img_array[y, x] > T else 0

    return Image.fromarray(binary_img)

print("computing wan threshold...")
binn = compute_wan_threshold(semitone_img)
binn.save(f'/Users/darina_samoylenko/Lab1/Lab2/rslt_img/{filename}.png')
