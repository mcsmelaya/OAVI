import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


input_bmp = "romantic_phrase.bmp"  # Входное изображение
segment_dir = "segments"  # Папка для сохранения вырезанных букв
os.makedirs(segment_dir, exist_ok=True)


def load_image(path):
    """Загрузка BMP и бинаризация"""
    img = Image.open(path).convert("L")  # в градации серого
    img = np.array(img)
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return bin_img

def get_profiles(binary_image):
    """Получение горизонтального и вертикального профиля"""
    profile_y = np.sum(binary_image, axis=1)
    profile_x = np.sum(binary_image, axis=0)
    return profile_x, profile_y

def draw_profiles(profile, axis, out_path):
    """Сохранение профиля в PNG"""
    plt.figure(figsize=(6, 3))
    plt.bar(range(len(profile)), profile, color='black')
    plt.title(f"{axis}-профиль")
    plt.xticks(range(0, len(profile), max(1, len(profile) // 10)))
    plt.savefig(out_path)
    plt.close()

def segment_characters(image, min_gap=2, min_height=5):
    """Сегментация символов по вертикальному профилю с прореживанием"""
    profile_x = np.sum(image, axis=0)
    height = image.shape[0]
    segments = []

    in_char = False
    start = 0
    for i, val in enumerate(profile_x):
        if val > 0 and not in_char:
            in_char = True
            start = i
        elif val == 0 and in_char:
            end = i
            if end - start > min_gap:
                char_img = image[:, start:end]
                y_top, y_bottom = get_vertical_bounds(char_img)
                if y_bottom - y_top > min_height:
                    segments.append((start, y_top, end - start, y_bottom - y_top))
            in_char = False


    if in_char:
        end = len(profile_x)
        char_img = image[:, start:end]
        y_top, y_bottom = get_vertical_bounds(char_img)
        if y_bottom - y_top > min_height:
            segments.append((start, y_top, end - start, y_bottom - y_top))

    return segments

def get_vertical_bounds(cropped):
    """Определение вертикальных границ символа"""
    proj = np.sum(cropped, axis=1)
    top = next((i for i, v in enumerate(proj) if v > 0), 0)
    bottom = next((i for i, v in reversed(list(enumerate(proj))) if v > 0), len(proj))
    return top, bottom

def draw_boxes(original, boxes):
    """Рисуем прямоугольники вокруг символов"""
    img_color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite("segmented_with_boxes.png", img_color)


binary = load_image(input_bmp)


profile_x, profile_y = get_profiles(binary)
draw_profiles(profile_x, 'X', 'line_profile_x.png')
draw_profiles(profile_y, 'Y', 'line_profile_y.png')


boxes = segment_characters(binary)


draw_boxes(binary, boxes)


for i, (x, y, w, h) in enumerate(boxes):
    symbol_img = binary[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(segment_dir, f"char_{i}.png"), symbol_img)
    sx, sy = get_profiles(symbol_img)
    draw_profiles(sx, 'X', os.path.join(segment_dir, f"char_{i}_x.png"))
    draw_profiles(sy, 'Y', os.path.join(segment_dir, f"char_{i}_y.png"))

print("Сегментация завершена и профили построены.")
