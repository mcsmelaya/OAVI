import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFont, ImageDraw


font_path = "times.ttf"
font_size = 52
output_dir = "persian_characters"
csv_file = "../Lab7/etalon_features.csv"
profiles_dir = "profiles"


persian_alphabet = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"


os.makedirs(output_dir, exist_ok=True)
os.makedirs(profiles_dir, exist_ok=True)


def trim_whitespace(image):
    """Обрезает белые поля вокруг символа"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    return image.crop((x, y, x + w, y + h))


def calculate_features(image_path):
    """Вычисление признаков для изображения"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)[1]
    h, w = img.shape

    if h == 0 or w == 0:
        return [0] * 16, [], []  # Возвращаем нули, если изображение пустое


    q1 = img[:h // 2, :w // 2]
    q2 = img[:h // 2, w // 2:]
    q3 = img[h // 2:, :w // 2]
    q4 = img[h // 2:, w // 2:]

    weights = [np.sum(q) for q in [q1, q2, q3, q4]]
    total_area = (h * w) / 4
    norm_weights = [w / total_area if total_area != 0 else 0 for w in weights]


    y_coords, x_coords = np.where(img > 0)
    cx = np.mean(x_coords) if len(x_coords) > 0 else 0
    cy = np.mean(y_coords) if len(y_coords) > 0 else 0
    norm_cx = cx / w if w != 0 else 0
    norm_cy = cy / h if h != 0 else 0


    I_x = np.sum((y_coords - cy) ** 2) if len(y_coords) > 0 else 0
    I_y = np.sum((x_coords - cx) ** 2) if len(x_coords) > 0 else 0
    norm_I_x = I_x / (h ** 2) if h != 0 else 0
    norm_I_y = I_y / (w ** 2) if w != 0 else 0


    profile_x = np.sum(img, axis=0)
    profile_y = np.sum(img, axis=1)

    return weights + norm_weights + [cx, cy, norm_cx, norm_cy, I_x, I_y, norm_I_x, norm_I_y], profile_x, profile_y


def save_profile(profile, filename):
    """Сохранение профиля в виде диаграммы"""
    plt.figure(figsize=(6, 3))
    plt.bar(range(len(profile)), profile, color='black')
    plt.xticks(range(0, len(profile), max(1, len(profile) // 10)))
    plt.savefig(filename)
    plt.close()



data = []
for char in persian_alphabet:
    img = Image.new("RGB", (150, 150), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text((20, 20), char, font=font, fill="black")
    img = trim_whitespace(img)
    file_path = os.path.join(output_dir, f"{char}.png")
    img.save(file_path)


    features, profile_x, profile_y = calculate_features(file_path)
    data.append([char] + features)


    save_profile(profile_x, os.path.join(profiles_dir, f"{char}_x.png"))
    save_profile(profile_y, os.path.join(profiles_dir, f"{char}_y.png"))


columns = ["Char", "W1", "W2", "W3", "W4", "NW1", "NW2", "NW3", "NW4", "Cx", "Cy", "NCx", "NCy", "Ix", "Iy", "NIx",
           "NIy"]
df = pd.DataFrame(data, columns=columns)
df.to_csv(csv_file, sep=';', index=False)

print("Генерация изображений и вычисление признаков завершены.")