from PIL import Image
import numpy as np
from numpy.ma.core import arccos

print("Enter image name: ")
filename = input()

with Image.open(f'/Users/darina_samoylenko/Lab1/src_img/{filename}.png') as img:
    img.load()

img = img.convert("RGB")


image = np.array(img)

print("splitting channels...")
r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]


r_colored = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=2)
g_colored = np.stack([np.zeros_like(g), g, np.zeros_like(g)], axis=2)
b_colored = np.stack([np.zeros_like(b), np.zeros_like(b), b], axis=2)


Image.fromarray(r_colored.astype(np.uint8)).save(f'/Users/darina_samoylenko/Lab1/new_img/red_channel_{filename}.png')
Image.fromarray(g_colored.astype(np.uint8)).save(f"/Users/darina_samoylenko/Lab1/new_img/green_channel_{filename}.png")
Image.fromarray(b_colored.astype(np.uint8)).save(f"/Users/darina_samoylenko/Lab1/new_img/blue_channel_{filename}.png")




def rgb_to_hsi(image):
    new_image = np.array(image) / 255
    r, g, b = new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2]

    i = (r + g + b) / 3

    min_rgb = np.minimum(np.minimum(r, g), b)
    s = 1 - (3 * min_rgb/ (r + g + b + 1e-8))

    numerator = 0.5 * (2 * r - g - b)
    denominator = np.sqrt(np.maximum((r - g) ** 2 + (r - b) * (g - b), 0)) + 1e-8
    theta = arccos(numerator / denominator)
    h = np.where(b <= g, theta, 2 * np.pi - theta)
    h = h / (2 * np.pi)

    h = np.clip(h, 0, 1)
    s = np.clip(s, 0, 1)
    i = np.clip(i, 0, 1)


    hsi_image = np.stack([h, s, i], axis=2) * 255
    hsi_image = Image.fromarray(hsi_image.astype('uint8'))
    hsi_image.save(f'/Users/darina_samoylenko/Lab1/new_img/rgb_to_hsi_{filename}.png')

    i = Image.fromarray((i * 255).astype('uint8'))
    i.save(f'/Users/darina_samoylenko/Lab1/new_img/intensity_{filename}.png')

print("converting from RGB to HSI...")
rgb_to_hsi(img)

def invert_intensity(src_img):
    new_image = np.array(src_img) / 255
    r, g, b = new_image[:, :, 0], new_image[:, :, 1], new_image[:, :, 2]
    i = (r + g + b) / 3
    i_inv = 1 - i

    r_new = r * (i_inv / (i + 1e-10))
    g_new = g * (i_inv / (i + 1e-10))
    b_new = b * (i_inv / (i + 1e-10))

    r_new = np.clip(r_new, 0, 1)
    g_new = np.clip(g_new, 0, 1)
    b_new = np.clip(b_new, 0, 1)

    inv_img = np.stack([r_new, g_new, b_new],  axis=2) * 255
    inv_img = Image.fromarray(inv_img.astype('uint8'))
    inv_img.save(f'/Users/darina_samoylenko/Lab1/new_img/inverted_intensity_{filename}.png')

print("inverting intensity...")
invert_intensity(img)


def bilinear_resize(src_image, m):
    src_width, src_height = src_image.size
    pixels = np.array(src_image)
    new_width, new_height = m * src_width, m * src_height

    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            gx = (x / new_width) * src_width
            gy = (y / new_height) * src_height

            x1, y1 = int(gx), int(gy)
            x2, y2 = min(x1 + 1, src_width -1), min(y1 + 1, src_height - 1)

            dx = gx - x1
            dy = gy - y1

            f11 = pixels[y1, x1]
            f21 = pixels[y1, x2]
            f12 = pixels[y2, x1]
            f22 = pixels[y2, x2]

            fx1 = (1 - dx) * f11 + dx * f21
            fx2 = (1 - dx) * f12 + dx * f22

            new_p = (1 - dy) * fx1 + dy * fx2

            new_image[y, x] = new_p

    return Image.fromarray(new_image)


print("Enter M: ")
M = int(input())
print("bilinear resizing...")

resized_img = bilinear_resize(img, M)
resized_img.save(f'/Users/darina_samoylenko/Lab1/new_img/resized_in_{M}_times_{filename}.png')

def mean_resize(src_img, n):
    pixels = np.array(src_img)
    src_height, src_width, C = pixels.shape

    new_width, new_height = src_width // n, src_height // n
    new_img = np.zeros((new_height, new_width, C), dtype=np.uint8)

    for y in range(new_height):
        for x in range(new_width):
            y_start, y_end = y * n, (y + 1) * n
            x_start, x_end = x * n, (x + 1) * n

            block = pixels[y_start:y_end, x_start:x_end]

            for c in range(C):
                new_img[y, x, c] = np.mean(block[:, :, c])

    return Image.fromarray(new_img)


print("Enter N: ")
N = int(input())
print("mean resizing...")
mean_img = mean_resize(img, 2)
mean_img.save(f'/Users/darina_samoylenko/Lab1/new_img/resized_in_{N}_times_{filename}.png')


def two_pass_resampling(src_img, m, n):
    resized_img_m = bilinear_resize(src_img, m)
    result_img = mean_resize(resized_img_m, n)
    result_img.save(f'/Users/darina_samoylenko/Lab1/new_img/two_pass_resampling_in_{m}{n}_times_{filename}.png')

print("2 pass resampling...")
two_pass_resampling(img, M, N)


def one_pass_resampling(src_img, K):
    w, h = src_img.size

    new_w = max(1, int(round(w * K)))
    new_h = max(1, int(round(h * K)))

    new_img = Image.new(img.mode, (new_w, new_h))
    old_pixels = img.load()
    new_pixels = new_img.load()

    for y_new in range(new_h):
        for x_new in range(new_w):
            old_x = int(x_new / K)
            old_y = int(y_new / K)
            new_pixels[x_new, y_new] = old_pixels[old_x, old_y]

    new_img.save(f'/Users/darina_samoylenko/Lab1/new_img/one_pass_resampling_in_{K}_times_{filename}.png')


print("Enter K: ")
K = float(input())
print("1 pass resampling...")
one_pass_resampling(img, K)