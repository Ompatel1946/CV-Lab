# OM PATEL, 23BIT059

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread(r"EX3\image.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found.")
    exit()

# Display Original Image
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# Add Salt and Pepper Noise

def add_salt_pepper_noise(image, probability=0.6):

    noisy = image.copy()

    random_matrix = np.random.rand(*image.shape)

    noisy[random_matrix < probability / 2] = 0
    noisy[random_matrix > 1 - probability / 2] = 255

    return noisy

# Add stronger noise
noisy_image = add_salt_pepper_noise(
    image,
    probability=0.6
)

plt.imshow(noisy_image, cmap="gray")
plt.title("Salt and Pepper Noisy Image")
plt.axis("off")
plt.show()

# Median Filter

def median_filter(image, kernel_size=3, stride=1):

    h, w = image.shape

    pad = kernel_size // 2

    padded = np.pad(
        image,
        pad,
        mode="reflect"
    )

    out_h = (
        h - kernel_size + 2 * pad
    ) // stride + 1

    out_w = (
        w - kernel_size + 2 * pad
    ) // stride + 1

    output = np.zeros(
        (out_h, out_w),
        dtype=image.dtype
    )

    for i in range(out_h):

        for j in range(out_w):

            window = padded[
                i * stride:i * stride + kernel_size,
                j * stride:j * stride + kernel_size
            ]

            output[i, j] = np.median(window)

    return output

# Mean Filter

def mean_filter(image, kernel_size=3, stride=1):

    h, w = image.shape

    pad = kernel_size // 2

    padded = np.pad(
        image,
        pad,
        mode="reflect"
    )

    out_h = (
        h - kernel_size + 2 * pad
    ) // stride + 1

    out_w = (
        w - kernel_size + 2 * pad
    ) // stride + 1

    output = np.zeros(
        (out_h, out_w),
        dtype=image.dtype
    )

    for i in range(out_h):

        for j in range(out_w):

            window = padded[
                i * stride:i * stride + kernel_size,
                j * stride:j * stride + kernel_size
            ]

            output[i, j] = np.mean(window)

    return output

# Compare Original, Noisy, Mean and Median

mean_result = mean_filter(
    noisy_image,
    kernel_size=7,
    stride=1
)

median_result = median_filter(
    noisy_image,
    kernel_size=7,
    stride=1
)


plt.figure(figsize=(16, 5))


plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")


plt.subplot(1, 4, 2)
plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")


plt.subplot(1, 4, 3)
plt.imshow(mean_result, cmap="gray")
plt.title("Mean Filter 7x7")
plt.axis("off")


plt.subplot(1, 4, 4)
plt.imshow(median_result, cmap="gray")
plt.title("Median Filter 7x7")
plt.axis("off")


plt.tight_layout()
plt.show()

# Different Median Filter Kernel Sizes

k3 = median_filter(
    noisy_image,
    kernel_size=3,
    stride=1
)

k7 = median_filter(
    noisy_image,
    kernel_size=7,
    stride=1
)

k11 = median_filter(
    noisy_image,
    kernel_size=11,
    stride=1
)


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.imshow(k3, cmap="gray")
plt.title("Kernel 3x3")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(k7, cmap="gray")
plt.title("Kernel 7x7")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(k11, cmap="gray")
plt.title("Kernel 11x11")
plt.axis("off")


plt.tight_layout()
plt.show()

# Different Stride Values

s1 = median_filter(
    noisy_image,
    kernel_size=5,
    stride=1
)

s2 = median_filter(
    noisy_image,
    kernel_size=5,
    stride=2
)

s4 = median_filter(
    noisy_image,
    kernel_size=5,
    stride=4
)


print("Stride 1 Shape:", s1.shape)
print("Stride 2 Shape:", s2.shape)
print("Stride 4 Shape:", s4.shape)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(s1, cmap="gray")
plt.title(f"Stride 1\nShape: {s1.shape}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(s2, cmap="gray")
plt.title(f"Stride 2\nShape: {s2.shape}")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(s4, cmap="gray")
plt.title(f"Stride 4\nShape: {s4.shape}")
plt.axis("off")

plt.tight_layout()
plt.show()