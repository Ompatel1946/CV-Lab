import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read grayscale image
image = cv2.imread(r"EX2\image.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()


# Add Salt and Pepper Noise
def add_salt_pepper_noise(image, probability=0.35):

    noisy_image = image.copy()

    random_values = np.random.rand(*image.shape)

    noisy_image[random_values < probability / 2] = 0
    noisy_image[random_values > 1 - probability / 2] = 255

    return noisy_image


# Changed probability from 0.2 to 0.35
noise_level = 0.35

noisy_image = add_salt_pepper_noise(image, noise_level)

plt.imshow(noisy_image, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")
plt.show()