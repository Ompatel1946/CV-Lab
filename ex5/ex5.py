#experiment-5

#23BIT059

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("D:\CV LAB\ex5\image.png", cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

def sobel_filter(img):
    img = img.astype(np.float64)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    h, w = img.shape
    padded = np.pad(img, 1, mode="constant")

    gx = np.zeros((h, w), dtype=np.float64)
    gy = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(region * kx)
            gy[i, j] = np.sum(region * ky)

    edge_image = np.sqrt(gx**2 + gy**2)
    edge_image = (edge_image / edge_image.max() * 255).astype(np.uint8)

    return gx, gy, edge_image

gx, gy, edge_image = sobel_filter(image)

plt.imshow(np.abs(gx), cmap="gray")
plt.title("Sobel X")
plt.axis("off")
plt.show()

plt.imshow(np.abs(gy), cmap="gray")
plt.title("Sobel Y")
plt.axis("off")
plt.show()

plt.imshow(edge_image, cmap="gray")
plt.title("Final Sobel Edge Image")
plt.axis("off")
plt.show()