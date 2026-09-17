import cv2
import numpy as np


def sobel_operator(gray):
    # Sobel kernels
    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    Gy = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    # Create output image
    rows, cols = gray.shape
    output = np.zeros((rows, cols), dtype=np.float32)

    # Apply kernels manually
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            # Take 3x3 region
            region = gray[i-1:i+2, j-1:j+2]

            # Calculate gradients
            x = np.sum(Gx * region)
            y = np.sum(Gy * region)

            # Gradient magnitude
            value = np.sqrt(x**2 + y**2)

            output[i, j] = value

    # Convert to 8-bit
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


# Read image
image = cv2.imread("D:\CV LAB\ex5\image.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use our manually created Sobel function
sobel = sobel_operator(gray)

# Resize for small output
small_original = cv2.resize(image, (500, 350))
small_sobel = cv2.resize(sobel, (500, 350))

# Display
cv2.imshow("Original Image", small_original)
cv2.imshow("Manual Sobel Edge Detection", small_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()