import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV Version:", cv2.__version__)
print("NumPy Version :", np.__version__)

# Read Image
img = cv2.imread(r"EX1\image.png")

if img is None:
    print("Image not found!")
    exit()

print("Image loaded successfully.")
print("Shape :", img.shape)
print("Data Type :", img.dtype)

# Display Image (OpenCV)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display Image (Matplotlib)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(7,5))
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")
plt.show()

# Image Details
height, width, channels = img.shape

print("Height :", height)
print("Width :", width)
print("Channels :", channels)

# Access Pixel
print("Pixel Value at (150,150):")
print(img[150,150])

# Modify Pixel
modified = img.copy()
modified[150,150] = [255,0,0]

cv2.imshow("Modified Pixel", modified)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Region of Interest
roi = img[80:280,80:280]

cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Resize
resized = cv2.resize(img, (500,350))

cv2.imshow("Resized Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Flip
horizontal = cv2.flip(img,1)
vertical = cv2.flip(img,0)

cv2.imshow("Horizontal Flip", horizontal)
cv2.imshow("Vertical Flip", vertical)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotate
rotated = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Draw Rectangle
rectangle = img.copy()

cv2.rectangle(
    rectangle,
    (80,80),
    (350,280),
    (255,0,0),
    2
)

cv2.imshow("Rectangle", rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Add Text
text = img.copy()

cv2.putText(
    text,
    "OpenCV Demo",
    (40,40),
    cv2.FONT_HERSHEY_COMPLEX,
    0.9,
    (0,255,255),
    2
)

cv2.imshow("Image with Text", text)
cv2.waitKey(0)
cv2.destroyAllWindows()