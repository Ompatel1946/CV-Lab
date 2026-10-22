#Ex7
#23BIT059
import sys
print("Python:", sys.executable)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

HEIGHT, WIDTH = 512, 512
image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

# Sky-like smooth region
sky = np.tile(np.linspace(170, 220, WIDTH), (256, 1))
sky += np.random.normal(0, 2, (256, WIDTH))
image[:256] = np.clip(sky, 0, 255).astype(np.uint8)

# Grass-like textured region
grass = np.random.normal(100, 35, (256, WIDTH))
image[256:] = np.clip(grass, 0, 255).astype(np.uint8)

print("Image shape:", image.shape)
print("Image data type:", image.dtype)

plt.figure(figsize=(7, 6))
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Grayscale Image")
plt.axis("off")
plt.show()

PATCH_SIZE = 21

grass_locations = [
    (280, 50),
    (320, 180),
    (400, 300),
    (455, 455)
]

sky_locations = [
    (30, 30),
    (80, 150),
    (120, 300),
    (180, 450)
]

grass_patches = [
    image[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
    for r, c in grass_locations
]

sky_patches = [
    image[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
    for r, c in sky_locations
]

print("Grass patches:", len(grass_patches))
print("Sky patches:", len(sky_patches))
print("Patch size:", grass_patches[0].shape)

def calculate_glcm(patch, distance=5, angle=0, levels=32):
    """
    angle=0: horizontal pixel pairs.
    distance: separation between paired pixels.
    levels: grayscale quantization levels.
    """
    quantized = (
        patch.astype(np.int32) * levels // 256
    ).clip(0, levels - 1)

    if angle == 0:
        first = quantized[:, :-distance]
        second = quantized[:, distance:]
    elif angle == np.pi / 2:
        first = quantized[:-distance, :]
        second = quantized[distance:, :]
    else:
        raise ValueError("Supported angles: 0 and pi/2")

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (first.ravel(), second.ravel()), 1)
    np.add.at(glcm, (second.ravel(), first.ravel()), 1)

    total = glcm.sum()
    if total > 0:
        glcm /= total

    return glcm

def extract_features(glcm):
    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))

    difference = i - j
    contrast = np.sum(glcm * difference**2)
    dissimilarity = np.sum(glcm * np.abs(difference))
    homogeneity = np.sum(glcm / (1 + difference**2))
    asm = np.sum(glcm**2)
    energy = np.sqrt(asm)

    pi = glcm.sum(axis=1)
    pj = glcm.sum(axis=0)

    mi = np.sum(np.arange(levels) * pi)
    mj = np.sum(np.arange(levels) * pj)

    si = np.sqrt(np.sum((np.arange(levels) - mi)**2 * pi))
    sj = np.sqrt(np.sum((np.arange(levels) - mj)**2 * pj))

    if si > 0 and sj > 0:
        correlation = np.sum(
            glcm * (i - mi) * (j - mj)
        ) / (si * sj)
    else:
        correlation = 1.0

    return {
        "Dissimilarity": dissimilarity,
        "Correlation": correlation,
        "Contrast": contrast,
        "Homogeneity": homogeneity,
        "ASM": asm,
        "Energy": energy
    }

grass_features = [
    extract_features(calculate_glcm(p))
    for p in grass_patches
]

sky_features = [
    extract_features(calculate_glcm(p))
    for p in sky_patches
]

print("\nGLCM Texture Features")
print("-" * 60)

for label, feature_list in [
    ("Grass", grass_features),
    ("Sky", sky_features)
]:
    for number, features in enumerate(feature_list, 1):
        print(f"\n{label} {number}")
        for name, value in features.items():
            print(f"  {name}: {value:.4f}")

fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(image, cmap="gray", vmin=0, vmax=255)

for row, col in grass_locations:
    ax.add_patch(Rectangle(
        (col, row), PATCH_SIZE, PATCH_SIZE,
        edgecolor="red", facecolor="none", linewidth=2
    ))

for row, col in sky_locations:
    ax.add_patch(Rectangle(
        (col, row), PATCH_SIZE, PATCH_SIZE,
        edgecolor="blue", facecolor="none", linewidth=2
    ))

ax.set_title("Grass Patches (Red) and Sky Patches (Blue)")
ax.axis("off")
plt.show()

grass_dissimilarity = [
    f["Dissimilarity"] for f in grass_features
]
grass_correlation = [
    f["Correlation"] for f in grass_features
]
sky_dissimilarity = [
    f["Dissimilarity"] for f in sky_features
]
sky_correlation = [
    f["Correlation"] for f in sky_features
]

plt.figure(figsize=(8, 6))
plt.scatter(
    grass_dissimilarity, grass_correlation,
    label="Grass", s=150
)
plt.scatter(
    sky_dissimilarity, sky_correlation,
    label="Sky", s=150
)
plt.xlabel("Dissimilarity")
plt.ylabel("Correlation")
plt.title("GLCM: Dissimilarity vs Correlation")
plt.legend()
plt.grid(True)
plt.show()

grass_contrast = [f["Contrast"] for f in grass_features]
sky_contrast = [f["Contrast"] for f in sky_features]

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    grass_dissimilarity, grass_correlation, grass_contrast,
    label="Grass", s=150
)
ax.scatter(
    sky_dissimilarity, sky_correlation, sky_contrast,
    label="Sky", s=150
)

ax.set_xlabel("Dissimilarity")
ax.set_ylabel("Correlation")
ax.set_zlabel("Contrast")
ax.set_title("3D GLCM Texture Feature Space")
ax.legend()
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i, patch in enumerate(grass_patches):
    axes[0, i].imshow(patch, cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(f"Grass {i + 1}")
    axes[0, i].axis("off")

for i, patch in enumerate(sky_patches):
    axes[1, i].imshow(patch, cmap="gray", vmin=0, vmax=255)
    axes[1, i].set_title(f"Sky {i + 1}")
    axes[1, i].axis("off")

plt.suptitle("Texture Patches Used for GLCM Analysis")
plt.tight_layout()
plt.show()
