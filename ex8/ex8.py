
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate an image containing three lines
W, H = 640, 420
img = np.zeros((H, W), dtype=np.uint8)

line_params = [
    (0.6, 35),
    (-0.35, 310),
    (0.15, 120)
]

for slope, intercept in line_params:
    xa, xb = 0, W - 1
    ya = int(slope * xa + intercept)
    yb = int(slope * xb + intercept)

    cv2.line(img, (xa, ya), (xb, yb), 255, 2)

# 2. Find edges
edge_img = cv2.Canny(img, 40, 130)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Input Image")
ax[0].axis("off")

ax[1].imshow(edge_img, cmap="gray")
ax[1].set_title("Detected Edges")
ax[1].axis("off")

plt.tight_layout()
plt.show()

# 3. Hough parameter space (m, c)
py, px = np.where(edge_img > 0)
slopes = np.linspace(-1.5, 1.5, 400)

plt.figure(figsize=(8, 5))

if len(px) > 0:
    sample_ids = np.linspace(
        0, len(px) - 1, min(80, len(px)), dtype=int
    )

    for idx in sample_ids:
        intercepts = py[idx] - slopes * px[idx]
        plt.plot(slopes, intercepts, color="gray", alpha=0.2)

for j, (m, c) in enumerate(line_params, start=1):
    plt.scatter(m, c, s=60, label=f"Line {j}")

plt.title("Hough Space: Slope and Intercept")
plt.xlabel("Slope (m)")
plt.ylabel("Intercept (c)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 4. Hough parameter space (rho, theta)
angles = np.deg2rad(np.arange(0, 180, 2))

plt.figure(figsize=(8, 5))

if len(px) > 0:
    for idx in sample_ids:
        rho_vals = (
            px[idx] * np.cos(angles)
            + py[idx] * np.sin(angles)
        )

        plt.plot(
            np.rad2deg(angles),
            rho_vals,
            color="steelblue",
            alpha=0.18
        )

plt.title("Hough Space: Rho and Theta")
plt.xlabel("Theta (degrees)")
plt.ylabel("Rho (pixels)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Detect lines using OpenCV
found_lines = cv2.HoughLines(
    edge_img,
    rho=1,
    theta=np.pi / 180,
    threshold=90
)

print("Original line parameters (m, c):")
for number, (m, c) in enumerate(line_params, start=1):
    print(f"Line {number}: m={m}, c={c}")

print("\nDetected parameters (rho, theta):")

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if found_lines is not None:
    for number, item in enumerate(found_lines[:10], start=1):
        rho, angle = item[0]
        print(
            f"Line {number}: rho={rho:.1f}, "
            f"theta={np.rad2deg(angle):.1f} degrees"
        )

        a, b = np.cos(angle), np.sin(angle)
        x0, y0 = a * rho, b * rho

        pt1 = (
            int(x0 + 1000 * (-b)),
            int(y0 + 1000 * a)
        )
        pt2 = (
            int(x0 - 1000 * (-b)),
            int(y0 - 1000 * a)
        )

        cv2.line(output, pt1, pt2, (0, 255, 0), 1)
else:
    print("No lines found. Try lowering the threshold.")

# 6. Show the detected lines
plt.figure(figsize=(8, 5))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Hough Transform Line Detection")
plt.axis("off")
plt.tight_layout()
plt.show()

