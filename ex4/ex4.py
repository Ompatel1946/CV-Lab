import numpy as np
import cv2
import matplotlib.pyplot as plt


class NoisyImageDenoiser:
    """Loads a grayscale image, adds Gaussian noise, and compares
    a custom Gaussian smoothing implementation against OpenCV's."""

    def __init__(self, image_path):
        self.original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        self.noisy = None

    def add_gaussian_noise(self, mean=0.0, std_dev=75.0, seed=None):
        rng = np.random.default_rng(seed)
        noise = rng.normal(mean, std_dev, self.original.shape)
        corrupted = self.original.astype(np.float32) + noise
        self.noisy = np.clip(corrupted, 0, 255).astype(np.uint8)
        return self.noisy

    @staticmethod
    def _build_separable_kernel(size, sigma):
        """1D Gaussian kernel; full 2D filter is applied as two passes."""
        half = size // 2
        coords = np.arange(-half, half + 1)
        kernel_1d = np.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d /= kernel_1d.sum()
        return kernel_1d

    def custom_gaussian_blur(self, kernel_size=15, sigma=5.0):
        """Manual Gaussian smoothing using separable convolution
        (blur rows, then blur columns) for a lighter compute cost
        than a full 2D kernel pass."""
        if self.noisy is None:
            raise ValueError("Call add_gaussian_noise() first.")

        half = kernel_size // 2
        kernel_1d = self._build_separable_kernel(kernel_size, sigma)

        src = self.noisy.astype(np.float32)
        h, w = src.shape

        # Horizontal pass
        padded_h = np.pad(src, ((0, 0), (half, half)), mode='edge')
        temp = np.zeros_like(src)
        for col in range(w):
            window = padded_h[:, col:col + kernel_size]
            temp[:, col] = window @ kernel_1d

        # Vertical pass
        padded_v = np.pad(temp, ((half, half), (0, 0)), mode='edge')
        result = np.zeros_like(src)
        for row in range(h):
            window = padded_v[row:row + kernel_size, :]
            result[row, :] = kernel_1d @ window

        return np.clip(result, 0, 255).astype(np.uint8)

    def opencv_gaussian_blur(self, kernel_size=15, sigma=5.0):
        if self.noisy is None:
            raise ValueError("Call add_gaussian_noise() first.")
        return cv2.GaussianBlur(self.noisy, (kernel_size, kernel_size), sigma)


def show_image(img, title=None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_side_by_side(img_left, title_left, img_right, title_right):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_left, cmap="gray")
    plt.title(title_left)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_right, cmap="gray")
    plt.title(title_right)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    denoiser = NoisyImageDenoiser("EX1/image.png")

    noisy_img = denoiser.add_gaussian_noise(mean=0, std_dev=75)
    show_image(noisy_img, title="Noisy Image")

    print("Original shape:", denoiser.original.shape)
    print("Noisy shape:", noisy_img.shape)

    cv2_result = denoiser.opencv_gaussian_blur(kernel_size=15, sigma=5)
    show_image(cv2_result, title="Gaussian Blur using OpenCV")

    custom_result = denoiser.custom_gaussian_blur(kernel_size=15, sigma=5)
    show_image(custom_result, title="Gaussian Filter - Our Implementation")

    show_side_by_side(
        cv2_result, "OpenCV Gaussian Blur",
        custom_result, "Our Gaussian Filter"
    )