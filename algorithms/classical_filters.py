import numpy as np
import cv2
import math

def apply_gaussian(
    image: np.ndarray,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Gaussian low-pass filter.

    Replaces each pixel with a weighted average of its neighbours,
    weights follow a Gaussian distribution. Larger sigma = stronger
    smoothing but more blurring of vessel edges.

    Parameters
    ----------
    image : np.ndarray
        Float32 grayscale image in [0, 1], shape (H, W).
    sigma : float
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray
        Smoothed image, float32 [0, 1], same shape as input.
    """

    k = 2 * math.ceil(3 * sigma) + 1
    kernel_size = k if k % 2 == 1 else k + 1

    img_u8 = (image * 255).clip(0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_u8, (kernel_size, kernel_size), sigma)
    return blurred.astype(np.float32) / 255.0


def apply_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Bilateral edge-preserving filter.

    Extends Gaussian smoothing with an intensity weight — pixels with
    similar intensity are averaged strongly, pixels across edges are
    averaged weakly. Preserves vessel edges better than Gaussian.

    Parameters
    ----------
    image : np.ndarray
        Float32 grayscale image in [0, 1], shape (H, W).
    d : int
        Diameter of pixel neighbourhood.
    sigma_color : float
        Intensity similarity weight — higher = smoother across edges.
    sigma_space : float
        Spatial weight — higher = wider neighbourhood.

    Returns
    -------
    np.ndarray
        Filtered image, float32 [0, 1], same shape as input.
    """
    img_u8 = (image * 255).clip(0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_u8, d, sigma_color, sigma_space)
    return filtered.astype(np.float32) / 255.0