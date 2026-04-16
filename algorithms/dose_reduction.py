import numpy as np
import cv2
from pathlib import Path

def reduce_dose(
    image: np.ndarray,
    f: float,
    N0: int = 200,
) -> np.ndarray:
    """
    Simulate reduced-dose X-ray image via Poisson noise.

    X-ray noise is dominated by photon counting noise. Reducing dose by
    factor f scales the expected photon count by f, increasing noise by 1/sqrt(f).

    Parameters
    ----------
    image : np.ndarray
        Float32 grayscale image in [0, 1], treated as noise-free full-dose reference.
    f : float
        Dose fraction in (0, 1]. 0.5 = half dose, 0.25 = quarter dose, 0.1 = 10% dose.
    N0 : int
        Reference photon count at full dose. Controls absolute noise level.

    Returns
    -------
    np.ndarray
        Simulated low-dose image, float32, clipped to [0, 1].
    """
    if f == 1.0:
        return image.copy()

    counts = image.astype(np.float64) * N0 * f
    noisy_counts = np.random.poisson(counts)
    result = noisy_counts / (f * N0)
    return np.clip(result, 0, 1).astype(np.float32)

def load_noisy(filename, f):
    if f == 1.0:
        path = Path(f"../data/syntax/val/images/{filename}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img.astype(np.float32) / 255.0
    stem = Path(filename).stem
    dose_str = str(f).replace(".", "i")
    path = Path(f"../data/noisy/{f}/{stem}_dose{dose_str}.png")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Noisy image not found: {path}")
    return img.astype(np.float32) / 65535.0
