import numpy as np
import json
from pathlib import Path
from skimage.filters import frangi as skimage_frangi

params_file = Path(__file__).parent / "frangi_dice_params.json"
saved_params = json.loads(params_file.read_text()) if params_file.exists() else {}
scale_range = saved_params.get("scale_range", [5.0, 10.0])
gamma = saved_params.get("gamma", 2.0)


def apply_frangi(
    image: np.ndarray,
    scale_range: list[float, float] = scale_range,
    scale_step: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = gamma,
    black_ridges: bool = True,
    margin: int = 30
) -> np.ndarray:
    """
    Frangi vesselness filter.

    Computes a vessel probability map from the eigenvalues of the local
    Hessian at multiple scales. Output is normalised to [0, 1] where
    high values indicate vessel-like structures.

    Parameters
    ----------
    image : np.ndarray
        Float32 grayscale image in [0, 1], shape (H, W).
    scale_range : tuple[float, float]
        Min and max scale (sigma) for multiscale detection.
    scale_step : float
        Step between scales.
    alpha : float
        Sensitivity to blob-like vs tube-like structures.
    beta : float
        Sensitivity to background noise.
    gamma : float
        Structureness threshold.
    black_ridges : bool
        True for dark vessels on bright background.
    margin: int
        Masks out the frame artifact
    Returns
    -------
    np.ndarray
        Float32 vesselness map in [0, 1], shape (H, W).
    """
    sigmas = np.arange(scale_range[0], scale_range[1] + 1e-6, scale_step)
    vesselness = skimage_frangi(
        image,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
    )
    # filter out the frame
    vesselness[:margin, :] = 0
    vesselness[-margin:, :] = 0
    vesselness[:, :margin] = 0
    vesselness[:, -margin:] = 0
    #v_min, v_max = vesselness.min(), vesselness.max()
    # clipping by 99.9 percentile to avoid skewed values
    v_max = np.percentile(vesselness, 99.9)
    v_min = vesselness.min()
    if v_max > v_min:
        vesselness = np.clip(vesselness, 0, v_max)
        vesselness = (vesselness - v_min) / (v_max - v_min)
    else:
        return np.zeros_like(vesselness)
    
    return vesselness.astype(np.float32)