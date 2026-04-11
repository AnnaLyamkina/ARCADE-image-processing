import numpy as np
import cv2

def binarise_vesselness(
    vesselness: np.ndarray,
    threshold: float = 0.02,
    dilate_radius: int = 1,
) -> np.ndarray:
    """Threshold and dilate vesselness map to binary mask."""
    pred = (vesselness > threshold).astype(np.uint8)
    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (dilate_radius*2+1, dilate_radius*2+1))
        pred = cv2.dilate(pred, kernel, iterations=2)
    return pred

def compute_dice(
    vesselness: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.15,
    dilate_radius: int = 1,
    return_map: bool = False
) -> float:
    """
    Compute Dice score between binarised vesselness map and ground truth mask.

    Parameters
    ----------
    vesselness : np.ndarray
        Float32 vesselness map in [0, 1], shape (H, W).
    mask : np.ndarray
        Binary uint8 ground truth mask, vessel=1, shape (H, W).
    threshold : float
        Threshold for binarising vesselness map
    dilate_radius: int
        Dilation of vesselnessmap to account for frangi centerline detection
    Returns
    -------
    float — Dice score in [0, 1], higher is better.
    """
    pred = binarise_vesselness(vesselness, threshold, dilate_radius)

    intersection = (pred & mask).sum()
    denominator = pred.sum() + mask.sum()

    if denominator == 0:
        return float("nan")
    
    dice = float(2 * intersection / denominator)
    
    if return_map:
        return dice, pred
    else:
        return dice