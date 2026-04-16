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
) -> tuple[float, float, float] | tuple[float, float, float, np.ndarray]:
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
    return_map : bool
        If True, return a binary mask for visualization
    Returns
    -------
    tuple of (dice, precision, recall).
    If return_map=True, returns (dice, precision, recall, pred).
    """
    pred = binarise_vesselness(vesselness, threshold, dilate_radius)

    tp = np.sum((pred == 1) & (mask == 1))
    fp = np.sum((pred == 1) & (mask == 0))
    fn = np.sum((pred == 0) & (mask == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2*tp / (2*tp + fp + fn)
    
    if return_map:
        return precision, recall, dice, pred
    else:
        return precision, recall, dice