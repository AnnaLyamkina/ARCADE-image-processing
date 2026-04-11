import numpy as np
import cv2

def get_bg_mask(
    mask: np.ndarray,
    margin: int = 20
    ):
    '''
    Get background mask excluding dilated vessels and frame artifact
    '''
    
    border_mask = np.zeros_like(mask)
    border_mask[margin:-margin, margin:-margin] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(mask, kernel, iterations=2)

    bg_mask = ((dilated == 0) & (border_mask == 1)).astype(np.uint8)

    return bg_mask

def compute_snr_cnr(
    image: np.ndarray,
    mask: np.ndarray,
    margin: int = 30,
) -> dict[str, float]:
    
    """
    Compute SNR and CNR from vessel and background ROIs.

    SNR = std(vessel) / std(background)
    CNR = |mean(vessel) - mean(background)| / std(background)

    Background is defined as non-vessel pixels outside the frame border (visible artifact).
    Vessel mask is dilated before background sampling to avoid edge leakage.

    Parameters
    ----------
    image : np.ndarray
        Float32 grayscale image in [0, 1], shape (H, W).
    mask : np.ndarray
        Binary uint8, vessel=1 background=0, shape (H, W).
    margin : int
        Border margin in pixels to exclude frame artifacts.

    Returns
    -------
    dict with keys 'snr' and 'cnr'. Returns nan if ROI too small.
    """

    # border_mask = np.zeros_like(mask)
    # border_mask[margin:-margin, margin:-margin] = 1

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # dilated = cv2.dilate(mask, kernel, iterations=2)
    bg_mask = get_bg_mask(mask, margin)

    vessel_pixels = image[mask == 1]
    #bg_pixels = image[(dilated == 0) & (border_mask == 1)]
    bg_pixels = image[bg_mask == 1]

    if len(vessel_pixels) < 10 or len(bg_pixels) < 10:
        return {"snr": float("nan"), "cnr": float("nan")}

    noise_std = float(np.std(bg_pixels))
    if noise_std < 1e-9:
        return {"snr": float("nan"), "cnr": float("nan")}

    snr = float(np.std(vessel_pixels)) / noise_std
    cnr = float(abs(np.mean(vessel_pixels) - np.mean(bg_pixels))) / noise_std

    return {"snr": snr, "cnr": cnr}