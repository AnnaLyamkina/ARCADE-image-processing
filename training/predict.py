"""Apply trained U-Net denoiser to a full image."""

from pathlib import Path

import numpy as np
import torch

from training.unet import UNet

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model_cache: dict[Path, UNet] = {}


def _load_model(checkpoint: Path) -> UNet:
    if checkpoint not in _model_cache:
        model = UNet(base_channels=16).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        model.eval()
        _model_cache[checkpoint] = model
    return _model_cache[checkpoint]


def apply_unet(image: np.ndarray, checkpoint: Path = CHECKPOINT_PATH) -> np.ndarray:
    """
    Denoise a full image using the trained U-Net.

    Parameters
    ----------
    image : np.ndarray
        Float32 grayscale image in [0, 1], shape (H, W).
    checkpoint : Path
        Path to model checkpoint. Defaults to best.pt.

    Returns
    -------
    np.ndarray
        Denoised image, float32 [0, 1], same shape as input.
    """
    model = _load_model(checkpoint)
    tensor = torch.from_numpy(image[np.newaxis, np.newaxis]).float().to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze().cpu().numpy()
