"""Noise2Void pixel masking for self-supervised denoising."""

import numpy as np
import torch

MASK_FRACTION = 0.02  # fraction of pixels masked per patch

NEIGHBOUR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0,  -1),           (0,  1),
    (1,  -1), (1,  0), (1,  1),
]


def mask_single_patch(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Noise2Void masking to a single patch.

    For each masked pixel, its value is replaced with a randomly selected
    value from its 8-connected neighbourhood.

    Parameters
    ----------
    patch : np.ndarray
        Noisy patch, shape (H, W), float32.

    Returns
    -------
    masked : np.ndarray
        Patch with masked pixels replaced by neighbour values, (H, W).
    original : np.ndarray
        Unmodified noisy patch, (H, W).
    mask : np.ndarray
        Binary mask, 1 at masked positions, (H, W), float32.
    """
    H, W = patch.shape
    original = patch.copy()
    masked = patch.copy()
    mask = np.zeros((H, W), dtype=np.float32)

    n_masked = max(1, int(H * W * MASK_FRACTION))
    ys = np.random.randint(1, H - 1, size=n_masked)
    xs = np.random.randint(1, W - 1, size=n_masked)

    for y, x in zip(ys, xs):
        dy, dx = NEIGHBOUR_OFFSETS[np.random.randint(len(NEIGHBOUR_OFFSETS))]
        masked[y, x] = masked[y + dy, x + dx]
        mask[y, x] = 1.0

    return masked, original, mask


def mask_batch(
    batch: torch.Tensor,
    perc_pix: float = 0.02,
    roi_size: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply N2V blind-spot masking to a batch of patches.

    Each masked pixel is replaced with a value sampled uniformly from its
    roi_size × roi_size neighbourhood (excluding the pixel itself).

    Parameters
    ----------
    batch : torch.Tensor
        Noisy patches, shape (B, 1, H, W), float32, CPU.
    perc_pix : float
        Fraction of pixels masked per patch.
    roi_size : int
        Side length of the square neighbourhood used for replacement sampling.

    Returns
    -------
    masked, original, mask : torch.Tensor, each (B, 1, H, W).
    """
    B, _, H, W = batch.shape
    half = roi_size // 2

    offsets = np.array(
        [(dy, dx) for dy in range(-half, half + 1)
                  for dx in range(-half, half + 1)
                  if not (dy == 0 and dx == 0)],
        dtype=np.int32,
    )

    original = batch.clone()
    masked = batch.clone()
    mask = torch.zeros_like(batch)

    n_masked = max(1, int(H * W * perc_pix))

    for b in range(B):
        patch = masked[b, 0].numpy().copy()
        ys = np.random.randint(half, H - half, size=n_masked)
        xs = np.random.randint(half, W - half, size=n_masked)
        oi = np.random.randint(len(offsets), size=n_masked)
        patch[ys, xs] = patch[ys + offsets[oi, 0], xs + offsets[oi, 1]]
        masked[b, 0] = torch.from_numpy(patch)
        mask[b, 0, ys, xs] = 1.0

    return masked, original, mask


def apply_n2v_mask(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Noise2Void masking to a batch of patches.

    Parameters
    ----------
    batch : torch.Tensor
        Noisy patches, shape (B, 1, H, W), float32.

    Returns
    -------
    masked_batch, original_batch, mask_batch : torch.Tensor, each (B, 1, H, W).
    """
    B = batch.shape[0]
    masked_list, original_list, mask_list = [], [], []

    for b in range(B):
        patch = batch[b, 0].numpy()
        masked, original, mask = mask_single_patch(patch)
        masked_list.append(masked)
        original_list.append(original)
        mask_list.append(mask)

    to_tensor = lambda arr: torch.from_numpy(np.stack(arr))[:, None]
    return to_tensor(masked_list), to_tensor(original_list), to_tensor(mask_list)
