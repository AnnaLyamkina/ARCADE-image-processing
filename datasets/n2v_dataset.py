"""Noise2Void training dataset for coronary angiography images."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from algorithms.dose_reduction import reduce_dose
from datasets.arcade_dataset import ArcadeDataset
from metrics.snr_cnr import get_bg_mask
from training.n2v_masking import mask_single_patch

DOSE_LEVELS = [1.0, 0.5, 0.25, 0.1]
DOSE_PROBS  = [0.2, 0.3, 0.30, 0.2]
PATCH_SIZE = 128
VESSEL_BIAS = 0.7  # fraction of patches sampled from vessel regions


def _pick_coords(
    valid_vessel: np.ndarray,
    valid_bg: np.ndarray,
    fallback: np.ndarray,
) -> tuple[int, int]:
    """Pick a random (y, x) centre using vessel-biased sampling."""
    if len(valid_vessel) > 0 and random.random() < VESSEL_BIAS:
        coords = valid_vessel
    else:
        coords = valid_bg if len(valid_bg) > 0 else fallback
    return coords[np.random.randint(len(coords))]


def _sample_patch_clean(
    image: np.ndarray,
    valid_vessel: np.ndarray,
    valid_bg: np.ndarray,
    fallback: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """Sample a single patch from a clean image."""
    half = patch_size // 2
    y, x = _pick_coords(valid_vessel, valid_bg, fallback)
    return image[y - half:y + half, x - half:x + half]


def _sample_patch(
    noisy: np.ndarray,
    clean: np.ndarray,
    valid_vessel: np.ndarray,
    valid_bg: np.ndarray,
    fallback: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a patch at the same location from noisy and clean images."""
    half = patch_size // 2
    y, x = _pick_coords(valid_vessel, valid_bg, fallback)
    return noisy[y - half:y + half, x - half:x + half], clean[y - half:y + half, x - half:x + half]


def _precompute_coords(mask: np.ndarray, patch_size: int = PATCH_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute valid vessel, background, and fallback sampling coordinates."""
    half = patch_size // 2
    h, w = mask.shape
    valid = np.zeros_like(mask)
    valid[half:h - half, half:w - half] = 1
    bg_mask = get_bg_mask(mask)
    valid_vessel = np.argwhere((mask == 1) & (valid == 1))
    valid_bg = np.argwhere((bg_mask == 1) & (valid == 1))
    fallback = np.argwhere(valid == 1)
    return valid_vessel, valid_bg, fallback


def make_train_val_datasets(
    root_dir: str,
    patches_per_image: int = 16,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple["N2VDataset", "N2VValDataset"]:
    """
    Split training images into train and val at image level to avoid leakage.

    Parameters
    ----------
    root_dir : str
        Path to dataset root.
    patches_per_image : int
        Patches sampled per image per epoch (train only).
    val_fraction : float
        Fraction of images held out for validation.
    seed : int
        Random seed for reproducible split.

    Returns
    -------
    train_ds, val_ds
    """
    all_samples = ArcadeDataset(root_dir=root_dir, split="train")
    n = len(all_samples)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))

    train_indices = indices[n_val:].tolist()
    val_indices = indices[:n_val].tolist()

    train_ds = N2VDataset(all_samples, train_indices, patches_per_image)
    val_ds = N2VValDataset(all_samples, val_indices, patches_per_image)
    return train_ds, val_ds


class N2VDataset(Dataset):
    """
    Training split: returns clean patches only — dose reduction and masking
    happen per batch in the training loop.

    Sampling coordinates are cached per image on first access so
    _precompute_coords runs once per image rather than once per patch.

    Parameters
    ----------
    dataset : ArcadeDataset
        Full training image dataset.
    indices : list[int]
        Image-level indices assigned to this split.
    patches_per_image : int
        Number of patches sampled per image per epoch.
    """

    def __init__(self, dataset: ArcadeDataset, indices: list, patches_per_image: int = 16):
        self.dataset = dataset
        self.indices = indices
        self.patches_per_image = patches_per_image
        self.patch_size = PATCH_SIZE
        self._coords_cache: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.indices) * self.patches_per_image

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_pos = idx // self.patches_per_image
        img_idx = self.indices[img_pos]
        sample = self.dataset[img_idx]
        if img_idx not in self._coords_cache:
            self._coords_cache[img_idx] = _precompute_coords(sample.mask)
        valid_vessel, valid_bg, fallback = self._coords_cache[img_idx]

        patch = _sample_patch_clean(sample.image, valid_vessel, valid_bg, fallback)
        return torch.from_numpy(patch[np.newaxis]).float()


class N2VValDataset(Dataset):
    """
    Validation split: same lazy sampling as N2VDataset but with fixed per-item
    seeds so validation loss is comparable across epochs.

    Parameters
    ----------
    dataset : ArcadeDataset
        Full training image dataset.
    indices : list[int]
        Image-level indices assigned to this split.
    patches_per_image : int
        Number of patches per image per epoch.
    """

    def __init__(self, dataset: ArcadeDataset, indices: list, patches_per_image: int = 16):
        self.dataset = dataset
        self.indices = indices
        self.patches_per_image = patches_per_image
        self.patch_size = PATCH_SIZE

    def __len__(self) -> int:
        return len(self.indices) * self.patches_per_image

    def __getitem__(self, idx: int):
        # fix seed per item so validation is reproducible across epochs
        np.random.seed(idx)
        random.seed(idx)

        img_pos = idx // self.patches_per_image
        sample = self.dataset[self.indices[img_pos]]
        valid_vessel, valid_bg, fallback = _precompute_coords(sample.mask)

        f = random.choices(DOSE_LEVELS, weights=DOSE_PROBS, k=1)[0]
        noisy = reduce_dose(sample.image, f)
        patch_noisy, patch_clean = _sample_patch(noisy, sample.image, valid_vessel, valid_bg, fallback)
        masked, original, mask = mask_single_patch(patch_noisy)

        to_tensor = lambda arr: torch.from_numpy(arr[np.newaxis]).float()
        return to_tensor(masked), to_tensor(original), to_tensor(mask), to_tensor(patch_clean), f
