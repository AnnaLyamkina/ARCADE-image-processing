"""Noise2Void training loop."""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.dose_reduction import reduce_dose
from datasets.n2v_dataset import DOSE_LEVELS, DOSE_PROBS, make_train_val_datasets
from training.n2v_masking import mask_batch
from training.unet import UNet

# --- config ---
ROOT_DIR = "data"
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
VAL_FRACTION = 0.1
PATCHES_PER_IMAGE = 16
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
LOG_PATH = Path(__file__).parent.parent / "results" / "training_log.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_COLUMNS = [
    "epoch", "train_loss", "val_loss", "lr", "epoch_time_sec",
    "train_loss_std",
    "loss_dose_1.0", "loss_dose_0.5", "loss_dose_0.25", "loss_dose_0.1",
    "val_loss_dose_1.0", "val_loss_dose_0.5", "val_loss_dose_0.25", "val_loss_dose_0.1",
]


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss computed only at masked pixel positions."""
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def _log_epoch(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(epochs: int = EPOCHS, root_dir: str = ROOT_DIR, train_ds=None, val_ds=None, resume: bool = False):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if train_ds is None or val_ds is None:
        train_ds, val_ds = make_train_val_datasets(
            root_dir=root_dir,
            patches_per_image=PATCHES_PER_IMAGE,
            val_fraction=VAL_FRACTION,
        )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # model
    model = UNet(base_channels=16).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_checkpoint_path: Path | None = None
    epochs_no_improve = 0
    early_stopping_patience = 10
    start_epoch = 1

    if resume:
        last = CHECKPOINT_DIR / "last.pt"
        ckpt = torch.load(last, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_loss = ckpt["best_val_loss"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        start_epoch = ckpt["epoch"] + 1
        _bcp = ckpt.get("best_checkpoint_path")
        best_checkpoint_path = Path(_bcp) if _bcp is not None else None  # handles str or Path
        tqdm.write(f"Resumed from epoch {ckpt['epoch']}, best val loss {best_val_loss:.6f}")

    epoch_bar = tqdm(range(start_epoch, epochs + 1), desc="epochs", unit="ep")
    for epoch in epoch_bar:
        epoch_start = time.time()

        # train
        model.train()
        batch_losses: list[float] = []
        batch_doses: list[float] = []
        batch_bar = tqdm(train_loader, desc=f"train {epoch:03d}", unit="batch", leave=False)
        for clean_batch in batch_bar:
            f = random.choices(DOSE_LEVELS, weights=DOSE_PROBS, k=1)[0]
            clean_np = clean_batch[:, 0].numpy()
            noisy_np = reduce_dose(clean_np, f)
            noisy_batch = torch.from_numpy(noisy_np[:, None]).float()

            masked, original, mask = mask_batch(noisy_batch)
            masked = masked.to(DEVICE)
            original = original.to(DEVICE)
            mask = mask.to(DEVICE)

            pred = model(masked)
            loss = masked_mse(pred, original, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_doses.append(f)
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

        train_loss = float(np.mean(batch_losses))
        train_loss_std = float(np.std(batch_losses))
        dose_losses = {
            d: float(np.mean([l for l, fd in zip(batch_losses, batch_doses) if fd == d]) )
            if any(fd == d for fd in batch_doses) else float("nan")
            for d in DOSE_LEVELS
        }

        # validate
        model.eval()
        val_loss = 0.0
        val_dose_losses: dict[float, list[float]] = {d: [] for d in DOSE_LEVELS}
        with torch.no_grad():
            for masked, original, mask, _, doses in tqdm(val_loader, desc="val", unit="batch", leave=False):
                masked = masked.to(DEVICE)
                original = original.to(DEVICE)
                mask = mask.to(DEVICE)
                pred = model(masked)
                val_loss += masked_mse(pred, original, mask).item()
                for d in DOSE_LEVELS:
                    sel = (doses == d).nonzero(as_tuple=True)[0]
                    if len(sel) > 0:
                        val_dose_losses[d].append(
                            masked_mse(pred[sel], original[sel], mask[sel]).item()
                        )
        val_loss /= len(val_loader)
        val_dose_losses_mean = {
            d: float(np.mean(v)) if v else float("nan")
            for d, v in val_dose_losses.items()
        }

        epoch_time = time.time() - epoch_start
        scheduler.step(val_loss)

        epoch_bar.set_postfix(
            train=f"{train_loss:.6f}", val=f"{val_loss:.6f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        _log_epoch({
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            "epoch_time_sec": f"{epoch_time:.1f}",
            "train_loss_std": f"{train_loss_std:.6f}",
            "loss_dose_1.0": f"{dose_losses[1.0]:.6f}" if not np.isnan(dose_losses[1.0]) else "",
            "loss_dose_0.5": f"{dose_losses[0.5]:.6f}" if not np.isnan(dose_losses[0.5]) else "",
            "loss_dose_0.25": f"{dose_losses[0.25]:.6f}" if not np.isnan(dose_losses[0.25]) else "",
            "loss_dose_0.1": f"{dose_losses[0.1]:.6f}" if not np.isnan(dose_losses[0.1]) else "",
            "val_loss_dose_1.0": f"{val_dose_losses_mean[1.0]:.6f}" if not np.isnan(val_dose_losses_mean[1.0]) else "",
            "val_loss_dose_0.5": f"{val_dose_losses_mean[0.5]:.6f}" if not np.isnan(val_dose_losses_mean[0.5]) else "",
            "val_loss_dose_0.25": f"{val_dose_losses_mean[0.25]:.6f}" if not np.isnan(val_dose_losses_mean[0.25]) else "",
            "val_loss_dose_0.1": f"{val_dose_losses_mean[0.1]:.6f}" if not np.isnan(val_dose_losses_mean[0.1]) else "",
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                best_checkpoint_path.unlink()
            val_str = f"{val_loss:.6f}".replace(".", "i")
            best_checkpoint_path = CHECKPOINT_DIR / f"best_val{val_str}.pt"
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            epochs_no_improve += 1

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        }, CHECKPOINT_DIR / "last.pt")

        if epochs_no_improve >= early_stopping_patience:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(epochs=args.epochs, resume=args.resume)
