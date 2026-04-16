"""
Generate simulated low-dose images for all val samples and save as 16-bit PNG.

Output structure:
    data/noisy/{dose}/
        {stem}_dose{dose_pct}.png   e.g. 121_dose50.png

Run from project root:
    python scripts/generate_noisy_dataset.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from algorithms.dose_reduction import reduce_dose
from datasets.arcade_dataset import ArcadeDataset

DOSES = [0.5, 0.25, 0.1, 0.05]
N0 = 200

dataset = ArcadeDataset(root_dir="../data", split="val")
print(f"Dataset size: {len(dataset)}")

for f in DOSES:
    dose_pct = int(f * 100)
    out_dir = Path(f"../data/noisy/{f}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(dataset, desc=f"dose={f}"):
        noisy = reduce_dose(sample.image, f, N0=N0)
        img_16 = (noisy * 65535).astype(np.uint16)
        stem = sample.path.stem
        dose_str = str(f).replace(".", "i")  # 0.5 → 0i5, 0.25 → 0i25
        out_path = out_dir / f"{stem}_dose{dose_str}.png"
        cv2.imwrite(str(out_path), img_16)

print("Done.")
