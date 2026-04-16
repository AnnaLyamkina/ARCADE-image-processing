# Low-Dose X-ray Angiography: Impact of Denoising on Vessel Visibility and Segmentation

## Overview

Low-dose X-ray angiography reduces radiation exposure but significantly degrades image quality due to increased quantum noise. This project investigates whether classical image preprocessing filters and a self-supervised deep learning denoiser improve **coronary vessel visibility** and **downstream segmentation performance** under low-dose conditions.

The focus is not only on perceptual image quality, but on **task-based evaluation**:

> Do denoising methods actually improve vessel segmentation?

---

## Key Questions

* How does low-dose (Poisson) noise affect vessel visibility in angiography images?
* Do classical filters improve or degrade segmentation performance?
* Can a self-supervised CNN denoiser (Noise2Void) better preserve vessel structures?
* Is improved image quality correlated with better segmentation?

---

## Dataset

* **ARCADE coronary angiography dataset** (syntax task only)
* 1000 training images (used for self-supervised denoising, no masks needed)
* 200 validation images with vessel annotations (used for benchmarking)
!!! Background ROI may contain unlabelled small vessels (<1.5mm), as SYNTAX annotations only cover major coronary segments. Metrics are consistent across filters but absolute CNR values may be slightly underestimated.

**Data format:**

* Grayscale PNG images, normalized to [0, 1], vessels appear dark on bright background
* Vessel annotations provided as COCO polygons → rasterized to binary masks (uint8, vessel=1, background=0)

---

## Project Structure

```
.
├── data/                          # Dataset (unpacked from arcade.zip, git-ignored)
│   └── syntax/
│       ├── train/                 # 1000 images for N2V training
│       │   ├── images/
│       │   └── annotations/train.json
│       └── val/                   # 200 images with masks for benchmarking
│           ├── images/
│           └── annotations/val.json
├── datasets/
│   └── arcade_dataset.py          # ArcadeDataset class (loads images + masks)
├── algorithms/                    # One file per filter 
│   ├── gaussian.py
│   ├── bilateral.py
│   ├── frangi.py
│   └── n2v.py
├── metrics/                       # One file per metric group
│   ├── snr_cnr.py             # SNR, CNR
│   └── dice.py           # Frangi + Dice for task-based evaluation
│   
├── training/
│   └── train_n2v.py               # N2V training loop and N2VDataset
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Explore val dataset, compute baselines
│   ├── 02_low_dose.ipynb # Apply dose reduction, compute metrics
│   └── 03_filter_comparison.ipynb  # Apply filters to denoise, produce plots
├── results/                       # Generated outputs (git-ignored)
│   └── metrics_table.csv          # Benchmark results
├── .gitignore                     # Excludes data/, results/, etc.
├── .env                           # Local paths (ARCADE_ROOT=data/)
└── README.md                      # This file
```

## Methods

### Low-dose simulation

To simulate reduced radiation dose, Poisson noise is applied to the images:

* Intensity scaling → Poisson sampling → rescaling
* Noise levels: **50%, 25%, 10% dose**

This approximates photon-counting statistics in X-ray imaging.

---

### Denoising methods

We compare representative approaches from different paradigms:

**Classical filters**

* Gaussian filter (baseline smoothing)
* Bilateral filter (edge-preserving)
* CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Learning-based**

* Noise2Void (self-supervised U-Net, PyTorch)

---

### Segmentation pipeline

A simple, fixed downstream task is used:

* Frangi vesselness filter
* Thresholding
* Binary vessel mask

This allows controlled evaluation of how preprocessing affects segmentation.

---

## Evaluation

### Image quality metrics

* SNR (Signal-to-Noise Ratio)
* CNR (Contrast-to-Noise Ratio)

### Task-based metrics

* Dice coefficient (primary metric)
* Optional: precision / recall

### Efficiency

* Execution time per method

Results are saved to `results/metrics_table.csv` for analysis.


## Experimental Setup

For each noise level (50%, 25%, 10%, 5%):

1. Apply noise to clean images
2. Apply each denoising method
3. Run segmentation pipeline
4. Compute metrics

---

## Results

### Visual comparison

Example results for decreasing dose levels:

* Original image
* Noisy image
* Gaussian / Bilateral / Noise2Void outputs
* Segmentation overlays

*(Insert figure here)*

---

### Quantitative results

| Method     | Dose | SNR ↑ | CNR ↑ | SSIM ↑ | Dice ↑ | Time ↓ |
| ---------- | ---- | ----- | ----- | ------ | ------ | ------ |
| Noisy      | 10%  | ...   | ...   | ...    | ...    | ...    |
| Gaussian   |      |       |       |        |        |        |
| Bilateral  |      |       |       |        |        |        |
| Frangi     |      |       |       |        |        |        |
| Noise2Void |      |       |       |        |        |        |

---

## Key Findings

*(Fill this after experiments — this section is critical)*

Example structure:

* Classical filters improve SNR but may **reduce vessel contrast**, harming segmentation
* Bilateral filtering preserves edges better than Gaussian but struggles at very low dose
* Noise2Void improves both perceptual quality and Dice score at moderate noise levels


---

## Discussion

### Image quality vs task performance

Improved perceptual quality does not always translate to better segmentation performance. Task-based metrics are essential for evaluating clinical relevance.

### Failure cases

* Small vessels are often lost after aggressive smoothing
* Frangi filter sensitivity decreases with reduced contrast
* Learning-based methods may introduce artifacts under extreme noise

---

## Conclusion

This study shows that:

* Denoising must be evaluated **in the context of downstream tasks**, not only visual quality
* Classical filters provide limited benefit under strong noise conditions
* Self-supervised deep learning offers a promising trade-off between noise reduction and structure preservation

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* scikit-image
* piq

---


## Future Work

* Incorporate learned segmentation models (U-Net)
* Extend to temporal angiography sequences
* Use more realistic noise models (including detector effects)
* Evaluate clinical endpoints beyond Dice score

---

## Author

Dr. Anna Lyamkina
Physicist | Medical Imaging | Machine Learning
