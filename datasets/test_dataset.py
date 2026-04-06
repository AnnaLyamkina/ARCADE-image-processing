from datasets.arcade_dataset import ArcadeDataset
import numpy as np

# Load validation set
dataset = ArcadeDataset('data/', split='val')
print(f'Dataset size: {len(dataset)}')

# Get first sample
sample = dataset[0]
print(f'Image shape: {sample.image.shape}, dtype: {sample.image.dtype}')
print(f'Image range: [{sample.image.min():.3f}, {sample.image.max():.3f}]')
print(f'Mask shape: {sample.mask.shape}, dtype: {sample.mask.dtype}')
print(f'Mask values: {np.unique(sample.mask).tolist()}')
print(f'Vessel pixels: {(sample.mask == 1).sum()}, Background pixels: {(sample.mask == 0).sum()}')
print(f'Image path: {sample.path}')

# Test training set
print('\n--- Training set ---')
train_dataset = ArcadeDataset('data/', split='train')
print(f'Training dataset size: {len(train_dataset)}')
