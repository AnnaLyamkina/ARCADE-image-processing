"""ARCADE dataset loader for coronary angiography images and vessel masks."""

import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# visual control showed strong frame artifacts
EXCLUDE_SAMPLES = {
    "val": ['58.png', '87.png', '111.png', '158.png', '189.png'],
    "train": [],
}

class Sample(NamedTuple):
    """A single sample from the dataset: image, binary vessel mask, and file path."""
    image: np.ndarray  # Grayscale float32 [0, 1]
    mask: np.ndarray   # Binary uint8, vessel=1, background=0
    path: Path         # Path to the original image file


class ArcadeDataset:
    """
    ARCADE coronary angiography dataset with COCO annotations.
    
    Loads grayscale PNG images and rasterizes polygon masks from COCO JSON annotations.
    Images are float32 [0, 1] with vessels appearing dark on bright background.
    Masks are binary uint8 with vessel pixels = 1, background = 0.
    
    Args:
        root_dir: Root directory containing data/syntax/{train,val}/{images,annotations}
        split: Either 'train' or 'val'
    """
    
    def __init__(self, root_dir: str, split: str = "val"):
        """Initialize dataset.
        
        Args:
            root_dir: Path to dataset root (e.g., 'data/')
            split: Dataset split, 'train' or 'val'
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Build paths
        self.images_dir = self.root_dir / "syntax" / split / "images"
        self.annotations_file = self.root_dir / "syntax" / split / "annotations" / f"{split}.json"
        
        # Load COCO annotations
        with open(self.annotations_file, "r") as f:
            self.coco = json.load(f)
        
        
        # Build image_id -> list of polygons mapping (from annotations)
        self.image_id_to_polygons = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_polygons:
                self.image_id_to_polygons[img_id] = []
            
            # Store polygon from segmentation
            segmentation = ann["segmentation"]
            if isinstance(segmentation, list) and len(segmentation) > 0:
                # segmentation is a list of polygons (outer + holes), take first
                polygon = segmentation[0]
                self.image_id_to_polygons[img_id].append(polygon)
        
        # Image IDs in order
        self.image_ids = sorted(
            (img for img in self.coco["images"]
            if img["file_name"] not in EXCLUDE_SAMPLES.get(self.split, [])),
            key=lambda x: x["id"],
        )
# Build image_id -> filename and filename -> idx mapping
        self.image_id_to_path = {}
        self.filename_to_idx = {}
        # Build mappings from filtered list only
        for idx, img in enumerate(self.image_ids):
            self.image_id_to_path[img["id"]] = self.images_dir / img["file_name"]
            self.filename_to_idx[img["file_name"]] = idx 
            
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Sample:
        """Get a single sample.
        
        Args:
            idx: Index into dataset
            
        Returns:
            Sample with image (float32 [0,1]), mask (uint8 {0,1}), and path
        """
        img_info = self.image_ids[idx]
        img_id = img_info["id"]
        img_path = self.image_id_to_path[img_id]
        
        # Load image as grayscale float32 [0, 1]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        # Convert to float32 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Create binary vessel mask
        height, width = img.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Rasterize all polygons for this image
        if img_id in self.image_id_to_polygons:
            polygons = self.image_id_to_polygons[img_id]
            for polygon in polygons:
                # Convert polygon coordinate list to OpenCV format: array of shape (N, 1, 2)
                # polygon is a flat list [x1, y1, x2, y2, ...]
                coords = np.array(polygon).reshape(-1, 2).astype(np.int32)
                coords = coords.reshape(-1, 1, 2)  # Convert to (N, 1, 2) format
                
                # Draw filled polygon (vessel = 1)
                cv2.fillPoly(mask, [coords], 1)
        
        return Sample(image=img, mask=mask, path=img_path)

    def get_by_filename(self, filename: str) -> Sample:
        if filename in EXCLUDE_SAMPLES.get(self.split, []):
            raise ValueError(f"{filename} is excluded from the dataset")
        return self[self.filename_to_idx[filename]]
