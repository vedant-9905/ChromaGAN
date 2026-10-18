"""
core/dataset.py
===============
High-performance PyTorch Dataset and DataLoader for CIELAB Photo Colorization.
Extracts Luminance (L) as model input and Chrominance (a, b) as training target.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}


class ColorizationDataset(Dataset):
    """
    PyTorch Dataset for Conditional GAN Photo Colorization in CIELAB space.
    
    Transforms RGB images into:
    - L channel: normalized to [-1, 1] as generator input tensor [1, H, W]
    - ab channels: normalized to [-1, 1] as ground truth target tensor [2, H, W]
    - rgb: normalized [0, 1] tensor [3, H, W] for validation and metric computation
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        image_size: int = 256,
        is_train: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.is_train = is_train

        # Collect image files recursively
        self.image_paths: List[Path] = [
            p for p in self.root_dir.rglob("*") if p.suffix in VALID_EXTENSIONS
        ]
        self.image_paths.sort()

        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]

        if not self.image_paths:
            print(f"[ColorizationDataset] Warning: No images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = str(self.image_paths[idx])
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img_bgr is None:
            # Fallback for corrupted image
            img_bgr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Convert to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize to fixed training resolution
        img_rgb = cv2.resize(
            img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )

        # Data augmentation: Random horizontal flip during training
        if self.is_train and np.random.rand() > 0.5:
            img_rgb = cv2.flip(img_rgb, 1)

        # Convert RGB to CIELAB using OpenCV SIMD C++
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(img_lab)

        # Normalize L: [0, 255] -> [-1, 1]
        l_norm = (l_chan.astype(np.float32) / 127.5) - 1.0
        l_tensor = torch.from_numpy(l_norm).unsqueeze(0).float()  # [1, H, W]

        # Normalize ab: [0, 255] -> [-1, 1] where 128 is neutral gray 0.0
        ab_stack = np.stack([a_chan, b_chan], axis=-1)
        ab_norm = (ab_stack.astype(np.float32) / 127.5) - 1.0
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()  # [2, H, W]

        # Normalized RGB [0, 1] tensor for PSNR/SSIM evaluation
        rgb_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).float()

        return l_tensor, ab_tensor, rgb_tensor


def get_colorization_dataloader(
    root_dir: Union[str, Path],
    batch_size: int = 16,
    image_size: int = 256,
    is_train: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Creates a configured DataLoader for training or evaluation."""
    dataset = ColorizationDataset(
        root_dir=root_dir,
        image_size=image_size,
        is_train=is_train,
        max_samples=max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train and len(dataset) > batch_size,
    )
    return dataloader
