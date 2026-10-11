"""
core/inference.py
=================
Production inference pipeline for ChromaGAN photo colorization.

Converts input photographs to the CIELAB color space, executes
the U-Net Generator on the luminance channel, synthesizes chrominance (a, b),
applies user-controlled artistic color grading (saturation, warmth),
and returns the colorized image, chromatic heatmap, and color metrics.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import torch

from core.models import UNetColorGenerator

INFERENCE_SIZE: int = 256


def compute_colorfulness(img_bgr: np.ndarray) -> float:
    """
    Computes image colorfulness based on the Hasler & Süsstrunk (2003) metric:
    M = sqrt(sigma_rg^2 + sigma_yb^2) + 0.3 * sqrt(mu_rg^2 + mu_yb^2)
    Higher score indicates rich, diverse chromatic spread.
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)

    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)

    return float(std_root + (0.3 * mean_root))


def generate_chroma_heatmap(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    """
    Computes the chromatic deviation magnitude sqrt((a-128)^2 + (b-128)^2)
    and maps it to an energetic Jet/Plasma heatmap visualization.
    """
    a_dev = a_u8.astype(np.float32) - 128.0
    b_dev = b_u8.astype(np.float32) - 128.0
    magnitude = np.sqrt(a_dev ** 2 + b_dev ** 2)
    norm = np.clip((magnitude / 100.0) * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return heatmap


def build_generator(device: torch.device) -> UNetColorGenerator:
    """Instantiates UNetColorGenerator and initializes weights."""
    model = UNetColorGenerator(in_channels=1, out_channels=2, base_filters=64)
    model.to(device)
    model.eval()
    return model


def run_colorization(
    image_bytes: bytes,
    saturation: float = 1.2,
    color_style: str = "vibrant",  # natural | vibrant | warm_vintage | cool_cinematic
    device: Optional[torch.device] = None,
    generator: Optional[UNetColorGenerator] = None,
) -> Dict:
    """
    End-to-end CIELAB cGAN photo colorization pipeline.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if generator is None:
        generator = build_generator(device)

    # Decode uploaded image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image. Ensure a valid PNG, JPEG, or BMP file.")

    orig_h, orig_w = img_bgr.shape[:2]

    # Convert to CIELAB space
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(img_lab)

    # Grayscale original preview image
    bw_original_bgr = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2BGR)

    # Resize L channel for neural network processing (256x256)
    l_resized = cv2.resize(l_channel, (INFERENCE_SIZE, INFERENCE_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize L channel from [0, 255] to [-1, 1]
    l_norm = (l_resized.astype(np.float32) / 127.5) - 1.0
    l_tensor = torch.from_numpy(l_norm).unsqueeze(0).unsqueeze(0).to(device)

    # Forward pass through U-Net Generator
    with torch.no_grad():
        ab_pred = generator(l_tensor)

    # Convert predicted ab tensor [-1, 1] -> numpy
    ab_pred_np = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    a_pred = ab_pred_np[:, :, 0]
    b_pred = ab_pred_np[:, :, 1]

    # Resize predicted ab back to original photograph dimensions
    a_full = cv2.resize(a_pred, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    b_full = cv2.resize(b_pred, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Apply artistic color style presets
    if color_style == "warm_vintage":
        # Boost warm yellow-red tones
        a_full = a_full * 1.1 + 0.05
        b_full = b_full * 1.25 + 0.12
    elif color_style == "cool_cinematic":
        # Shift toward teal & cyan shadows
        a_full = a_full * 0.95 - 0.05
        b_full = b_full * 0.85 - 0.10
    elif color_style == "vibrant":
        # Expand dynamic range of chrominance
        a_full = a_full * 1.35
        b_full = b_full * 1.35
    else:  # natural
        a_full = a_full * 1.05
        b_full = b_full * 1.05

    # Apply user saturation slider
    a_full = a_full * saturation
    b_full = b_full * saturation

    # Clip and map back to uint8 [0, 255] for OpenCV LAB (where 128 is neutral gray)
    a_u8 = np.clip((a_full * 127.0) + 128.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip((b_full * 127.0) + 128.0, 0, 255).astype(np.uint8)

    # Recombine (L, a, b) and convert to BGR
    colorized_lab = cv2.merge([l_channel, a_u8, b_u8])
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)

    # Generate colorfulness metric and chromatic heatmap
    colorfulness_score = compute_colorfulness(colorized_bgr)
    chroma_heatmap = generate_chroma_heatmap(a_u8, b_u8)

    return {
        "bw_original_bgr": bw_original_bgr,
        "colorized_bgr": colorized_bgr,
        "chroma_heatmap": chroma_heatmap,
        "colorfulness_score": round(colorfulness_score, 2),
        "resolution": f"{orig_w}x{orig_h}",
    }
