"""
core/inference.py
=================
Production inference pipeline for RadSwin-WGAN Generator.

Handles:
  - Hounsfield windowing (Lung / Soft Tissue presets)
  - Min-max intensity normalization to [-1, 1]
  - Generator forward pass
  - Output rescaling and residual heatmap generation
  - Quantitative metrics: PSNR, SSIM
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from core.models import ResUNetGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "lung":        (-600.0, 1500.0),  # (center-ish, width) → stored as (min_hu, max_hu)
    "soft_tissue": (-160.0,  240.0),
    "default":     (-1000.0, 3000.0),
}

INFERENCE_RESOLUTION: int = 256   # spatial size fed to model

# ---------------------------------------------------------------------------
# Hounsfield / Intensity Utilities
# ---------------------------------------------------------------------------


def hounsfield_window(
    arr: np.ndarray,
    window_min: float,
    window_max: float,
) -> np.ndarray:
    """
    Clips a CT array to the HU window and rescales linearly to [0, 1].

    Args:
        arr        : float32 array of HU values, shape (H, W)
        window_min : lower HU bound
        window_max : upper HU bound

    Returns:
        float32 array clipped & rescaled to [0, 1]
    """
    arr = np.clip(arr, window_min, window_max).astype(np.float32)
    span = window_max - window_min
    return (arr - window_min) / (span + 1e-8)


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Min-max rescale array to [0, 1]; returns as float32.
    Falls back to all-zeros if std is zero.
    """
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def to_model_input(arr_01: np.ndarray, size: int = INFERENCE_RESOLUTION) -> torch.Tensor:
    """
    Convert a [0, 1] float32 HxW array to a model-ready tensor.
    Output: [1, 1, size, size] in [-1, 1].
    """
    # Resize
    resized = cv2.resize(arr_01, (size, size), interpolation=cv2.INTER_LINEAR)
    # [0,1] → [-1,1]
    normalized = resized * 2.0 - 1.0
    # Add batch + channel dims
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
    return tensor


def from_model_output(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert generator output tensor [1, 1, H, W] in [-1, 1] → uint8 [H, W].
    """
    arr = tensor.squeeze().cpu().detach().numpy()
    arr = (arr + 1.0) * 0.5          # [-1,1] → [0,1]
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Computes PSNR between two uint8 images.
    Returns float in dB; returns 100.0 if images are identical.
    """
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)
    mse = np.mean((o - r) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def compute_ssim_numpy(
    original: np.ndarray,
    reconstructed: np.ndarray,
    window_size: int = 11,
) -> float:
    """
    Computes mean SSIM between two uint8 images using a Gaussian window.
    Returns scalar in [0, 1].
    """
    o = original.astype(np.float64)
    r = reconstructed.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getGaussianKernel(window_size, 1.5)
    kernel_2d = kernel * kernel.T

    def _filter(img: np.ndarray) -> np.ndarray:
        return cv2.filter2D(img, -1, kernel_2d)

    mu1 = _filter(o)
    mu2 = _filter(r)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sigma1_sq = _filter(o * o) - mu1_sq
    sigma2_sq = _filter(r * r) - mu2_sq
    sigma12   = _filter(o * r) - mu12

    numerator   = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# Heatmap Generation
# ---------------------------------------------------------------------------


def make_residual_heatmap(
    original_u8: np.ndarray,
    denoised_u8: np.ndarray,
) -> np.ndarray:
    """
    Computes absolute residual difference and applies Jet colormap.

    Args:
        original_u8  : HxW uint8 input image
        denoised_u8  : HxW uint8 denoised image

    Returns:
        HxW×3 uint8 BGR heatmap suitable for JPEG encoding
    """
    diff = np.abs(original_u8.astype(np.int32) - denoised_u8.astype(np.int32))
    diff_norm = np.clip(diff, 0, 255).astype(np.uint8)
    # Apply equalization for better visibility
    diff_eq = cv2.equalizeHist(diff_norm)
    heatmap = cv2.applyColorMap(diff_eq, cv2.COLORMAP_JET)
    return heatmap


# ---------------------------------------------------------------------------
# Weight Initialization (random initialization for demo)
# ---------------------------------------------------------------------------


def build_generator(device: torch.device) -> ResUNetGenerator:
    """
    Instantiates the ResUNet generator and moves it to device.
    Weights are initialized via the model's _init_weights method.
    In production, call load_weights() after building.
    """
    gen = ResUNetGenerator(
        in_channels=1,
        out_channels=1,
        base_filters=64,
        n_res_blocks=6,
    )
    gen.to(device)
    gen.eval()
    return gen


def load_weights(
    generator: ResUNetGenerator,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """
    Loads generator weights from a checkpoint file.

    The checkpoint may be a bare state_dict or a dict with a 'generator'
    key produced by the training loop.

    Args:
        generator       : instantiated ResUNetGenerator
        checkpoint_path : path to .pth file
        device          : torch device
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        state_dict = ckpt["generator"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()


# ---------------------------------------------------------------------------
# Main Inference Runner
# ---------------------------------------------------------------------------


def run_inference(
    image_bytes: bytes,
    window_preset: str = "lung",
    device: Optional[torch.device] = None,
    generator: Optional[ResUNetGenerator] = None,
) -> Dict:
    """
    End-to-end inference pipeline.

    Args:
        image_bytes    : raw bytes of a grayscale PNG/JPEG/BMP image
        window_preset  : HU windowing preset ("lung" | "soft_tissue" | "default")
        device         : torch device; auto-detects if None
        generator      : pre-built generator model; built fresh if None

    Returns:
        dict with keys:
            original_u8      : HxW uint8 — windowed input
            denoised_u8      : HxW uint8 — model output
            heatmap_bgr      : HxW×3 uint8 — residual heatmap (BGR)
            psnr             : float (dB)
            ssim             : float [0,1]
            reconstruction_fidelity : float percentage [0,100]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if generator is None:
        generator = build_generator(device)

    # --- Decode input ---
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img_bgr is None:
        raise ValueError("Failed to decode image bytes. Ensure valid PNG/JPEG/BMP data.")

    # Convert to grayscale float32
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif len(img_bgr.shape) == 2:
        img_gray = img_bgr.astype(np.float32)
    else:
        raise ValueError(f"Unexpected image shape: {img_bgr.shape}")

    # --- Windowing ---
    window_min, window_max = WINDOW_PRESETS.get(window_preset, WINDOW_PRESETS["default"])

    # Remap pixel [0,255] to approximate HU range if not DICOM float
    if img_gray.max() <= 255.0 and img_gray.min() >= 0.0:
        # Treat pixel values as-is and apply normalization
        arr_01 = minmax_normalize(img_gray)
    else:
        # Genuine HU values
        arr_01 = hounsfield_window(img_gray, window_min, window_max)

    # Keep original for metric computation (resized to model size)
    original_resized = cv2.resize(arr_01, (INFERENCE_RESOLUTION, INFERENCE_RESOLUTION),
                                  interpolation=cv2.INTER_LINEAR)
    original_u8 = (original_resized * 255.0).astype(np.uint8)

    # --- Model forward pass ---
    input_tensor = to_model_input(arr_01, size=INFERENCE_RESOLUTION).to(device)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    denoised_u8 = from_model_output(output_tensor)

    # --- Residual heatmap ---
    heatmap_bgr = make_residual_heatmap(original_u8, denoised_u8)

    # --- Metrics ---
    psnr = compute_psnr(original_u8, denoised_u8)
    ssim = compute_ssim_numpy(original_u8, denoised_u8)
    reconstruction_fidelity = max(0.0, min(100.0, ssim * 100.0))

    return {
        "original_u8":            original_u8,
        "denoised_u8":            denoised_u8,
        "heatmap_bgr":            heatmap_bgr,
        "psnr":                   psnr,
        "ssim":                   ssim,
        "reconstruction_fidelity": reconstruction_fidelity,
    }
