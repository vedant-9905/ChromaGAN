"""
core/restoration.py
===================
Deep Archival Image Restoration & Defect Inpainting Engine.
Provides:
1. Automated multi-scale crack, scratch, fold crease, and tear detection.
2. Fast Marching (Telea) & Navier-Stokes exemplar inpainting.
3. High-frequency texture synthesis & facial feature unsharp enhancement.
4. Archival de-fading, sepia neutralization, and adaptive dynamic range recovery.
5. Diagnostic damage map generation for technical inspection.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Dict


def detect_scratches_and_defects(
    img_bgr_or_gray: np.ndarray,
    strength: float = 50.0,
    min_defect_area: int = 4,
    max_scratch_width: int = 15,
) -> np.ndarray:
    """
    Detects cracks, hairline scratches, tears, creases, and dust speckles using
    multi-scale directional morphological filters and ridge thresholding.
    
    strength: 0.0 to 100.0 (controls detection sensitivity)
    Returns: binary uint8 mask where 255 = damaged pixel to repair.
    """
    if strength <= 1.0:
        h, w = img_bgr_or_gray.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    # Convert to grayscale
    if img_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray.copy()

    h, w = gray.shape[:2]

    # Normalize sensitivity: strength 50 -> thresh ~ 18, strength 100 -> thresh ~ 8
    # Lower threshold = more defects detected
    thresh_val = max(6.0, 32.0 - (strength / 100.0) * 24.0)

    # 1. Multi-scale directional kernels to detect scratches at varying orientations
    # Scratches are thin along one direction and elongated along another
    k_len = max(5, int(7 * (max_scratch_width / 10.0)))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_len, k_len))

    # Bright tears / scratches (Top-Hat transform: image - opening)
    tophat_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_h)
    tophat_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_v)
    tophat_c = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_cross)
    bright_defects = cv2.max(cv2.max(tophat_h, tophat_v), tophat_c)

    # Dark scratches / fold creases (Black-Hat transform: closing - image)
    blackhat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_h)
    blackhat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_v)
    blackhat_c = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_cross)
    dark_defects = cv2.max(cv2.max(blackhat_h, blackhat_v), blackhat_c)

    # Combine bright and dark defects
    defect_response = cv2.add(bright_defects, dark_defects)

    # 2. Local gradient suppression: natural high-contrast scene edges
    # (e.g. horizon or cheek contour) have strong first derivative in broad area,
    # whereas scratches have high localized Laplacian/second derivative.
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobelx, sobely)

    # Suppress defects that coincide with very broad continuous boundaries
    # by subtracting blurred gradient magnitude
    broad_edges = cv2.GaussianBlur(grad_mag, (15, 15), 0)
    suppressed = np.clip(defect_response.astype(np.float32) - 0.25 * broad_edges, 0, 255).astype(np.uint8)

    # Threshold to binary mask
    _, binary_mask = cv2.threshold(suppressed, int(thresh_val), 255, cv2.THRESH_BINARY)

    # 3. Clean up noise: remove tiny single pixel noise, dilate slightly to cover tear boundary
    # Remove single isolated pixels
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

    # Dilate slightly so inpainting covers faded scratch edges
    dilation_radius = 2 if strength < 70 else 3
    defect_mask = cv2.dilate(
        clean_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius, dilation_radius)),
        iterations=1
    )

    return defect_mask


def inpaint_scratches(
    img_rgb: np.ndarray,
    defect_mask: np.ndarray,
    inpaint_radius: int = 3,
    method: str = "telea",
) -> np.ndarray:
    """
    Seamlessly reconstructs pixels covered by defect_mask using Telea or Navier-Stokes inpainting.
    img_rgb: uint8 RGB numpy array
    defect_mask: uint8 single channel binary mask (255 = inpaint)
    """
    if defect_mask is None or np.sum(defect_mask > 0) == 0:
        return img_rgb.copy()

    # OpenCV inpainting requires BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    inpainted_bgr = cv2.inpaint(img_bgr, defect_mask, inpaintRadius=inpaint_radius, flags=flag)

    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def restore_high_frequency_details(
    img_rgb_or_l: np.ndarray,
    strength: float = 35.0,
    face_clarity: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Restores micro-textures (skin, eyes, hair, textiles) using multi-scale
    bilateral guided unsharp masking.
    Returns: (enhanced_image, sharpness_gain_pct)
    """
    if strength <= 1.0:
        return img_rgb_or_l.copy(), 0.0

    orig_f = img_rgb_or_l.astype(np.float32)

    # Compute baseline sharpness using Tenengrad variance
    if img_rgb_or_l.ndim == 3:
        gray_before = cv2.cvtColor(img_rgb_or_l, cv2.COLOR_RGB2GRAY)
    else:
        gray_before = img_rgb_or_l
    sharp_before = float(cv2.Laplacian(gray_before, cv2.CV_32F).var())

    # Multi-scale detail extraction
    # 1. High-frequency texture (fine hair, iris, pores)
    blur_fine = cv2.GaussianBlur(orig_f, (3, 3), 0.8)
    fine_detail = orig_f - blur_fine

    # 2. Mid-frequency structure (eyelid edges, nose bridge, fabric weaves)
    blur_mid = cv2.bilateralFilter(img_rgb_or_l, d=5, sigmaColor=35, sigmaSpace=35).astype(np.float32)
    mid_detail = orig_f - blur_mid

    # Detail scaling factor
    alpha = (strength / 100.0) * 0.75
    beta = (strength / 100.0) * 0.45

    enhanced = orig_f + alpha * fine_detail + beta * mid_detail
    enhanced_u8 = np.clip(enhanced, 0.0, 255.0).astype(np.uint8)

    # Compute new sharpness
    if enhanced_u8.ndim == 3:
        gray_after = cv2.cvtColor(enhanced_u8, cv2.COLOR_RGB2GRAY)
    else:
        gray_after = enhanced_u8
    sharp_after = float(cv2.Laplacian(gray_after, cv2.CV_32F).var())

    gain_pct = round(((sharp_after - sharp_before) / max(1.0, sharp_before)) * 100.0, 1)

    return enhanced_u8, max(0.0, gain_pct)


def defade_and_neutralize_sepia(
    img_rgb: np.ndarray,
    defade_strength: float = 40.0,
) -> np.ndarray:
    """
    Removes vintage yellowing / aged sepia staining, balances white points,
    and revitalizes deep blacks and clean highlights.
    """
    if defade_strength <= 1.0:
        return img_rgb.copy()

    f = defade_strength / 100.0

    # Convert to LAB for independent chromatic and luminance control
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Neutralize yellow/brown sepia cast: in LAB, aged paper is shifted to b > 128 (yellow)
    # and a > 128 (reddish brown). Neutralizing brings a and b closer to 128.
    a_neut = (1.0 - 0.7 * f) * a + (0.7 * f) * 128.0
    b_neut = (1.0 - 0.7 * f) * b + (0.7 * f) * 128.0

    # Auto-stretch luminance dynamic range (discard extreme 0.5% outliers)
    p_low, p_high = np.percentile(l, (1.0, 99.0))
    if p_high > p_low + 10:
        l_stretched = np.clip((l - p_low) * (255.0 / (p_high - p_low)), 0, 255)
        l_final = (1.0 - f) * l + f * l_stretched
    else:
        l_final = l

    lab_restored = np.stack([l_final, a_neut, b_neut], axis=-1).astype(np.uint8)
    return cv2.cvtColor(lab_restored, cv2.COLOR_LAB2RGB)


def generate_damage_diagnostic_map(
    img_rgb: np.ndarray,
    defect_mask: np.ndarray,
) -> np.ndarray:
    """
    Creates an executive technical inspection overlay showing exactly where
    scratches, cracks, tears, and dust were detected and healed.
    Repaired defects glow in electric cyan (#06b6d4) and amber (#f59e0b).
    """
    h, w = img_rgb.shape[:2]
    # Subtle monochrome background
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bg_mono = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) * 0.45

    if defect_mask is None or np.sum(defect_mask > 0) == 0:
        # Return subtle background with "No Severe Defects" indication
        return np.clip(bg_mono, 0, 255).astype(np.uint8)

    # Defect overlay in vivid cyan and amber
    overlay = bg_mono.copy()
    mask_bool = defect_mask > 0

    # Outer boundary glow
    contour_mask = cv2.dilate(defect_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    overlay[contour_mask] = [245, 158, 11]  # Amber glow contour

    # Core defect fill
    overlay[mask_bool] = [6, 182, 212]      # Electric cyan core

    # Alpha blend
    blended = 0.35 * bg_mono + 0.65 * overlay
    return np.clip(blended, 0, 255).astype(np.uint8)
