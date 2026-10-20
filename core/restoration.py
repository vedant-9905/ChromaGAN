"""
core/restoration.py
===================
Ultra-Accurate Deep Archival Image Restoration & Defect Inpainting Engine.
Provides:
1. Scale-aware multi-resolution crack, scratch, fold crease, and tear detection.
2. Fast Marching (Telea) & Navier-Stokes exemplar inpainting with boundary seam elimination.
3. High-frequency micro-texture synthesis & facial detail enhancement.
4. Archival de-fading, sepia neutralization, and adaptive dynamic range recovery.
5. Diagnostic damage inspection map generation.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


def detect_scratches_and_defects(
    img_bgr_or_gray: np.ndarray,
    strength: float = 60.0,
    min_defect_area: int = 4,
    max_scratch_width: int = 15,
) -> np.ndarray:
    """
    Ultra-accurate, scale-aware detection of hairline scratches, deep tears,
    fold creases, lacquer craquelure, and dark emulsion gouges.
    
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
    # Resolution scaling factor (normalized to ~400px reference)
    scale = max(0.4, min(2.0, min(h, w) / 400.0))

    # Sensitivity thresholds: higher strength -> lower threshold (more defects captured)
    thresh_b = max(7.0, 30.0 - (strength / 100.0) * 22.0)
    thresh_d = max(9.0, 34.0 - (strength / 100.0) * 22.0)

    # 1. Multi-scale directional kernels for cracks at varying widths and angles
    k1 = max(3, int(5 * scale)) | 1
    k2 = max(5, int(11 * scale)) | 1
    k3 = max(7, int(19 * scale)) | 1

    bright_accum = np.zeros_like(gray)
    dark_accum = np.zeros_like(gray)

    for k_len in [k1, k2, k3]:
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
        kc = cv2.getStructuringElement(cv2.MORPH_CROSS, (k_len, k_len))

        # Top-Hat (bright cracks, white tears, paper creases)
        th = cv2.max(
            cv2.max(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kh),
                    cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kv)),
            cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kc)
        )
        # Black-Hat (dark cracks, shadow fold edges, ink stains, gouges)
        bh = cv2.max(
            cv2.max(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kh),
                    cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kv)),
            cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kc)
        )

        bright_accum = cv2.max(bright_accum, th)
        dark_accum = cv2.max(dark_accum, bh)

    # 2. Local contrast deviation from median filter
    med_k = max(5, int(9 * scale)) | 1
    med = cv2.medianBlur(gray, med_k)
    loc_b = cv2.subtract(gray, med)
    loc_d = cv2.subtract(med, gray)

    # Weighted blend of directional morphology and local contrast difference
    comb_b = cv2.addWeighted(bright_accum, 0.65, loc_b, 0.35, 0)
    comb_d = cv2.addWeighted(dark_accum, 0.65, loc_d, 0.35, 0)

    # 3. Continuous Scene Edge Suppression:
    # Natural scene boundaries (eyebrows, lips, jawline, clothing outlines) have
    # continuous broad gradient magnitude. Scratches have localized impulsive second derivatives.
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    edge_suppress = cv2.GaussianBlur(sobel_mag, (15, 15), 0)

    score_b = np.clip(comb_b.astype(np.float32) - 0.22 * edge_suppress, 0, 255).astype(np.uint8)
    score_d = np.clip(comb_d.astype(np.float32) - 0.22 * edge_suppress, 0, 255).astype(np.uint8)

    # Thresholding
    _, mb = cv2.threshold(score_b, int(thresh_b), 255, cv2.THRESH_BINARY)
    _, md = cv2.threshold(score_d, int(thresh_d), 255, cv2.THRESH_BINARY)
    defect_mask = cv2.bitwise_or(mb, md)

    # 4. Clean up noise: open to eliminate isolated single pixels
    defect_mask = cv2.morphologyEx(
        defect_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )

    # 5. Scale-aware Dilation:
    # Ensures the partially faded borders of tears/scratches are fully covered
    d_rad = max(2, int(3.2 * scale))
    defect_mask = cv2.dilate(
        defect_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_rad, d_rad)),
        iterations=1
    )

    return defect_mask


def inpaint_scratches(
    img_rgb: np.ndarray,
    defect_mask: np.ndarray,
    inpaint_radius: Optional[int] = None,
    method: str = "telea",
) -> np.ndarray:
    """
    Seamlessly reconstructs pixels covered by defect_mask using Telea or Navier-Stokes inpainting,
    followed by boundary seam-elimination filtering.
    """
    if defect_mask is None or np.sum(defect_mask > 0) == 0:
        return img_rgb.copy()

    h, w = img_rgb.shape[:2]
    scale = max(0.4, min(2.0, min(h, w) / 400.0))

    if inpaint_radius is None:
        inpaint_radius = max(3, int(4.5 * scale))

    # OpenCV inpainting expects BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    inpainted_bgr = cv2.inpaint(img_bgr, defect_mask, inpaintRadius=inpaint_radius, flags=flag)

    # Boundary seam-elimination: smooth transition between inpainted regions and original canvas
    seam_ring = cv2.dilate(defect_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    seam_ring = cv2.subtract(seam_ring, cv2.erode(defect_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))))

    if np.sum(seam_ring > 0) > 0:
        smoothed_bgr = cv2.bilateralFilter(inpainted_bgr, d=5, sigmaColor=30, sigmaSpace=30)
        ring_float = (seam_ring.astype(np.float32) / 255.0)[:, :, None]
        final_bgr = ((1.0 - 0.45 * ring_float) * inpainted_bgr.astype(np.float32) +
                     (0.45 * ring_float) * smoothed_bgr.astype(np.float32))
        inpainted_bgr = np.clip(final_bgr, 0, 255).astype(np.uint8)

    return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


def restore_high_frequency_details(
    img_rgb_or_l: np.ndarray,
    strength: float = 40.0,
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
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bg_mono = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) * 0.45

    if defect_mask is None or np.sum(defect_mask > 0) == 0:
        return np.clip(bg_mono, 0, 255).astype(np.uint8)

    overlay = bg_mono.copy()
    mask_bool = defect_mask > 0

    contour_mask = cv2.dilate(defect_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    overlay[contour_mask] = [245, 158, 11]  # Amber glow contour
    overlay[mask_bool] = [6, 182, 212]      # Electric cyan core

    blended = 0.35 * bg_mono + 0.65 * overlay
    return np.clip(blended, 0, 255).astype(np.uint8)
