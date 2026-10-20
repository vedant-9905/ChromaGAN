"""
core/pipeline.py
================
ReviveGAN Studio Pro — Ultra-High Processing Archival Restoration Engine.
Combines:
1. Multi-scale Laplacian ridge & directional defect inpainting (Telea Fast Marching).
2. Natural human skin tone gamut calibration (anti-oversaturation, clear eyes).
3. SOTA Dual Neural Ensemble Colorization (SIGGRAPH17 + ECCV16).
4. Bilateral edge snapping for zero color bleed.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.colorizers import eccv16, siggraph17
from core.restoration import (
    detect_scratches_and_defects,
    inpaint_scratches,
    restore_high_frequency_details,
    defade_and_neutralize_sepia,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_sig = None
_model_ecc = None


def get_models() -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Lazy-loads and caches both pre-trained models on DEVICE."""
    global _model_sig, _model_ecc
    if _model_sig is None:
        _model_sig = siggraph17(pretrained=True).eval().to(DEVICE)
    if _model_ecc is None:
        _model_ecc = eccv16(pretrained=True).eval().to(DEVICE)
    return _model_sig, _model_ecc


class ReviveGANEngine:
    """Production Archival Photo Restoration & Defect Inpainting Engine."""

    def __init__(self):
        self.device = DEVICE
        self._cache = {}
        if self.device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)

    def restore_and_relive(
        self,
        input_image: np.ndarray,
        defect_repair_strength: float = 70.0,
        detail_enhancement: float = 35.0,
        defade_strength: float = 30.0,
        ensemble_blend: float = 0.75,
        resolution: int = 256,
    ) -> Dict:
        """
        Executes ultra-accurate defect inpainting, natural skin colorization,
        and high-frequency detail restoration.
        """
        if input_image is None:
            return {}

        start_time = time.time()

        if input_image.ndim == 2:
            img_rgb = np.tile(input_image[:, :, None], 3)
        else:
            img_rgb = input_image[:, :, :3]

        orig_h, orig_w = img_rgb.shape[:2]
        scale = max(0.4, min(2.0, min(orig_h, orig_w) / 400.0))

        # Check Cache
        img_bytes = img_rgb.tobytes()
        cache_key = (hash(img_bytes), round(defect_repair_strength, 1), round(detail_enhancement, 1))
        if cache_key in self._cache:
            return self._cache[cache_key]

        # -------------------------------------------------------------------
        # Step 1: Ultra-Accurate Defect & Crack Inpainting
        # -------------------------------------------------------------------
        defect_mask = detect_scratches_and_defects(img_rgb, strength=defect_repair_strength)
        damaged_pixel_count = int(np.sum(defect_mask > 0))

        if damaged_pixel_count > 0:
            r_inpaint = max(3, int(4.0 * scale))
            healed_rgb = inpaint_scratches(img_rgb, defect_mask, inpaint_radius=r_inpaint, method="telea")
        else:
            healed_rgb = img_rgb.copy()

        # -------------------------------------------------------------------
        # Step 2: Archival De-fading & Dynamic Range Balancing
        # -------------------------------------------------------------------
        if defade_strength > 1.0:
            defaded_rgb = defade_and_neutralize_sepia(healed_rgb, defade_strength=defade_strength)
        else:
            defaded_rgb = healed_rgb

        # High-frequency detail sharpening
        restored_sharp_rgb, _ = restore_high_frequency_details(defaded_rgb, strength=detail_enhancement)

        # Restored Luminance Channel
        gray_healed = cv2.cvtColor(restored_sharp_rgb, cv2.COLOR_RGB2GRAY)

        # Subtle contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_clahe = clahe.apply(gray_healed)
        l_final_u8 = cv2.addWeighted(gray_healed, 0.75, l_clahe, 0.25, 0)
        bw_restored_rgb = np.stack([l_final_u8, l_final_u8, l_final_u8], axis=-1)

        # -------------------------------------------------------------------
        # Step 3: Neural Ensemble Colorization (SIGGRAPH17 + ECCV16)
        # -------------------------------------------------------------------
        l_in = l_final_u8.astype(np.float32) * (100.0 / 255.0)
        l_rs = cv2.resize(l_in, (resolution, resolution), interpolation=cv2.INTER_AREA)
        tens_rs_l = torch.from_numpy(l_rs).unsqueeze(0).unsqueeze(0).float().to(self.device)

        m_sig, m_ecc = get_models()

        with torch.inference_mode():
            out_sig = m_sig(tens_rs_l)
            out_ecc = m_ecc(tens_rs_l)
            w_sig = min(1.0, max(0.0, ensemble_blend))
            out_ab_tens = w_sig * out_sig + (1.0 - w_sig) * out_ecc

            if (orig_h, orig_w) != (resolution, resolution):
                out_ab_full = F.interpolate(out_ab_tens, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            else:
                out_ab_full = out_ab_tens

            out_ab_np = out_ab_full.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # -------------------------------------------------------------------
        # Step 4: Natural Skin Gamut Calibration & Eye Sclera Protection
        # -------------------------------------------------------------------
        # Strictly clamp 'a' (red) to natural human threshold [-14.0, 13.0]
        # and 'b' (yellow) to [-18.0, 20.0] to prevent red clown blotches
        a_ch = np.clip(out_ab_np[:, :, 0], -14.0, 13.0)
        b_ch = np.clip(out_ab_np[:, :, 1], -18.0, 20.0)

        # Desaturate high-contrast facial contours (eyes, pupils, nostrils)
        sobel = cv2.magnitude(cv2.Sobel(l_final_u8, cv2.CV_32F, 1, 0), cv2.Sobel(l_final_u8, cv2.CV_32F, 0, 1))
        eye_desat_mask = (sobel > 40) & (l_in > 25) & (l_in < 80)
        a_ch = np.where(eye_desat_mask, a_ch * 0.40, a_ch)

        # Snap colors to high-resolution luminance edges with bilateral filter
        a_cv = np.clip(a_ch + 128.0, 0, 255).astype(np.uint8)
        b_cv = np.clip(b_ch + 128.0, 0, 255).astype(np.uint8)

        a_cv = cv2.bilateralFilter(a_cv, d=7, sigmaColor=35, sigmaSpace=35)
        b_cv = cv2.bilateralFilter(b_cv, d=7, sigmaColor=35, sigmaSpace=35)

        lab_final = cv2.merge([l_final_u8, a_cv, b_cv])
        final_rgb = cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)

        total_latency_ms = round((time.time() - start_time) * 1000, 1)

        result_dict = {
            "original_damaged_rgb": img_rgb,
            "final_masterpiece_rgb": final_rgb,
            "restored_bw_rgb": bw_restored_rgb,
            "resolution_str": f"{orig_w} × {orig_h} px",
            "latency_ms": total_latency_ms,
        }

        if len(self._cache) > 32:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result_dict

        return result_dict

    def colorize(self, *args, **kwargs) -> Dict:
        return self.restore_and_relive(*args, **kwargs)


def export_image(img_rgb: np.ndarray, format_type: str = "PNG", quality: int = 95) -> str:
    """Exports image cleanly as lossless PNG."""
    filename = "revivegan_restored.png"
    Image.fromarray(img_rgb).save(filename, format="PNG", compress_level=4)
    return filename


studio_engine = ReviveGANEngine()
ChromaGANEngine = ReviveGANEngine
RELIVE_MODES = {}
PRESETS = RELIVE_MODES

