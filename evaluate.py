"""
evaluate.py
===========
Quantitative Evaluation and Benchmarking Suite for ChromaGAN.
Computes PSNR, SSIM, Delta-E, Hasler-Süsstrunk Colorfulness, and Latency.
Generates academic benchmark tables for reports and research papers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.pipeline import compute_colorfulness, studio_engine


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return float(20.0 * np.log10(max_pixel / np.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two RGB images.
    Uses Gaussian weighting window following Wang et al. (2004).
    """
    # Convert RGB to grayscale for structural similarity
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Gaussian kernel
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(gray1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(gray2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(gray1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(gray2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(gray1 * gray2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.mean(ssim_map))


def calculate_delta_e(img_rgb1: np.ndarray, img_rgb2: np.ndarray) -> float:
    """Computes mean CIE76 color difference Delta-E in CIELAB space."""
    lab1 = cv2.cvtColor(img_rgb1, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab2 = cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2LAB).astype(np.float64)
    delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    return float(np.mean(delta_e))


def run_benchmark(
    samples_dir: str = "assets/samples",
    output_json: str = "benchmark_results.json",
) -> Dict:
    print("=" * 70)
    print("  ChromaGAN Quantitative Evaluation & Deep Learning Benchmark Suite")
    print("=" * 70)

    dir_path = Path(samples_dir)
    # Filter ground truth RGB images (images from ILSVRC or samples that have color)
    image_paths = sorted([p for p in dir_path.glob("*.JPEG")] + [p for p in dir_path.glob("*.jpg")])

    if not image_paths:
        image_paths = sorted(list(dir_path.glob("*.png")))[:4]

    models_to_test = [
        "Portrait SOTA (SIGGRAPH17)",
        "Vibrant Scenery (ECCV16)",
        "Hybrid SOTA Ensemble",
    ]

    results_by_model: Dict[str, Dict[str, List[float]]] = {
        m: {"psnr": [], "ssim": [], "delta_e": [], "colorfulness": [], "latency_ms": []}
        for m in models_to_test
    }

    print(f"\nEvaluating {len(image_paths)} benchmark images across {len(models_to_test)} models...\n")

    for img_p in image_paths:
        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb_gt = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_gt = cv2.resize(rgb_gt, (256, 256), interpolation=cv2.INTER_AREA)

        # Baseline Grayscale for comparison
        gray = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        gt_colorfulness, _ = compute_colorfulness(rgb_gt)

        print(f"--- Testing Image: {img_p.name} (GT Colorfulness: {gt_colorfulness:.1f}) ---")

        for model_name in models_to_test:
            # Execute inference
            t0 = time.time()
            out = studio_engine.colorize(
                input_image=gray_rgb,
                model_choice=model_name,
                resolution=256,
                ensemble_blend=0.65,
                clahe_strength=0.0,  # Pure model comparison without CLAHE distortion
                edge_snap=0.0,
                temperature=0.0,
                tint=0.0,
                vibrance=0.0,
                saturation=1.0,
            )
            lat = (time.time() - t0) * 1000.0

            color_pred = out["colorized_rgb"]
            psnr = calculate_psnr(rgb_gt, color_pred)
            ssim = calculate_ssim(rgb_gt, color_pred)
            delta_e = calculate_delta_e(rgb_gt, color_pred)
            col_score = out["colorfulness_score"]

            results_by_model[model_name]["psnr"].append(psnr)
            results_by_model[model_name]["ssim"].append(ssim)
            results_by_model[model_name]["delta_e"].append(delta_e)
            results_by_model[model_name]["colorfulness"].append(col_score)
            results_by_model[model_name]["latency_ms"].append(lat)

            print(
                f"  [{model_name[:20]:20s}] PSNR: {psnr:5.2f} dB | SSIM: {ssim:5.3f} "
                f"| Delta-E: {delta_e:5.2f} | Color: {col_score:5.1f} | Latency: {lat:5.1f} ms"
            )

    # Compute averages
    summary_table = []
    print("\n" + "=" * 70)
    print("  SUMMARY BENCHMARK RESULTS (Average Across Test Set)")
    print("=" * 70)
    print(f"{'Model Architecture':<30} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Delta-E':<8} | {'Colorfulness':<12} | {'Latency (ms)':<10}")
    print("-" * 88)

    summary_dict = {}

    for m in models_to_test:
        avg_psnr = float(np.mean(results_by_model[m]["psnr"])) if results_by_model[m]["psnr"] else 0.0
        avg_ssim = float(np.mean(results_by_model[m]["ssim"])) if results_by_model[m]["ssim"] else 0.0
        avg_de = float(np.mean(results_by_model[m]["delta_e"])) if results_by_model[m]["delta_e"] else 0.0
        avg_col = float(np.mean(results_by_model[m]["colorfulness"])) if results_by_model[m]["colorfulness"] else 0.0
        avg_lat = float(np.mean(results_by_model[m]["latency_ms"])) if results_by_model[m]["latency_ms"] else 0.0

        summary_dict[m] = {
            "mean_psnr_db": round(avg_psnr, 2),
            "mean_ssim": round(avg_ssim, 3),
            "mean_delta_e": round(avg_de, 2),
            "mean_colorfulness": round(avg_col, 1),
            "mean_latency_ms": round(avg_lat, 1),
        }

        print(
            f"{m:<30} | {avg_psnr:<10.2f} | {avg_ssim:<8.3f} | {avg_de:<8.2f} | {avg_col:<12.1f} | {avg_lat:<10.1f}"
        )

    print("=" * 88)

    # Save to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\n[Benchmark] Results saved to {output_json}")

    return summary_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaGAN Benchmarking Suite")
    parser.add_argument("--samples_dir", type=str, default="assets/samples")
    parser.add_argument("--output_json", type=str, default="benchmark_results.json")
    args = parser.parse_args()

    run_benchmark(args.samples_dir, args.output_json)
