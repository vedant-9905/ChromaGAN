"""
main.py
=======
RadSwin-WGAN FastAPI Application

Endpoints:
  POST /api/denoise  — Upload image, run WGAN-GP inference, return base64 results + metrics
  GET  /api/health   — Health check with device/model status

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.inference import build_generator, run_inference, WINDOW_PRESETS
from core.models import ResUNetGenerator


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

class AppState:
    generator: Optional[ResUNetGenerator] = None
    device: Optional[torch.device] = None
    startup_time: float = 0.0
    model_loaded: bool = False


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the generator on startup; teardown on shutdown."""
    app_state.startup_time = time.time()
    app_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[RadSwin-WGAN] Initializing generator on {app_state.device}...")
    try:
        app_state.generator = build_generator(app_state.device)
        app_state.model_loaded = True
        print("[RadSwin-WGAN] Generator ready.")
    except Exception as exc:
        print(f"[RadSwin-WGAN] WARNING: Failed to load generator: {exc}")
        app_state.model_loaded = False

    yield

    # Cleanup
    if app_state.generator is not None:
        del app_state.generator
        if app_state.device and app_state.device.type == "cuda":
            torch.cuda.empty_cache()
    print("[RadSwin-WGAN] Shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RadSwin-WGAN API",
    description=(
        "End-to-end LDCT image denoising API using a multi-scale Residual U-Net "
        "Generator trained with WGAN-GP + perceptual losses."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    gpu_name: Optional[str]
    torch_version: str


class DenoiseResponse(BaseModel):
    original_b64: str       # base64 PNG of windowed input
    denoised_b64: str       # base64 PNG of denoised output
    heatmap_b64: str        # base64 PNG of residual heatmap
    psnr: float             # Peak Signal-to-Noise Ratio (dB)
    ssim: float             # Structural Similarity Index [0,1]
    reconstruction_fidelity: float  # ssim * 100 (%)
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def ndarray_to_base64_png(arr: np.ndarray) -> str:
    """Encode a NumPy array (uint8) to a base64-encoded PNG string."""
    success, buf = cv2.imencode(".png", arr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Returns service health, device info, and model status."""
    uptime = time.time() - app_state.startup_time
    device_str = str(app_state.device) if app_state.device else "unknown"

    gpu_name: Optional[str] = None
    if app_state.device and app_state.device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(app_state.device)
        except Exception:
            gpu_name = "Unknown GPU"

    return HealthResponse(
        status="ok" if app_state.model_loaded else "degraded",
        model_loaded=app_state.model_loaded,
        device=device_str,
        uptime_seconds=round(uptime, 2),
        gpu_name=gpu_name,
        torch_version=torch.__version__,
    )


@app.post("/api/denoise", response_model=DenoiseResponse, tags=["Inference"])
async def denoise_image(
    file: UploadFile = File(..., description="CT scan image (PNG/JPEG/BMP)"),
    window_preset: str = Form(default="lung", description="HU window preset: lung | soft_tissue | default"),
) -> DenoiseResponse:
    """
    Accepts a CT slice image and returns:
    - Original (windowed) image as base64 PNG
    - WGAN-GP denoised image as base64 PNG
    - Residual heatmap (Jet colormap) as base64 PNG
    - Quantitative metrics: PSNR, SSIM, Reconstruction Fidelity
    """
    if not app_state.model_loaded or app_state.generator is None:
        raise HTTPException(
            status_code=503,
            detail="Generator model not loaded. Service unavailable.",
        )

    # Validate window preset
    valid_presets = list(WINDOW_PRESETS.keys())
    if window_preset not in valid_presets:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid window_preset '{window_preset}'. Must be one of: {valid_presets}",
        )

    # Read uploaded bytes
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    # Run inference
    t_start = time.perf_counter()
    try:
        results = run_inference(
            image_bytes=image_bytes,
            window_preset=window_preset,
            device=app_state.device,
            generator=app_state.generator,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    t_end = time.perf_counter()

    inference_ms = (t_end - t_start) * 1000.0

    # Encode outputs to base64 PNG
    try:
        original_b64 = ndarray_to_base64_png(results["original_u8"])
        denoised_b64 = ndarray_to_base64_png(results["denoised_u8"])
        heatmap_b64  = ndarray_to_base64_png(results["heatmap_bgr"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image encoding error: {exc}")

    return DenoiseResponse(
        original_b64=original_b64,
        denoised_b64=denoised_b64,
        heatmap_b64=heatmap_b64,
        psnr=round(results["psnr"], 4),
        ssim=round(results["ssim"], 4),
        reconstruction_fidelity=round(results["reconstruction_fidelity"], 2),
        inference_time_ms=round(inference_ms, 2),
    )


# ---------------------------------------------------------------------------
# Dev Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
