"""
main.py
=======
ChromaGAN FastAPI Application
Production-grade REST API for cGAN B&W Photo Colorization.

Endpoints:
  POST /api/colorize — Ingests B&W photo, predicts CIELAB chrominance (a, b),
                       returns colorized result, chromatic heatmap, and metrics.
  GET  /api/health   — System and PyTorch device health check.

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.inference import build_generator, run_colorization
from core.models import UNetColorGenerator


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------

class AppState:
    generator: Optional[UNetColorGenerator] = None
    device: Optional[torch.device] = None
    startup_time: float = 0.0
    model_loaded: bool = False


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan Management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes PyTorch generator on startup and handles clean teardown."""
    app_state.startup_time = time.time()
    app_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[ChromaGAN] Initializing U-Net Color Generator on {app_state.device}...")
    try:
        app_state.generator = build_generator(app_state.device)
        app_state.model_loaded = True
        print("[ChromaGAN] U-Net Color Generator loaded successfully.")
    except Exception as exc:
        print(f"[ChromaGAN] WARNING: Model loading encountered an error: {exc}")
        app_state.model_loaded = False

    yield

    if app_state.generator is not None:
        del app_state.generator
        if app_state.device and app_state.device.type == "cuda":
            torch.cuda.empty_cache()
    print("[ChromaGAN] Service shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI App Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ChromaGAN Studio API",
    description="Conditional Generative Adversarial Network (cGAN) for photo colorization in CIELAB color space.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration allowing local development workstations
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    gpu_name: Optional[str]
    torch_version: str


class ColorizeResponse(BaseModel):
    original_b64: str          # Grayscale B&W original
    colorized_b64: str         # Synthesized full-color image
    heatmap_b64: str           # Chromatic distribution heatmap
    colorfulness_score: float  # Hasler-Süsstrunk metric
    resolution: str
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ndarray_to_base64_png(img_bgr: np.ndarray) -> str:
    """Encodes BGR numpy array to base64 PNG string."""
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image to PNG format.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Returns runtime health, device backend, and PyTorch status."""
    uptime = time.time() - app_state.startup_time
    device_str = str(app_state.device) if app_state.device else "unknown"

    gpu_name: Optional[str] = None
    if app_state.device and app_state.device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(app_state.device)
        except Exception:
            gpu_name = "Unknown CUDA Device"

    return HealthResponse(
        status="ok" if app_state.model_loaded else "degraded",
        model_loaded=app_state.model_loaded,
        device=device_str,
        uptime_seconds=round(uptime, 2),
        gpu_name=gpu_name,
        torch_version=torch.__version__,
    )


@app.post("/api/colorize", response_model=ColorizeResponse, tags=["Inference"])
async def colorize_photo(
    file: UploadFile = File(..., description="B&W or Grayscale photograph (PNG/JPEG/BMP)"),
    saturation: float = Form(default=1.2, description="Color saturation multiplier [0.5, 2.5]"),
    color_style: str = Form(
        default="vibrant",
        description="Artistic color grade: natural | vibrant | warm_vintage | cool_cinematic",
    ),
) -> ColorizeResponse:
    """
    Colorizes a black-and-white or historical photograph:
    - Extracts Luminance channel L
    - U-Net Generator synthesizes (a, b) chrominance channels
    - Recombines to RGB and returns high-resolution colorized image + metrics
    """
    if not app_state.model_loaded or app_state.generator is None:
        raise HTTPException(status_code=503, detail="ChromaGAN Generator not loaded.")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}")

    t_start = time.perf_counter()
    try:
        results = run_colorization(
            image_bytes=image_bytes,
            saturation=max(0.5, min(2.5, saturation)),
            color_style=color_style,
            device=app_state.device,
            generator=app_state.generator,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    t_end = time.perf_counter()

    latency_ms = (t_end - t_start) * 1000.0

    original_b64 = ndarray_to_base64_png(results["bw_original_bgr"])
    colorized_b64 = ndarray_to_base64_png(results["colorized_bgr"])
    heatmap_b64 = ndarray_to_base64_png(results["chroma_heatmap"])

    return ColorizeResponse(
        original_b64=original_b64,
        colorized_b64=colorized_b64,
        heatmap_b64=heatmap_b64,
        colorfulness_score=results["colorfulness_score"],
        resolution=results["resolution"],
        inference_time_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
