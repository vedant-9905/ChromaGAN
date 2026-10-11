# RadSwin-WGAN: Architecture, Implementation & Execution Plan

## 1. System Overview & Problem Statement
Restoring high-fidelity anatomical structures from low-dose computed tomography (LDCT) scans while eliminating hallucinated micro-structures and mode collapse using:
- Multi-scale Residual Generator (ResUNet with identity shortcuts, InstanceNorm, LeakyReLU, Tanh)
- Wasserstein Discriminator (PatchGAN 70x70 receptive field with 1-Lipschitz gradient penalty $\lambda=10$)
- Multi-component perceptual & structural objective:
  $$\mathcal{L}_{total} = \mathcal{L}_{WGAN} + \alpha \mathcal{L}_1 + \beta \mathcal{L}_{SSIM} + \gamma \mathcal{L}_{perceptual}$$
- Hounsfield windowing (Lung [-600, 1500 HU], Soft Tissue [-160, 240 HU]) and dynamic range normalization.

---

## 2. Technical Stack
- **Backend**: Python 3.11, PyTorch, TorchVision, FastAPI, Uvicorn, OpenCV, NumPy, SciPy, Pydantic.
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide-React.
- **Theme**: Dark medical PACS / workstation aesthetic (`#090D16`, `#0F172A`, cyan/teal accents).

---

## 3. Implementation Breakdown

### Phase 1: Backend Architecture (`/backend`)
- [x] `backend/core/models.py`:
  - `ResUNetGenerator`: 4 downsampling stages, 6 residual blocks with identity shortcuts, 4 upsampling stages with skip connections, InstanceNorm, LeakyReLU(0.2), Tanh.
  - `PatchGANDiscriminator`: 70x70 receptive field, no Sigmoid (Wasserstein critic scores).
- [x] `backend/core/losses.py`:
  - `compute_gradient_penalty`: Exact 1-Lipschitz interpolation penalty ($\lambda=10$).
  - `MSSSIMLoss`: Differentiable multi-scale structural similarity loss.
  - `VGGPerceptualLoss`: Pre-trained VGG-19 features at `relu5_2`.
  - `GeneratorLoss`: Combined multi-objective generator loss.
- [x] `backend/core/inference.py`:
  - Hounsfield windowing (`lung`, `soft_tissue`, `default`).
  - Preprocessing and rescaling to $[-1, 1]$.
  - Generator inference with PyTorch.
  - Absolute residual difference heatmap with OpenCV Jet colormap.
  - Quantitative evaluation metrics: PSNR, SSIM, and Reconstruction Fidelity.
- [x] `backend/main.py`:
  - Lifespan management for model caching.
  - `GET /api/health`: System status, active device (CUDA/CPU), GPU model, uptime.
  - `POST /api/denoise`: Multi-part image ingestion, Hounsfield windowing, model execution, base64 response for input, denoised, heatmap, and metrics.
  - CORS middleware enabled for `http://localhost:3000`.
- [x] `backend/requirements.txt`: Pinned dependencies.

### Phase 2: Next.js 14 Frontend (`/frontend`)
- [ ] Next.js 14 project setup with App Router, TypeScript, and Tailwind CSS.
- [ ] Top Header:
  - System badge (Online/Ready/Degraded).
  - Runtime device indicator (CUDA GPU / CPU).
  - Quantitative metric badges (PSNR, SSIM, Reconstruction Fidelity, Latency).
- [ ] Left Radiologist Control Sidebar:
  - Drag-and-drop DICOM / CT slice uploader with file preview.
  - Sample presets:
    - Normal Dose reference
    - High Poisson/Gaussian Noise
    - Low-Dose CT Streak / Beam-Hardening Artifacts
  - Interactive Window Level (WL) and Window Width (WW) sliders with Lung / Soft Tissue presets.
  - Real-time trigger button for Denoise synthesis.
- [ ] Main Viewport:
  - 3-Way Comparative Synchronized Viewport:
    1. Low-Dose Input (Noisy).
    2. RadSwin-WGAN Reconstructed Output (Denoised).
    3. Artifact / Residual Difference Map (Jet Heatmap).
  - Interactive Split-Slider View (Before/After comparison with draggable divider).
  - Full-screen zoom and inspection tools.
- [ ] Synthetic CT Slice Generator & Preset assets.

### Phase 3: Verification & Execution
- Verify Next.js build and type-checking.
- Provide step-by-step local execution instructions.
