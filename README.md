# RadSwin-WGAN: Full-Stack Generative Adversarial CT Synthesis Framework

An end-to-end, full-stack generative adversarial image synthesis framework utilizing a multi-scale residual Generator and a Wasserstein Discriminator with Gradient Penalty (WGAN-GP), augmented by perceptual and structural dissimilarity losses, to restore high-fidelity anatomical structures from low-dose computed tomography (LDCT) scans while eliminating hallucinated micro-structures and mode collapse.

---

## Architecture Specification

```
                                 ┌──────────────────────────────────┐
                                 │ Low-Dose CT (LDCT) Single-Ch HU │
                                 └─────────────────┬────────────────┘
                                                   │
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ResUNet Generator                                      │
│                                                                                        │
│  [Conv 7x7, s1, f64]  ──► [Conv 4x4, s2, f128] ──► [Conv 4x4, s2, f256] ──► [f512]    │
│           │                       │                       │                  │         │
│     (Skip Link 1)           (Skip Link 2)           (Skip Link 3)            │         │
│           │                       │                       │                  ▼         │
│           │                       │                       │        [6x ResBlocks, 512] │
│           │                       │                       │                  │         │
│           ▼                       ▼                       ▼                  ▼         │
│      [Dec1 + Skip]  ◄───    [Dec2 + Skip]   ◄───    [Dec3 + Skip]   ◄───  [Dec4 + Skip]│
│           │                                                                            │
│           ▼                                                                            │
│  [Conv 7x7, Tanh]  ──► Normalized Output [-1, 1]                                       │
└──────────────────────────────────────────────────┬─────────────────────────────────────┘
                                                   │
                                                   ▼
       ┌───────────────────────────────────────────┴───────────────────────────────────────┐
       │                                                                                   │
       ▼                                                                                   ▼
┌──────────────────────────────┐                                       ┌───────────────────────────────────┐
│     PatchGAN Discriminator   │                                       │   Multi-Objective Loss Suite      │
│  (70x70 receptive field,     │                                       │  - WGAN-GP (λ=10 1-Lipschitz)     │
│   InstanceNorm, raw scores)  │                                       │  - MS-SSIM (Differentiable)       │
└──────────────────────────────┘                                       │  - L1 Pixel Loss                  │
                                                                       │  - VGG-19 Perceptual (relu5_2)    │
                                                                       └───────────────────────────────────┘
```

---

## Directory Layout

```
radswin-wgan/
├── PLAN.md                     # Complete execution and architecture roadmap
├── README.md                   # System guide and startup instructions
├── backend/
│   ├── core/
│   │   ├── __init__.py         # Package exports
│   │   ├── models.py           # ResUNet Generator & PatchGAN Discriminator
│   │   ├── losses.py           # WGAN-GP (λ=10), MS-SSIM, VGG-19 relu5_2 Loss
│   │   └── inference.py        # Hounsfield windowing, metrics (PSNR/SSIM), Jet heatmap
│   ├── main.py                 # FastAPI endpoints: POST /api/denoise, GET /api/health
│   ├── requirements.txt        # Pinned PyTorch, FastAPI, OpenCV dependencies
│   └── README.md
└── frontend/
    ├── app/
    │   ├── globals.css         # PACS dark-mode styles, scanlines, range thumbs
    │   ├── layout.tsx          # Root layout and metadata
    │   └── page.tsx            # Radiologist workstation main view
    ├── components/
    │   ├── Header.tsx          # Status HUD, GPU/CPU indicator, PSNR/SSIM badges
    │   ├── Sidebar.tsx         # Ingestion, presets, HU windowing sliders
    │   ├── ComparisonGrid.tsx  # 3-way comparative view (Input, Denoised, Heatmap)
    │   ├── SplitSlider.tsx     # Draggable micro-structure inspector
    │   └── SyntheticCTGenerator.ts # Procedural CT slice & artifact generator
    ├── types/
    │   └── index.ts            # TypeScript interfaces
    ├── package.json
    ├── tsconfig.json
    ├── tailwind.config.ts
    └── next.config.js
```

---

## Quickstart

### 1. Backend Service

```bash
cd backend

# Create & activate environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Health Status: `http://localhost:8000/api/health`
- Swagger Docs: `http://localhost:8000/docs`

### 2. Frontend Workstation

```bash
cd frontend

# Install dependencies
npm install

# Start Next.js development server
npm run dev
```

- Radiologist Workstation: `http://localhost:3000`
