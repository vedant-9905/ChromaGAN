# ChromaGAN: AI Black & White Photo Colorizer & Vintage Restorer

An intuitive, real-time photo restoration and colorization studio powered by a Conditional Generative Adversarial Network (cGAN / Pix2Pix). ChromaGAN ingests black-and-white, grayscale, and vintage archival photographs, maps luminance ($L$) to chromatic distribution ($a, b$) in the CIELAB color space, and synthesizes natural, vibrant, and historically authentic colors with zero color bleeding.

---

## Architecture Specification

```
                               ┌─────────────────────────────────┐
                               │ Grayscale Photo Luminance (L)   │
                               │ Shape: [B, 1, H, W] in [-1, 1]  │
                               └────────────────┬────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              U-Net Generator with Skip Links                           │
│                                                                                        │
│  [Conv 4x4, s2, f64]  ──► [Conv 4x4, s2, f128] ──► [Conv 4x4, s2, f256] ──► [f512]    │
│           │                       │                       │                  │         │
│     (Skip Link 1)           (Skip Link 2)           (Skip Link 3)            │         │
│           │                       │                       │                  ▼         │
│           │                       │                       │             [Bottleneck]   │
│           │                       │                       │                  │         │
│           ▼                       ▼                       ▼                  ▼         │
│      [Up1 + Skip]   ◄───    [Up2 + Skip]    ◄───    [Up3 + Skip]    ◄─── [Up4 + Skip]  │
│           │                                                                            │
│           ▼                                                                            │
│  [Conv 3x3, Tanh]  ──► Predicted Chrominance (a, b) in [-1, 1]                         │
└───────────────────────────────────────────────┬────────────────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Recombination in CIELAB Color Space                             │
│                  (L, a_pred, b_pred) ──► Reconstructed RGB Color Image                 │
└───────────────────────────────────────────────┬────────────────────────────────────────┘
                                                │
                                                ▼
                ┌───────────────────────────────┴───────────────────────────────┐
                │                                                               │
                ▼                                                               ▼
┌───────────────────────────────────────────────┐               ┌───────────────────────────────┐
│          PatchGAN Discriminator               │               │        cGAN Loss Suite        │
│    (70×70 receptive field, evaluates joint    │               │  - Adversarial Loss (BCE)     │
│     realism of full [L, a, b] channels)       │               │  - L1 Chrominance Loss (λ=100)│
└───────────────────────────────────────────────┘               └───────────────────────────────┘
```

---

## Key Advantages of the CIELAB GAN Formulation

1. **Zero Hallucination of Geometry**: Because luminance $L$ is passed directly through from the original photograph, light, shadows, textures, and edges remain 100% faithful to the historical original.
2. **Dedicated Chromatic Learning**: The network solves a 2-channel prediction task $(a, b)$ instead of an unconstrained 3-channel RGB synthesis task, eliminating color bleeding and desaturation artifacts.
3. **Instant Visual Impact**: Transforms 100-year-old archival photos into rich, authentic modern color images.

---

## Directory Structure

```
ChromaGAN/
├── PLAN.md                     # Architecture specification & roadmap
├── README.md                   # System guide and startup instructions
├── backend/
│   ├── core/
│   │   ├── __init__.py         # Core package exports
│   │   ├── models.py           # UNetColorGenerator & PatchGANColorDiscriminator
│   │   ├── losses.py           # Conditional adversarial and L1 chrominance loss
│   │   └── inference.py        # CIELAB color conversion, style grades, metrics
│   ├── main.py                 # FastAPI endpoints: POST /api/colorize, GET /api/health
│   ├── requirements.txt        # Backend dependencies
│   └── README.md
└── frontend/
    ├── app/
    │   ├── globals.css         # Studio dark-mode styles, scanlines, range sliders
    │   ├── layout.tsx          # Root layout and metadata
    │   └── page.tsx            # ChromaGAN studio main workstation
    ├── components/
    │   ├── Header.tsx          # Studio status HUD, metrics display, export HD button
    │   ├── Sidebar.tsx         # Upload, historical presets, color styles, saturation
    │   ├── SplitSlider.tsx     # Draggable interactive split before/after inspector
    │   ├── SideBySideGrid.tsx  # Tri-view studio (Input, Colorized, Chromatic Heatmap)
    │   └── SamplePresets.ts    # Procedural vintage B&W photo generator
    ├── types/
    │   └── index.ts            # TypeScript interfaces
    ├── package.json            # Configured for Port 3003
    ├── tailwind.config.ts
    └── next.config.js
```

---

## Quickstart

### 1. Launch Frontend Studio (Port 3003)

```bash
cd frontend
npm install
npm run dev
```

Open your browser at:
👉 **[http://localhost:3003](http://localhost:3003)**

### 2. Launch Backend Inference Server (Port 8000)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate       # On Windows
# source .venv/bin/activate  # On Linux/macOS
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive Swagger Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`
