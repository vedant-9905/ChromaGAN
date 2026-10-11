# ChromaGAN: AI Black & White Photo Colorizer and Vintage Restorer

## 1. Project Overview & Problem Statement
**Problem Statement:**
"To develop an intuitive, real-time photo restoration and colorization studio powered by a Conditional Generative Adversarial Network (cGAN / Pix2Pix). ChromaGAN ingests black-and-white, grayscale, and vintage archival photographs, maps luminance ($L$) to chromatic distribution ($a, b$) in the CIELAB color space, and synthesizes natural, vibrant, and historically authentic colors with zero color bleeding."

### Why this project is 10x better to showcase & understand:
- **Zero Medical Jargon**: No confusing DICOM, Hounsfield Units, or CT physics.
- **Instant Visual Impact**: Everyone understands turning an old black-and-white picture into vibrant color.
- **Industry Standard GAN**: Uses Conditional GAN (Pix2Pix U-Net + PatchGAN) in the CIELAB color space—the exact foundational architecture used in Adobe Photoshop Neural Filters and commercial colorizers.
- **Fast, Interactive Demo**: 1-click historical presets (1920s vintage car, classic portrait, old city, landscape), split-screen slider, saturation control, and 1-click download.

---

## 2. Technical Stack
- **Architecture**: Conditional GAN (cGAN / Pix2Pix in CIELAB space)
  - **Generator**: U-Net with skip connections (Encoder: 8 downsampling Convs; Decoder: 7 upsampling Convs with skip links)
  - **Discriminator**: 70×70 PatchGAN Critic
  - **Losses**: Adversarial GAN Loss + $\mathcal{L}_1$ Chrominance Consistency Loss ($\lambda=100$)
- **Backend**: Python 3.11, FastAPI, PyTorch, TorchVision, OpenCV-Python, NumPy, Pydantic
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide-React (configured on port `3003`)
- **Theme**: Premium Dark AI Photo Studio (`#0B0F19`, `#121827`, Electric Violet `#8B5CF6`, Cyan `#06B6D4`)

---

## 3. Directory Layout
```
ChromaGAN/
├── PLAN.md                   # Complete architectural and execution specification
├── README.md                 # Project documentation and quickstart
├── backend/
│   ├── core/
│   │   ├── __init__.py       # Package exports
│   │   ├── models.py         # UNetColorGenerator & PatchGANColorDiscriminator
│   │   ├── losses.py         # cGAN Adversarial Loss & L1 Color Loss
│   │   └── inference.py      # CIELAB color space converter, colorization pipeline
│   ├── main.py               # FastAPI endpoints: POST /api/colorize, GET /api/health
│   ├── requirements.txt      # Dependencies
│   └── README.md
└── frontend/
    ├── app/
    │   ├── globals.css       # Studio dark theme & slider styles
    │   ├── layout.tsx        # App layout and metadata
    │   └── page.tsx          # ChromaGAN Studio main interactive page
    ├── components/
    │   ├── Header.tsx        # Studio header, model status, export button
    │   ├── Sidebar.tsx       # Photo uploader, historical presets, color styles
    │   ├── SplitSlider.tsx   # Interactive before/after split slider
    │   ├── SamplePresets.ts  # Historical sample photo generator & presets
    │   └── SideBySideGrid.tsx# Dual view comparison
    ├── types/
    │   └── index.ts          # TypeScript interfaces
    ├── package.json          # Next.js configured on port 3003
    ├── tailwind.config.ts
    └── next.config.js
```

---

## 4. Implementation Steps
1. **Backend**:
   - Re-implement `backend/core/models.py` with `UNetColorGenerator` and `PatchGANColorDiscriminator`.
   - Update `backend/core/losses.py` with cGAN losses.
   - Update `backend/core/inference.py` with CIELAB conversion and fast inference.
   - Update `backend/main.py` with `POST /api/colorize` and health check.
2. **Frontend**:
   - Update `frontend/types/index.ts`.
   - Build `frontend/components/SamplePresets.ts` with vintage B&W generators.
   - Update `frontend/components/Header.tsx`, `Sidebar.tsx`, `SplitSlider.tsx`, and `SideBySideGrid.tsx`.
   - Update `frontend/app/page.tsx` for the ChromaGAN Studio.
3. **Verification**:
   - Compile Python backend.
   - Build Next.js frontend (`npm run build`).
   - Test `npm run dev` on port 3003.
   - Commit & push to GitHub.
