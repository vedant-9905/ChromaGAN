# 🎨 ChromaGAN: AI Black & White Photo Colorizer & Vintage Restorer

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

## Tech Stack (AI/ML Industry Standard)

- **Deep Learning Framework**: PyTorch 2.14+, TorchVision
- **Computer Vision & Colorimetry**: OpenCV-Python, NumPy, Pillow
- **Interactive Workstation**: Gradio 6 (Hugging Face)
- **Mathematical Color Space**: CIELAB ($L \in [0, 100]$, $a \in [-128, 127]$, $b \in [-128, 127]$)

---

## Directory Structure

```
ChromaGAN/
├── app.py                # Main Gradio 6 application (Port 3003)
├── core/
│   ├── __init__.py       # Core package exports
│   ├── models.py         # UNetColorGenerator & PatchGANColorDiscriminator
│   ├── losses.py         # cGAN Adversarial & L1 Chrominance Losses
│   └── inference.py      # CIELAB color conversion, Hasler-Süsstrunk metric
├── assets/
│   └── samples/          # Historical sample photographs (Portrait, Car, City, Landscape)
├── create_samples.py     # Script to generate sample B&W presets
├── requirements.txt      # Clean pinned Python dependencies
├── PLAN.md               # Architecture roadmap
└── README.md             # Project documentation & launch instructions
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch ChromaGAN Studio
```bash
python app.py
```

Open your browser at:
👉 **[http://localhost:3003](http://localhost:3003)**

---

## Key Studio Features

- **↔️ Interactive Split Slider**: Draggable before/after divider comparing Grayscale B&W input vs. cGAN Colorized Output in real-time.
- **🏛️ 1-Click Archival Presets**: Click any thumbnail in the examples gallery to instantly test (1920s Portrait, 1930s Roadster Car, Historic City, Yosemite Lake).
- **🎨 Color Grading Profiles**: Vibrant Modern, Natural Historic, Warm Golden Hour, Cool Cinematic.
- **⚡ Chromatic Saturation & Temperature**: Real-time slider controls for fine artistic calibration.
- **🔥 Turbo Chromatic Heatmap**: Visualizes $(a, b)$ chromatic energy spread across the photograph.
- **📊 Quantitative Scoreboard**: Displays Hasler & Süsstrunk Colorfulness Index, inference latency (ms), and native pixel matrix.
