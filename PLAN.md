# ChromaGAN: AI Black & White Photo Colorizer & Vintage Restorer

## 1. Project Overview & Problem Statement
**Problem Statement:**
"To develop an intuitive, real-time photo restoration and colorization studio powered by a Conditional Generative Adversarial Network (cGAN / Pix2Pix). ChromaGAN ingests black-and-white, grayscale, and vintage archival photographs, maps luminance ($L$) to chromatic distribution ($a, b$) in the CIELAB color space, and synthesizes natural, vibrant, and historically authentic colors with zero color bleeding."

### Why this architecture is the Generative AI Industry Standard:
- **Zero Medical Jargon**: No confusing DICOM, Hounsfield Units, or photon starvation math.
- **Pure ML Native Stack**: Built entirely on **PyTorch**, **TorchVision**, **OpenCV**, and **Gradio 6** (Hugging Face's industry-standard generative AI application framework).
- **Zero Node/TypeScript Overhead**: Single-command execution (`python app.py`), direct in-memory PyTorch tensor execution without serialization bottlenecks.
- **Hardware-Accelerated Before/After Split-Slider**: Native sub-pixel interactive split slider (`gr.ImageSlider`).
- **Mathematical Soundness (CIELAB Space)**: Separates Luminance ($L$) from Chrominance ($a, b$), eliminating geometry hallucinations and color bleeding.
- **1-Click Historical Presets Gallery**: Pre-loaded archival test cases (1920s Studio Portrait, 1930s Roadster Car, Historic City, Yosemite Wilderness).

---

## 2. Directory Layout
```
ChromaGAN/
├── app.py                   # Main Gradio 6 AI workstation (Port 3003)
├── core/
│   ├── __init__.py          # Core package exports
│   ├── models.py            # UNetColorGenerator & PatchGANColorDiscriminator
│   ├── losses.py            # cGAN Adversarial & L1 Losses
│   └── inference.py         # CIELAB color conversion, Hasler-Süsstrunk metric
├── assets/
│   └── samples/             # Historical sample photographs
├── create_samples.py        # Preset generation utility
├── requirements.txt         # Pinned Python dependencies
├── PLAN.md                  # Architecture & roadmap
└── README.md                # System documentation & usage guide
```

---

## 3. Execution Instructions
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch ChromaGAN Studio
python app.py
```
Open [http://localhost:3003](http://localhost:3003) in your browser.
