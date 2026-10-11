# ChromaGAN Backend

FastAPI inference server for the Conditional GAN (cGAN / Pix2Pix) Photo Colorization model in CIELAB color space.

## Architecture
- **Generator**: `UNetColorGenerator` with skip connections mapping $L \in [-1, 1] \to (a, b) \in [-1, 1]$.
- **Discriminator**: `PatchGANColorDiscriminator` (70×70 receptive field) scoring real vs generated $(L, a, b)$ triplets.
- **Objective**: $\mathcal{L}_{total} = \mathcal{L}_{cGAN}(G, D) + \lambda \mathcal{L}_1(ab_{real}, ab_{fake})$ with $\lambda = 100$.

## Setup & Run

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`
- Colorize Endpoint: `POST /api/colorize`
