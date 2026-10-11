# RadSwin-WGAN Backend

FastAPI inference server for the Residual U-Net WGAN-GP LDCT denoising model.

## Setup

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API

| Method | Endpoint      | Description |
|--------|---------------|-------------|
| GET    | /api/health   | Service health, device, model status |
| POST   | /api/denoise  | Denoise CT slice, returns base64 images + PSNR/SSIM |

### POST /api/denoise

**Form fields:**
- `file` (required): CT image (PNG/JPEG/BMP)
- `window_preset` (optional): `lung` (default) | `soft_tissue` | `default`

**Response JSON:**
```json
{
  "original_b64": "<base64 PNG>",
  "denoised_b64": "<base64 PNG>",
  "heatmap_b64":  "<base64 PNG>",
  "psnr":  34.21,
  "ssim":  0.8712,
  "reconstruction_fidelity": 87.12,
  "inference_time_ms": 42.3
}
```

## Interactive Docs

Visit `http://localhost:8000/docs` for Swagger UI.
