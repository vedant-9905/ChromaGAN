"""
app.py
======
ChromaGAN — Precision AI Photo Colorizer
Powered by Pre-Trained Deep Neural Colorization (Zhang et al. ECCV / SIGGRAPH)
Minimalist Workflow: Upload ➔ Interactive Split-Slider ➔ Download Color Image
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from PIL import Image
import gradio as gr
from skimage import color

# Add core to module search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.colorizers import siggraph17, eccv16, preprocess_img, postprocess_tens

# ---------------------------------------------------------------------------
# Pre-Trained SOTA Neural Network Initialization
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ChromaGAN] Loading pre-trained precision colorizer model on {device}...")

# Load pre-trained SIGGRAPH 2017 deep neural colorizer
colorizer = siggraph17(pretrained=True).eval()
colorizer.to(device)
print("[ChromaGAN] Pre-trained neural colorizer ready.")

OUTPUT_FILE = "colorized_photo.png"


# ---------------------------------------------------------------------------
# Precision Colorization Pipeline
# ---------------------------------------------------------------------------

def colorize_photo(input_image: np.ndarray | None):
    """
    Ingests an input photograph (RGB/Grayscale numpy array),
    extracts the luminance L channel, and uses the pre-trained deep neural network
    to synthesize accurate (a, b) chrominance channels matching commercial precision.
    """
    if input_image is None:
        return None, gr.update(visible=False, value=None)

    # If grayscale, tile to 3 channels for uniform processing
    if input_image.ndim == 2:
        img_rgb = np.tile(input_image[:, :, None], 3)
    else:
        img_rgb = input_image[:, :, :3]

    # Preprocess into native L and resized 256x256 L tensor
    (tens_l_orig, tens_l_rs) = preprocess_img(img_rgb, HW=(256, 256))
    tens_l_rs = tens_l_rs.to(device)

    # Neural inference
    with torch.no_grad():
        out_ab = colorizer(tens_l_rs).cpu()

    # Recombine in CIELAB space and convert to sRGB
    out_rgb = postprocess_tens(tens_l_orig, out_ab)
    colorized_u8 = (np.clip(out_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Grayscale original (exact original luminance and textures)
    l_orig_np = tens_l_orig.squeeze().cpu().numpy()
    bw_u8 = (np.clip(l_orig_np / 100.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    bw_rgb = np.stack([bw_u8, bw_u8, bw_u8], axis=-1)

    # Save high-resolution PNG for download
    Image.fromarray(colorized_u8).save(OUTPUT_FILE, format="PNG", quality=98)

    # Return slider tuple: (B&W Original, Colorized Output) and download path
    return (bw_rgb, colorized_u8), gr.update(visible=True, value=OUTPUT_FILE)


# ---------------------------------------------------------------------------
# Ultra-Minimalist UI
# ---------------------------------------------------------------------------

custom_css = """
.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 32px 16px !important;
}
.header-box {
    text-align: center;
    margin-bottom: 28px;
}
.header-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
    background: linear-gradient(135deg, #818cf8 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-sub {
    font-size: 15px;
    color: #94a3b8;
    margin: 0;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(title="AI Photo Colorizer") as demo:
    # Header
    gr.HTML(
        """
        <div class="header-box">
            <h1 class="header-title">AI Photo Colorizer</h1>
            <p class="header-sub">Upload an old black & white photograph to instantly restore natural, realistic colors.</p>
        </div>
        """
    )

    # Upload & Action
    with gr.Row():
        upload_input = gr.Image(
            label="Upload B&W Photo",
            type="numpy",
            sources=["upload", "clipboard"],
            height=280,
        )

    with gr.Row():
        colorize_btn = gr.Button("🎨 Colorize Photo", variant="primary", size="lg")

    # Interactive Before/After Split-Slider
    with gr.Row():
        slider_output = gr.ImageSlider(
            label="Before (B&W) vs. After (Color) — Drag Divider to Compare",
            type="numpy",
            height=520,
            interactive=True,
        )

    # 1-Click Download Button
    with gr.Row():
        download_btn = gr.DownloadButton(
            "⬇️ Download Color Photo",
            variant="primary",
            size="lg",
            visible=False,
        )

    # Quick Samples Gallery
    gr.Markdown("#### Quick Samples")
    examples = gr.Examples(
        examples=[
            ["assets/samples/ansel_adams3.jpg"],
            ["assets/samples/ansel_adams2.jpg"],
            ["assets/samples/classic_portrait.png"],
            ["assets/samples/vintage_car.png"],
            ["assets/samples/old_city.png"],
        ],
        inputs=[upload_input],
        outputs=[slider_output, download_btn],
        fn=colorize_photo,
        cache_examples=False,
    )

    # Wire Interactions
    colorize_btn.click(
        fn=colorize_photo,
        inputs=[upload_input],
        outputs=[slider_output, download_btn],
    )
    upload_input.change(
        fn=colorize_photo,
        inputs=[upload_input],
        outputs=[slider_output, download_btn],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3003))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=theme,
        css=custom_css,
        show_error=True,
    )
