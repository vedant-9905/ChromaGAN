"""
app.py
======
ChromaGAN — Minimalist AI Photo Colorizer
Upload ➔ Compare with Slider ➔ Download Color Photo
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

from core.models import UNetColorGenerator
from core.inference import INFERENCE_SIZE

# ---------------------------------------------------------------------------
# Model Setup
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = UNetColorGenerator(in_channels=1, out_channels=2, base_filters=64)
generator.to(device)
generator.eval()

OUTPUT_PATH = "colorized_photo.png"


# ---------------------------------------------------------------------------
# Colorization Logic
# ---------------------------------------------------------------------------

def colorize_image(input_image: np.ndarray | None):
    """
    Takes an input image, restores natural colors using CIELAB cGAN,
    and returns a before/after tuple for the split-slider plus a download file path.
    """
    if input_image is None:
        return None, gr.update(visible=False, value=None)

    orig_h, orig_w = input_image.shape[:2]

    # Convert RGB -> BGR -> CIELAB
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(img_lab)

    # Grayscale original (exact original textures & lighting preserved)
    bw_original_rgb = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2RGB)

    # Neural prediction on L channel
    l_resized = cv2.resize(l_channel, (INFERENCE_SIZE, INFERENCE_SIZE), interpolation=cv2.INTER_AREA)
    l_norm = (l_resized.astype(np.float32) / 127.5) - 1.0
    l_tensor = torch.from_numpy(l_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        ab_pred = generator(l_tensor)

    ab_np = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    a_pred = cv2.resize(ab_np[:, :, 0], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    b_pred = cv2.resize(ab_np[:, :, 1], (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Natural vibrancy boost
    a_pred = a_pred * 1.30
    b_pred = b_pred * 1.30

    a_u8 = np.clip((a_pred * 127.0) + 128.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip((b_pred * 127.0) + 128.0, 0, 255).astype(np.uint8)

    # Recombine in CIELAB space and convert to RGB
    colorized_lab = cv2.merge([l_channel, a_u8, b_u8])
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)

    # Save for download
    Image.fromarray(colorized_rgb).save(OUTPUT_PATH, quality=95)

    return (bw_original_rgb, colorized_rgb), gr.update(visible=True, value=OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Minimalist UI
# ---------------------------------------------------------------------------

custom_css = """
/* Minimalist Clean Design */
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
}
.header-box {
    text-align: center;
    margin-bottom: 24px;
}
.header-title {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-sub {
    font-size: 15px;
    color: #94a3b8;
    margin-bottom: 0;
}
"""

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(title="Photo Colorizer") as demo:
    # Header
    gr.HTML(
        """
        <div class="header-box">
            <h1 class="header-title">Photo Colorizer</h1>
            <p class="header-sub">Upload an old black & white photograph to instantly restore natural colors.</p>
        </div>
        """
    )

    # Upload + Colorize
    with gr.Row():
        upload_input = gr.Image(
            label="Upload Old Photo",
            type="numpy",
            sources=["upload", "clipboard"],
            height=280,
        )

    with gr.Row():
        colorize_btn = gr.Button("🎨 Colorize Photo", variant="primary", size="lg")

    # Before / After Split-Slider
    with gr.Row():
        slider_output = gr.ImageSlider(
            label="Original (Left) vs. Colorized (Right) — Drag Divider to Compare",
            type="numpy",
            height=500,
            interactive=True,
        )

    # 1-Click Download Button
    with gr.Row():
        download_btn = gr.DownloadButton(
            "⬇️ Download Color Photo",
            variant="secondary",
            size="lg",
            visible=False,
        )

    # Quick Samples
    gr.Markdown("#### Quick Samples")
    examples = gr.Examples(
        examples=[
            ["assets/samples/classic_portrait.png"],
            ["assets/samples/vintage_car.png"],
            ["assets/samples/old_city.png"],
            ["assets/samples/yosemite_landscape.png"],
        ],
        inputs=[upload_input],
        outputs=[slider_output, download_btn],
        fn=colorize_image,
        cache_examples=False,
    )

    # Interactions
    colorize_btn.click(
        fn=colorize_image,
        inputs=[upload_input],
        outputs=[slider_output, download_btn],
    )
    upload_input.change(
        fn=colorize_image,
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
