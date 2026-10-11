"""
app.py
======
ChromaGAN Studio — Industry-Standard cGAN Photo Colorization Application.
Powered by PyTorch, TorchVision, OpenCV, CIELAB color space, and Gradio 6.

Run with:
  python app.py
"""

from __future__ import annotations

import time
import os
import cv2
import numpy as np
import torch
import gradio as gr

from core.models import UNetColorGenerator
from core.inference import compute_colorfulness, generate_chroma_heatmap, INFERENCE_SIZE

# ---------------------------------------------------------------------------
# Device & Model Initialization
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ChromaGAN] Initializing U-Net Color Generator on device: {device}")

generator = UNetColorGenerator(in_channels=1, out_channels=2, base_filters=64)
generator.to(device)
generator.eval()
print("[ChromaGAN] Generator weights initialized and ready for inference.")


# ---------------------------------------------------------------------------
# Core Colorization Engine
# ---------------------------------------------------------------------------

def colorize_pipeline(
    input_image: np.ndarray | None,
    color_style: str,
    saturation: float,
    temperature: float,
):
    """
    Executes the end-to-end CIELAB cGAN photo colorization pipeline:
    1. Extracts Luminance (L) channel (preserving 100% of photographic texture)
    2. Passes normalized L into U-Net Generator
    3. Synthesizes (a, b) chromatic distribution
    4. Applies user color grading and saturation
    5. Reconstructs RGB and computes Hasler-Süsstrunk colorfulness metric
    """
    if input_image is None:
        raise gr.Error("Please upload an image or choose one from the sample presets below.")

    t_start = time.perf_counter()
    orig_h, orig_w = input_image.shape[:2]

    # Convert RGB -> BGR for OpenCV processing
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Convert to CIELAB space
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(img_lab)

    # Grayscale photographic reference (for before/after comparison)
    bw_original_rgb = cv2.cvtColor(l_channel, cv2.COLOR_GRAY2RGB)

    # Preprocess L channel for neural network (256x256)
    l_resized = cv2.resize(l_channel, (INFERENCE_SIZE, INFERENCE_SIZE), interpolation=cv2.INTER_AREA)
    l_norm = (l_resized.astype(np.float32) / 127.5) - 1.0
    l_tensor = torch.from_numpy(l_norm).unsqueeze(0).unsqueeze(0).to(device)

    # Generator forward pass
    with torch.no_grad():
        ab_pred = generator(l_tensor)

    # Map back to numpy
    ab_pred_np = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
    a_pred = ab_pred_np[:, :, 0]
    b_pred = ab_pred_np[:, :, 1]

    # Resize chromatic channels back to native photograph dimensions
    a_full = cv2.resize(a_pred, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    b_full = cv2.resize(b_pred, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Apply Artistic Color Grading Presets
    if color_style == "Warm Golden Hour":
        a_full = a_full * 1.15 + 0.08
        b_full = b_full * 1.30 + 0.16
    elif color_style == "Cool Cinematic":
        a_full = a_full * 0.95 - 0.06
        b_full = b_full * 0.85 - 0.12
    elif color_style == "Vibrant Modern":
        a_full = a_full * 1.35
        b_full = b_full * 1.35
    else:  # Natural Historic
        a_full = a_full * 1.05
        b_full = b_full * 1.05

    # Apply Manual Temperature bias
    b_full += temperature * 0.25
    a_full += temperature * 0.10

    # Apply Saturation multiplier
    a_full *= saturation
    b_full *= saturation

    # Clip to uint8 CIELAB range [0, 255]
    a_u8 = np.clip((a_full * 127.0) + 128.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip((b_full * 127.0) + 128.0, 0, 255).astype(np.uint8)

    # Recombine (L, a, b) -> BGR -> RGB
    colorized_lab = cv2.merge([l_channel, a_u8, b_u8])
    colorized_bgr = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)

    # Metrics
    colorfulness = round(compute_colorfulness(colorized_bgr), 2)
    chroma_heatmap_bgr = generate_chroma_heatmap(a_u8, b_u8)
    chroma_heatmap_rgb = cv2.cvtColor(chroma_heatmap_bgr, cv2.COLOR_BGR2RGB)

    t_end = time.perf_counter()
    latency_ms = round((t_end - t_start) * 1000.0, 1)

    slider_tuple = (bw_original_rgb, colorized_rgb)
    resolution_str = f"{orig_w} × {orig_h}"

    return (
        slider_tuple,
        colorized_rgb,
        chroma_heatmap_rgb,
        colorfulness,
        f"{latency_ms} ms",
        resolution_str,
    )


# ---------------------------------------------------------------------------
# Gradio 6 Interface Construction
# ---------------------------------------------------------------------------

custom_css = """
/* Dark Studio Aesthetic */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
.header-badge {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: inline-block;
}
"""

theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(title="ChromaGAN Studio | cGAN Photo Colorizer") as demo:
    # Top Header
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # 🎨 ChromaGAN: Neural B&W Photo Colorizer
                <span class="header-badge">Pix2Pix cGAN</span> <span class="header-badge">CIELAB Space</span> <span class="header-badge">Industry Standard</span>
                
                *Ingests archival black-and-white photographs, maps luminance ($L$) to chromatic distribution ($a, b$) in the CIELAB color space, and synthesizes natural, vibrant colors with zero color bleeding.*
                """
            )

    # Main Studio Layout
    with gr.Row():
        # Left Control Panel
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Input Photograph")
            input_image = gr.Image(
                label="Upload B&W / Vintage Photo",
                type="numpy",
                sources=["upload", "clipboard"],
            )

            with gr.Accordion("⚙️ Color Grading & Artistic Controls", open=True):
                color_style = gr.Radio(
                    choices=["Vibrant Modern", "Natural Historic", "Warm Golden Hour", "Cool Cinematic"],
                    value="Vibrant Modern",
                    label="Artistic Color Grade Palette",
                    info="Applies calibrated chromatic curve profiles",
                )
                saturation = gr.Slider(
                    minimum=0.5,
                    maximum=2.5,
                    value=1.2,
                    step=0.1,
                    label="Chromatic Saturation",
                    info="Multiplier for (a, b) chrominance intensity",
                )
                temperature = gr.Slider(
                    minimum=-0.5,
                    maximum=0.5,
                    value=0.0,
                    step=0.05,
                    label="Color Temperature",
                    info="Shift between cool blues and warm ambers",
                )

            colorize_btn = gr.Button(
                "✨ Synthesize Color with cGAN",
                variant="primary",
                size="lg",
            )

        # Right Viewport & Results
        with gr.Column(scale=2):
            gr.Markdown("### 🔬 Diagnostic Inspection Viewport")
            with gr.Tabs():
                with gr.TabItem("↔️ Interactive Split Slider"):
                    comparison_slider = gr.ImageSlider(
                        label="Before (B&W Input) vs. After (cGAN Colorized)",
                        type="numpy",
                    )
                with gr.TabItem("🖼️ Full-Resolution Color Output"):
                    colorized_output = gr.Image(
                        label="Synthesized High-Fidelity Color Photograph",
                        type="numpy",
                    )
                with gr.TabItem("🔥 Chromatic Energy Heatmap"):
                    heatmap_output = gr.Image(
                        label="CIELAB Chromatic Spread Energy (Turbo Colormap)",
                        type="numpy",
                    )

            # Quantitative Metrics Scoreboard
            with gr.Row():
                colorfulness_stat = gr.Number(
                    label="Colorfulness Index (Hasler & Süsstrunk)",
                    precision=2,
                )
                latency_stat = gr.Textbox(
                    label="Inference Latency",
                )
                resolution_stat = gr.Textbox(
                    label="Native Resolution",
                )

    # 1-Click Historical Presets Gallery
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🏛️ Archival Presets (1-Click Instant Test)")
            examples = gr.Examples(
                examples=[
                    ["assets/samples/classic_portrait.png", "Vibrant Modern", 1.2, 0.0],
                    ["assets/samples/vintage_car.png", "Vibrant Modern", 1.3, 0.05],
                    ["assets/samples/old_city.png", "Warm Golden Hour", 1.2, 0.1],
                    ["assets/samples/yosemite_landscape.png", "Cool Cinematic", 1.2, -0.05],
                ],
                inputs=[input_image, color_style, saturation, temperature],
                outputs=[
                    comparison_slider,
                    colorized_output,
                    heatmap_output,
                    colorfulness_stat,
                    latency_stat,
                    resolution_stat,
                ],
                fn=colorize_pipeline,
                cache_examples=False,
            )

    # Wire event handler
    colorize_btn.click(
        fn=colorize_pipeline,
        inputs=[input_image, color_style, saturation, temperature],
        outputs=[
            comparison_slider,
            colorized_output,
            heatmap_output,
            colorfulness_stat,
            latency_stat,
            resolution_stat,
        ],
    )


# ---------------------------------------------------------------------------
# Entry Point (Configured on Port 3003)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3003))
    print(f"[ChromaGAN] Launching Studio on http://localhost:{port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        theme=theme,
        css=custom_css,
    )
