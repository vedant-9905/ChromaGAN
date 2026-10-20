"""
app.py
======
ReviveGAN — AI Archival Photo Restoration
Minimal, Clean, Production-Grade Creative Interface.
"""

from __future__ import annotations

import os
import sys
import threading
import cv2
import numpy as np
from PIL import Image
import gradio as gr

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.pipeline import get_models, studio_engine, export_image

OUTPUT_FILE = "revivegan_restored.png"

# Pre-warm models & samples
get_models()

SAMPLE_PATHS = [
    "assets/samples/vintage_torn_portrait.png",
    "assets/samples/creased_native_woman.png",
    "assets/samples/crazed_archival_boy.png",
    "assets/samples/damaged_archival_couple.png",
    "assets/samples/creased_1950s_gentleman.png",
    "assets/samples/torn_adams_landscape.jpg",
]

def _prewarm_samples() -> None:
    for sp in SAMPLE_PATHS:
        if os.path.exists(sp):
            s_img = cv2.imread(sp)
            if s_img is not None:
                studio_engine.restore_and_relive(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB))

threading.Thread(target=_prewarm_samples, daemon=True).start()


# ---------------------------------------------------------------------------
# Minimal Clean CSS & Theme Tokens
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root, body.dark-theme, [data-theme="dark"] {
    --bg-page: #090d16;
    --card-bg: rgba(15, 23, 42, 0.75);
    --card-border: rgba(255, 255, 255, 0.08);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --switcher-bg: rgba(30, 41, 59, 0.85);
    --switcher-active: #6366f1;
}

body.light-theme, [data-theme="light"] {
    --bg-page: #f8fafc;
    --card-bg: rgba(255, 255, 255, 0.92);
    --card-border: rgba(0, 0, 0, 0.08);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --switcher-bg: rgba(226, 232, 240, 0.85);
    --switcher-active: #4f46e5;
}

body, .gradio-container {
    background-color: var(--bg-page) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    transition: background-color 0.2s ease, color 0.2s ease !important;
}

.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
}

/* Minimal Navbar */
.studio-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: var(--card-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    margin-bottom: 20px !important;
}

.studio-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.studio-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
}

.studio-sub {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0;
}

/* Theme Switcher */
.theme-switcher-container {
    display: inline-flex;
    align-items: center;
    background: var(--switcher-bg);
    padding: 4px;
    border-radius: 24px;
    gap: 2px;
}

.theme-pill-btn {
    border: none;
    background: transparent;
    color: var(--text-secondary);
    padding: 6px 12px;
    border-radius: 18px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
}

.theme-pill-btn.active {
    background: var(--switcher-active) !important;
    color: #ffffff !important;
}

/* Primary Action Button */
.btn-primary-action {
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
}

.btn-primary-action:hover {
    box-shadow: 0 6px 26px rgba(6, 182, 212, 0.5) !important;
    transform: translateY(-1px) !important;
}
"""

THEME_HEAD_JS = """
<script>
window.__setReviveTheme = function(mode) {
    try {
        localStorage.setItem('revive_theme', mode);
        var effective = mode;
        if (mode === 'system') {
            effective = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
        }
        var body = document.body;
        if (effective === 'dark') {
            body.classList.add('dark', 'dark-theme');
            body.classList.remove('light-theme');
        } else {
            body.classList.remove('dark', 'dark-theme');
            body.classList.add('light-theme');
        }
        ['light', 'dark', 'system'].forEach(function(id) {
            var el = document.getElementById('theme-btn-' + id);
            if (el) {
                if (id === mode) el.classList.add('active');
                else el.classList.remove('active');
            }
        });
    } catch(e) {}
};

document.addEventListener('DOMContentLoaded', function() {
    var saved = localStorage.getItem('revive_theme') || 'system';
    window.__setReviveTheme(saved);
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
            if ((localStorage.getItem('revive_theme') || 'system') === 'system') {
                window.__setReviveTheme('system');
            }
        });
    }
});
</script>
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


# ---------------------------------------------------------------------------
# Core Processing Function
# ---------------------------------------------------------------------------

def process_photo(input_image: np.ndarray | None):
    if input_image is None:
        return None, gr.update(visible=False, value=None)

    res = studio_engine.restore_and_relive(input_image)
    if not res:
        return None, gr.update(visible=False, value=None)

    orig_rgb = res["original_damaged_rgb"]
    final_rgb = res["final_masterpiece_rgb"]

    export_path = export_image(final_rgb)
    return (orig_rgb, final_rgb), gr.update(visible=True, value=export_path)


# ---------------------------------------------------------------------------
# Minimal UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="ReviveGAN") as demo:

    # 1. Clean Header
    gr.HTML(
        """
        <div class="studio-navbar">
            <div class="studio-brand">
                <span style="font-size: 26px;">✨</span>
                <div>
                    <h1 class="studio-title">ReviveGAN</h1>
                    <p class="studio-sub">Archival Photo Restoration & Neural Revival</p>
                </div>
            </div>
            <div class="theme-switcher-container">
                <button type="button" id="theme-btn-light" class="theme-pill-btn" onclick="window.__setReviveTheme('light')">☀️ Light</button>
                <button type="button" id="theme-btn-dark" class="theme-pill-btn active" onclick="window.__setReviveTheme('dark')">🌙 Dark</button>
                <button type="button" id="theme-btn-system" class="theme-pill-btn" onclick="window.__setReviveTheme('system')">💻 System</button>
            </div>
        </div>
        """
    )

    # 2. Upload Box & Action Button
    with gr.Row():
        upload_input = gr.Image(
            label="Upload Image",
            type="numpy",
            sources=["upload", "clipboard"],
            height=240,
        )

    with gr.Row():
        restore_btn = gr.Button(
            "Restore Photo",
            variant="primary",
            size="lg",
            elem_classes=["btn-primary-action"],
        )

    # 3. Clean Before / After Slider
    with gr.Row():
        slider_output = gr.ImageSlider(
            label="Before / After",
            type="numpy",
            height=580,
            interactive=True,
        )

    # 4. Download Button
    with gr.Row():
        download_btn = gr.DownloadButton(
            "Download Restored Photo",
            variant="primary",
            size="lg",
            visible=False,
            elem_classes=["btn-primary-action"],
        )

    # 5. Showcase Examples
    gr.Markdown("### Examples")
    gr.Examples(
        examples=[
            ["assets/samples/vintage_torn_portrait.png"],
            ["assets/samples/creased_native_woman.png"],
            ["assets/samples/crazed_archival_boy.png"],
            ["assets/samples/damaged_archival_couple.png"],
            ["assets/samples/creased_1950s_gentleman.png"],
            ["assets/samples/torn_adams_landscape.jpg"],
        ],
        inputs=[upload_input],
        label="Select a damaged photograph",
        examples_per_page=6,
    )

    # Interactions
    restore_btn.click(
        fn=process_photo,
        inputs=[upload_input],
        outputs=[slider_output, download_btn],
    )

    upload_input.change(
        fn=process_photo,
        inputs=[upload_input],
        outputs=[slider_output, download_btn],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import socket

    def _find_free_port(preferred: int = 3003, max_tries: int = 10) -> int:
        for p in range(preferred, preferred + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", p))
                    return p
                except OSError:
                    continue
        return preferred

    port = _find_free_port(int(os.environ.get("PORT", 3003)))
    share = "--share" in sys.argv or os.environ.get("SHARE", "false").lower() in ("true", "1")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        theme=THEME,
        css=CUSTOM_CSS,
        head=THEME_HEAD_JS,
        show_error=True,
    )
