"""
app.py
======
ReviveGAN Studio Pro — Deep Generative Archival Photo Restoration & Neural Relive Suite.
Transforms aged, torn, scratched, faded, and monochrome photographs into pristine,
vibrantly relived masterpieces.

Features:
- Multi-scale automated crack, scratch, fold crease, and tear healing (Inpainting)
- High-frequency micro-texture synthesis and facial detail enhancement
- Archival de-fading, sepia neutralization, and adaptive dynamic range recovery
- SOTA dual neural ensemble colorization (Zhang SIGGRAPH17 + ECCV16)
- Day Theme (Light), Night Theme (Dark), and System Default toggle with persistence
- Interactive Split-Screen Slider, Side-by-Side View, and Technical Defect Map
- Sub-second in-memory LRU caching for instant re-renders
"""

from __future__ import annotations

import os
import sys
import time
import threading
import cv2
import numpy as np
from PIL import Image
import gradio as gr

# Add workspace directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.pipeline import (
    get_models,
    studio_engine,
    RELIVE_MODES,
    export_image,
)

OUTPUT_FILE = "revivegan_master_restoration.png"

# ---------------------------------------------------------------------------
# Pre-warm SOTA Neural Models & Sample Gallery at Startup
# ---------------------------------------------------------------------------

print("[ReviveGAN] Initializing neural models and restoration pipeline...")
get_models()

SAMPLE_PATHS = [
    "assets/samples/vintage_torn_portrait.png",
    "assets/samples/creased_native_woman.png",
    "assets/samples/crazed_archival_boy.png",
    "assets/samples/damaged_archival_couple.png",
    "assets/samples/creased_1950s_gentleman.png",
    "assets/samples/torn_adams_landscape.jpg",
]

def _prewarm_samples_async() -> None:
    """Pre-warms gallery in background thread to ensure sub-millisecond click response."""
    for sp in SAMPLE_PATHS:
        if os.path.exists(sp):
            s_img = cv2.imread(sp)
            if s_img is not None:
                s_rgb = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
                studio_engine.restore_and_relive(s_rgb)
    print("[ReviveGAN] Background sample gallery pre-warming completed.")

threading.Thread(target=_prewarm_samples_async, daemon=True).start()
print("[ReviveGAN] Restoration engine initialized. Studio ready for instant launch.")


# ---------------------------------------------------------------------------
# Styling & Theme Design System (Dark, Light & System Modes)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Theme Design System Tokens */
:root {
    --bg-page: #080c14;
    --bg-mesh-1: rgba(99, 102, 241, 0.14);
    --bg-mesh-2: rgba(6, 182, 212, 0.08);
    --bg-mesh-3: rgba(168, 85, 247, 0.06);
    --card-bg: rgba(15, 23, 42, 0.78);
    --card-border: rgba(255, 255, 255, 0.09);
    --card-shadow: 0 16px 40px -12px rgba(0, 0, 0, 0.65);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --hud-bg: rgba(30, 41, 59, 0.90);
    --hud-border: rgba(255, 255, 255, 0.11);
    --hud-text: #f8fafc;
    --switcher-bg: rgba(30, 41, 59, 0.85);
    --switcher-btn-bg: transparent;
    --switcher-btn-text: #94a3b8;
    --switcher-active-bg: linear-gradient(135deg, #6366f1, #06b6d4);
    --switcher-active-text: #ffffff;
    --input-bg: rgba(15, 23, 42, 0.6);
    --input-border: rgba(255, 255, 255, 0.12);
}

body.light-theme, .light-theme, [data-theme="light"] {
    --bg-page: #f8fafc;
    --bg-mesh-1: rgba(99, 102, 241, 0.08);
    --bg-mesh-2: rgba(6, 182, 212, 0.06);
    --bg-mesh-3: rgba(244, 63, 94, 0.04);
    --card-bg: rgba(255, 255, 255, 0.92);
    --card-border: rgba(0, 0, 0, 0.08);
    --card-shadow: 0 14px 34px -10px rgba(0, 0, 0, 0.08);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --hud-bg: rgba(255, 255, 255, 0.94);
    --hud-border: rgba(0, 0, 0, 0.10);
    --hud-text: #0f172a;
    --switcher-bg: rgba(226, 232, 240, 0.85);
    --switcher-btn-bg: transparent;
    --switcher-btn-text: #64748b;
    --switcher-active-bg: linear-gradient(135deg, #4f46e5, #0284c7);
    --switcher-active-text: #ffffff;
    --input-bg: #ffffff;
    --input-border: rgba(0, 0, 0, 0.15);
}

/* Global Canvas */
body, .gradio-container {
    background-color: var(--bg-page) !important;
    background-image: 
        radial-gradient(circle at 50% 0%, var(--bg-mesh-1) 0%, transparent 65%),
        radial-gradient(circle at 100% 40%, var(--bg-mesh-2) 0%, transparent 55%),
        radial-gradient(circle at 0% 80%, var(--bg-mesh-3) 0%, transparent 50%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    transition: background-color 0.25s ease, color 0.25s ease !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 20px 16px !important;
}

/* Studio Nav Header */
.studio-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    padding: 18px 26px;
    background: var(--card-bg) !important;
    backdrop-filter: blur(24px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 20px !important;
    margin-bottom: 22px !important;
    box-shadow: var(--card-shadow) !important;
    transition: all 0.25s ease !important;
}

.studio-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.studio-logo-glow {
    font-size: 30px;
    filter: drop-shadow(0 0 16px rgba(99, 102, 241, 0.7));
}

.studio-title {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a5b4fc 0%, #38bdf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}

body.light-theme .studio-title {
    background: linear-gradient(135deg, #4338ca 0%, #0284c7 50%, #7e22ce 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.studio-sub {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0;
    font-weight: 500;
}

.studio-badge {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.1em;
    padding: 3px 9px;
    border-radius: 6px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(6, 182, 212, 0.25));
    border: 1px solid rgba(99, 102, 241, 0.5);
    color: #38bdf8;
    text-transform: uppercase;
}

/* Theme Switcher Pill */
.theme-switcher-container {
    display: inline-flex;
    align-items: center;
    background: var(--switcher-bg);
    padding: 4px;
    border-radius: 30px;
    border: 1px solid var(--card-border);
    gap: 2px;
}

.theme-pill-btn {
    border: none;
    background: transparent;
    color: var(--switcher-btn-text);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.theme-pill-btn:hover {
    color: var(--text-primary);
}

.theme-pill-btn.active {
    background: var(--switcher-active-bg) !important;
    color: var(--switcher-active-text) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
}

/* Studio Status Pill */
.studio-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 20px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
}

body.light-theme .studio-status-pill {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(5, 150, 105, 0.4);
    color: #059669;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulseGlow 2s infinite ease-in-out;
}

@keyframes pulseGlow {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

/* Glassmorphic Cards */
.studio-card {
    background: var(--card-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin-bottom: 18px !important;
    box-shadow: var(--card-shadow) !important;
    transition: all 0.25s ease !important;
}

/* Action Button */
.btn-primary-glow {
    background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: -0.01em !important;
    padding: 14px 30px !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 26px -4px rgba(99, 102, 241, 0.55) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.btn-primary-glow:hover {
    box-shadow: 0 8px 32px rgba(6, 182, 212, 0.70) !important;
    transform: translateY(-2px) !important;
}

/* HUD Telemetry Badges */
.hud-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px;
    margin: 14px 0;
}

.hud-card {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 16px;
    background: var(--hud-bg) !important;
    border: 1px solid var(--hud-border) !important;
    border-radius: 14px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: var(--hud-text) !important;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
    transition: all 0.25s ease;
}

.hud-card strong {
    color: #38bdf8 !important;
}

body.light-theme .hud-card strong {
    color: #0284c7 !important;
}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

THEME_HEAD_JS = """
<script>
window.__setReviveTheme = function(mode) {
    try {
        localStorage.setItem('revive_theme_preference', mode);
        var effectiveMode = mode;
        if (mode === 'system') {
            var isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            effectiveMode = isDark ? 'dark' : 'light';
        }
        
        var body = document.body;
        var docEl = document.documentElement;
        var containers = document.querySelectorAll('.gradio-container');
        
        if (effectiveMode === 'dark') {
            body.classList.add('dark', 'dark-theme');
            body.classList.remove('light-theme');
            docEl.setAttribute('data-theme', 'dark');
            containers.forEach(function(c) { c.classList.add('dark'); c.classList.remove('light-theme'); });
        } else {
            body.classList.remove('dark', 'dark-theme');
            body.classList.add('light-theme');
            docEl.setAttribute('data-theme', 'light');
            containers.forEach(function(c) { c.classList.remove('dark'); c.classList.add('light-theme'); });
        }
        
        ['light', 'dark', 'system'].forEach(function(id) {
            var btn = document.getElementById('theme-btn-' + id);
            if (btn) {
                if (id === mode) btn.classList.add('active');
                else btn.classList.remove('active');
            }
        });
    } catch(e) {
        console.warn('Theme switch error:', e);
    }
};

// Initialize theme on load
document.addEventListener('DOMContentLoaded', function() {
    var saved = localStorage.getItem('revive_theme_preference') || 'system';
    window.__setReviveTheme(saved);
    
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
            var current = localStorage.getItem('revive_theme_preference') || 'system';
            if (current === 'system') {
                window.__setReviveTheme('system');
            }
        });
    }
});
</script>
"""



# ---------------------------------------------------------------------------
# High-Speed Master Relive & Restoration Function
# ---------------------------------------------------------------------------

def relive_photo(
    input_image: np.ndarray | None,
    relive_mode: str,
    defect_repair: float,
    detail_enhancement: float,
    defade_strength: float,
    clahe_strength: float,
    temperature: float,
    tint: float,
    vibrance: float,
    saturation: float,
):
    """
    Executes deep archival defect healing, texture synthesis, and neural reliving.
    """
    if input_image is None:
        empty_hud = "<div class='hud-card'><strong>Status:</strong> Awaiting image upload...</div>"
        return (
            None, None, None, None, None,
            gr.update(value=empty_hud),
            gr.update(value=empty_hud),
            gr.update(value=empty_hud),
            gr.update(value=empty_hud),
            gr.update(value=empty_hud),
            gr.update(visible=False, value=None),
        )

    # Resolve model choice from relive mode
    cfg = RELIVE_MODES.get(relive_mode, RELIVE_MODES["🌟 Master Archival Revival"])
    model_choice = cfg["model_choice"]
    ensemble_blend = cfg.get("ensemble_blend", 0.65)

    # Execute deep restoration and reliving pipeline
    res = studio_engine.restore_and_relive(
        input_image=input_image,
        relive_mode=relive_mode,
        defect_repair_strength=defect_repair,
        detail_enhancement=detail_enhancement,
        defade_strength=defade_strength,
        colorization_model=model_choice,
        resolution=256,
        ensemble_blend=ensemble_blend,
        clahe_strength=clahe_strength,
        temperature=temperature,
        tint=tint,
        vibrance=vibrance,
        saturation=saturation,
    )

    if not res:
        empty_hud = "<div class='hud-card'><strong>Status:</strong> Processing error</div>"
        return (
            None, None, None, None, None,
            gr.update(value=empty_hud),
            gr.update(value=empty_hud),
            gr.update(value=empty_hud),
            gr.update(value=empty_hud),
            gr.update(value=empty_hud),
            gr.update(visible=False, value=None),
        )

    orig_rgb = res["original_damaged_rgb"]
    final_rgb = res["final_masterpiece_rgb"]
    damage_map_rgb = res["damage_map_rgb"]
    heatmap_rgb = res["heatmap_rgb"]

    # Save lossless PNG master
    export_file = export_image(final_rgb, format_type="PNG")

    # Format telemetry badges
    res_str = res["resolution_str"]
    lat_ms = res["latency_ms"]
    is_cached = res.get("from_cache", False)
    defect_pct = res["defect_pct"]
    damage_count = res.get("damaged_pixel_count", 0)
    sharp_gain = res["sharpness_gain"]

    lat_badge = (
        f"{lat_ms} ms <span style='font-size:11px; color:#38bdf8;'>(⚡ Instant Cache)</span>"
        if is_cached
        else f"{round(lat_ms/1000, 2)} s <span style='font-size:11px; color:#34d399;'>(🚀 Neural Speed)</span>"
    )

    hud_res_val = f"<div class='hud-card'>📐 <strong>Resolution:</strong> {res_str}</div>"
    hud_lat_val = f"<div class='hud-card'>⚡ <strong>Latency:</strong> {lat_badge}</div>"
    hud_defects_val = f"<div class='hud-card'>🩹 <strong>Defects Healed:</strong> {defect_pct}% <span style='font-size:11px; color:#38bdf8;'>({damage_count:,} px)</span></div>"
    hud_sharp_val = f"<div class='hud-card'>🔍 <strong>Clarity Gain:</strong> +{sharp_gain}% <span style='font-size:11px; color:#34d399;'>(Ultra-HD)</span></div>"
    hud_col_val = f"<div class='hud-card'>🌈 <strong>Colorfulness:</strong> {res['colorfulness_score']} <span style='font-size:11px; color:#a855f7;'>({res['colorfulness_label']})</span></div>"

    return (
        (orig_rgb, final_rgb),    # Split Slider: Original Damaged vs. Relived Masterpiece
        orig_rgb,                 # Side-by-Side: Original
        final_rgb,                # Side-by-Side: Relived
        damage_map_rgb,           # Diagnostic Tab: Repaired Defect Map
        heatmap_rgb,              # Diagnostic Tab: Chromatic Turbo Heatmap
        gr.update(value=hud_res_val),
        gr.update(value=hud_lat_val),
        gr.update(value=hud_defects_val),
        gr.update(value=hud_sharp_val),
        gr.update(value=hud_col_val),
        gr.update(visible=True, value=export_file),
    )


def apply_mode_defaults(mode_name: str):
    """Updates studio sliders when the user picks a preset mode."""
    cfg = RELIVE_MODES.get(mode_name, RELIVE_MODES["🌟 Master Archival Revival"])
    return (
        cfg["defect_repair"],
        cfg["detail_enhancement"],
        cfg["defade_strength"],
        cfg["clahe"],
        cfg["temperature"],
        cfg["tint"],
        cfg["vibrance"],
        cfg["saturation"],
    )


# ---------------------------------------------------------------------------
# Studio Layout Definition
# ---------------------------------------------------------------------------

with gr.Blocks(title="ReviveGAN Studio Pro — AI Archival Restoration") as demo:

    # 1. Top Creative Studio Navigation Bar & Theme Switcher
    gr.HTML(
        """
        <div class="studio-navbar">
            <div class="studio-brand">
                <span class="studio-logo-glow">✨</span>
                <div>
                    <h1 class="studio-title">ReviveGAN Studio Pro</h1>
                    <p class="studio-sub">Deep Generative Archival Photo Restoration & Neural Relive Suite</p>
                </div>
                <span class="studio-badge">V3.0 ARCHIVAL AI</span>
            </div>

            <!-- Segmented Theme Switcher -->
            <div class="theme-switcher-container">
                <button type="button" id="theme-btn-light" class="theme-pill-btn" onclick="window.__setReviveTheme('light')">
                    <span>☀️</span> Light
                </button>
                <button type="button" id="theme-btn-dark" class="theme-pill-btn active" onclick="window.__setReviveTheme('dark')">
                    <span>🌙</span> Dark
                </button>
                <button type="button" id="theme-btn-system" class="theme-pill-btn" onclick="window.__setReviveTheme('system')">
                    <span>💻</span> System
                </button>
            </div>

            <div class="studio-status-pill">
                <span class="pulse-dot"></span>
                <span>Neural Defect Inpainter & Colorizer Active</span>
            </div>
        </div>
        """
    )

    # 2. Upload Box & Relive Mode Selector
    with gr.Row(elem_classes=["studio-card"]):
        with gr.Column(scale=8):
            upload_input = gr.Image(
                label="📁 Upload Archival Photograph (Torn, Scratched, Worn, Faded, or Black & White)",
                type="numpy",
                sources=["upload", "clipboard"],
                height=260,
            )
        with gr.Column(scale=4):
            relive_mode_choice = gr.Dropdown(
                label="🔮 Generative Relive Mode",
                choices=list(RELIVE_MODES.keys()),
                value="🌟 Master Archival Revival",
                interactive=True,
            )
            mode_desc = gr.Markdown(
                "**Master Archival Revival:** Inpaint physical cracks, tears & scratches, synthesize 4K facial micro-textures, neutralize vintage fading, and infuse natural historical colors."
            )
            relive_btn = gr.Button(
                "✨ Relive & Restore Photograph",
                variant="primary",
                size="lg",
                elem_classes=["btn-primary-glow"],
            )

    # 3. Advanced Restoration & Grading Controls (Accordion)
    with gr.Accordion("⚙️ Advanced Restoration & Studio Controls (Fine-Tune Healing & Color)", open=False, elem_classes=["studio-card"]):
        with gr.Row():
            defect_repair = gr.Slider(0.0, 100.0, value=60.0, step=1.0, label="🩹 Defect & Tear Inpainting Strength")
            detail_enhancement = gr.Slider(0.0, 100.0, value=45.0, step=1.0, label="🔍 Micro-Texture & Facial Clarity Boost")
            defade_strength = gr.Slider(0.0, 100.0, value=40.0, step=1.0, label="⚖️ Archival De-Fading & Sepia Neutralize")
        with gr.Row():
            clahe_strength = gr.Slider(0.0, 50.0, value=20.0, step=1.0, label="🌓 CLAHE Dynamic Range Expansion")
            temperature = gr.Slider(-50.0, 50.0, value=2.0, step=1.0, label="🌡️ Color Temperature (Cool ↔ Warm)")
            tint = gr.Slider(-50.0, 50.0, value=0.0, step=1.0, label="🎨 Tint (Green ↔ Magenta)")
        with gr.Row():
            vibrance = gr.Slider(-50.0, 100.0, value=22.0, step=1.0, label="💎 Smart Vibrance (Skin Protected)")
            saturation = gr.Slider(0.0, 2.0, value=1.18, step=0.05, label="🌈 Color Saturation Multiplier")

    # 4. Executive Telemetry HUD Badges
    with gr.Row():
        hud_res = gr.HTML("<div class='hud-card'>📐 <strong>Resolution:</strong> —</div>")
        hud_lat = gr.HTML("<div class='hud-card'>⚡ <strong>Latency:</strong> —</div>")
        hud_defects = gr.HTML("<div class='hud-card'>🩹 <strong>Defects Healed:</strong> —</div>")
        hud_sharp = gr.HTML("<div class='hud-card'>🔍 <strong>Clarity Gain:</strong> —</div>")
        hud_col = gr.HTML("<div class='hud-card'>🌈 <strong>Colorfulness:</strong> —</div>")

    # 5. Visual Results: Split-Screen Slider, Side-by-Side, and Diagnostic Damage Map
    with gr.Tabs(elem_classes=["studio-card"]):
        with gr.Tab("🎚️ Split-Screen Interactive Slider"):
            slider_output = gr.ImageSlider(
                label="Before (Original Damaged Photograph) vs. After (Fully Restored & Relived Masterpiece) — Drag Divider",
                type="numpy",
                height=560,
                interactive=True,
            )

        with gr.Tab("🖼️ Side-by-Side Studio View"):
            with gr.Row():
                side_orig_output = gr.Image(
                    label="Original Worn / Torn Photograph",
                    type="numpy",
                    height=520,
                )
                side_relived_output = gr.Image(
                    label="Fully Restored & Relived Masterpiece",
                    type="numpy",
                    height=520,
                )

        with gr.Tab("🔍 Diagnostic Damage & Chromatic Inspector"):
            gr.Markdown(
                "**Technical Diagnostic View:** Inspect the precise algorithmic defect localization map (scratches, cracks, and tears detected and healed) alongside the CIELAB chromatic energy distribution."
            )
            with gr.Row():
                damage_map_output = gr.Image(
                    label="Defect & Tear Localization Map (Repaired Cracks Glowing in Cyan/Amber)",
                    type="numpy",
                    height=460,
                )
                heatmap_output = gr.Image(
                    label="CIELAB Chromatic Energy Distribution Heatmap",
                    type="numpy",
                    height=460,
                )

    # 6. One-Click Lossless PNG Download
    with gr.Row():
        download_btn = gr.DownloadButton(
            "⬇️ Download Fully Restored Masterpiece (Lossless PNG)",
            variant="primary",
            size="lg",
            visible=False,
            elem_classes=["btn-primary-glow"],
        )

    # 7. Curated Archival Gallery with Worn, Torn & Scratched Historical Photos
    gr.Markdown("### 🏛️ Archival Showcase Gallery (Authentic Worn & Torn Historical Photos — Instant < 1ms Preview)")
    examples = gr.Examples(
        examples=[
            ["assets/samples/vintage_torn_portrait.png", "🌟 Master Archival Revival"],
            ["assets/samples/creased_native_woman.png", "🌟 Master Archival Revival"],
            ["assets/samples/crazed_archival_boy.png", "💎 Ultra-HD Portrait Clarity"],
            ["assets/samples/damaged_archival_couple.png", "🌟 Master Archival Revival"],
            ["assets/samples/creased_1950s_gentleman.png", "🌟 Master Archival Revival"],
            ["assets/samples/torn_adams_landscape.jpg", "🏛️ Pristine Museum Monochrome"],
        ],
        inputs=[upload_input, relive_mode_choice],
        label="Click any historical archival photograph to test full restoration & reliving instantly",
        examples_per_page=6,
    )

    # Dynamic description update when mode changes
    def update_mode_description(mode_name: str) -> str:
        cfg = RELIVE_MODES.get(mode_name, RELIVE_MODES["🌟 Master Archival Revival"])
        return f"**{mode_name}:** {cfg['description']}"

    relive_mode_choice.change(
        fn=update_mode_description,
        inputs=[relive_mode_choice],
        outputs=[mode_desc],
    )

    # Update sliders when mode changes
    relive_mode_choice.change(
        fn=apply_mode_defaults,
        inputs=[relive_mode_choice],
        outputs=[
            defect_repair,
            detail_enhancement,
            defade_strength,
            clahe_strength,
            temperature,
            tint,
            vibrance,
            saturation,
        ],
    )

    # Wire Main Execution
    all_inputs = [
        upload_input,
        relive_mode_choice,
        defect_repair,
        detail_enhancement,
        defade_strength,
        clahe_strength,
        temperature,
        tint,
        vibrance,
        saturation,
    ]

    all_outputs = [
        slider_output,
        side_orig_output,
        side_relived_output,
        damage_map_output,
        heatmap_output,
        hud_res,
        hud_lat,
        hud_defects,
        hud_sharp,
        hud_col,
        download_btn,
    ]

    relive_btn.click(
        fn=relive_photo,
        inputs=all_inputs,
        outputs=all_outputs,
    )

    upload_input.change(
        fn=relive_photo,
        inputs=all_inputs,
        outputs=all_outputs,
    )


# ---------------------------------------------------------------------------
# Server Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import socket

    def _get_lan_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"

    def _find_free_port(preferred: int = 3003, max_tries: int = 10) -> int:
        for p in range(preferred, preferred + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", p))
                    return p
                except OSError:
                    continue
        return preferred

    desired_port = int(os.environ.get("PORT", 3003))
    port = _find_free_port(desired_port)
    lan_ip = _get_lan_ip()
    share = "--share" in sys.argv or os.environ.get("SHARE", "false").lower() in ("true", "1")

    print(f"\n=======================================================", flush=True)
    print(f"  ReviveGAN Studio Pro — Archival Restoration Engine   ", flush=True)
    print(f"=======================================================", flush=True)
    print(f"  -> Local Studio:   http://localhost:{port}", flush=True)
    print(f"  -> Network Studio: http://{lan_ip}:{port}\n", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        theme=THEME,
        css=CUSTOM_CSS,
        head=THEME_HEAD_JS,
        show_error=True,
    )
