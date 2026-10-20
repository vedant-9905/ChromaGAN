"""
ReviveGAN Core Package (formerly ChromaGAN)
Deep Neural Archival Photo Restoration & Generative Relive Suite
"""
from core.models import UNetColorGenerator, PatchGANColorDiscriminator
from core.losses import ChromaGANLoss
from core.inference import run_colorization, build_generator, compute_colorfulness

from core.pipeline import (
    studio_engine,
    ReviveGANEngine,
    ChromaGANEngine,
    RELIVE_MODES,
    PRESETS,
    export_image,
)
from core.restoration import (
    detect_scratches_and_defects,
    inpaint_scratches,
    restore_high_frequency_details,
    defade_and_neutralize_sepia,
    generate_damage_diagnostic_map,
)
from core.dataset import ColorizationDataset, get_colorization_dataloader

__all__ = [
    "UNetColorGenerator",
    "PatchGANColorDiscriminator",
    "ChromaGANLoss",
    "run_colorization",
    "build_generator",
    "compute_colorfulness",
    "studio_engine",
    "ReviveGANEngine",
    "ChromaGANEngine",
    "RELIVE_MODES",
    "PRESETS",
    "export_image",
    "detect_scratches_and_defects",
    "inpaint_scratches",
    "restore_high_frequency_details",
    "defade_and_neutralize_sepia",
    "generate_damage_diagnostic_map",
    "ColorizationDataset",
    "get_colorization_dataloader",
]
