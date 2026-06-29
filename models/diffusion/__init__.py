"""Discrete-diffusion expression generator (config-selectable A/B vs the transformer).

See docs/superpowers/specs/2026-06-29-diffusion-generator-design.md
"""
from .grammar import Grammar
from .denoiser import MaskedDiffusionDenoiser
from .sampler import diffusion_sample
from .controller import DiffusionController

__all__ = ["Grammar", "MaskedDiffusionDenoiser", "diffusion_sample", "DiffusionController"]
