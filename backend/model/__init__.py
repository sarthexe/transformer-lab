"""Model definition: config, hook-site namespace, and the instrumented GPT."""

from .config import ModelConfig
from .sites import layer_site, site_names, site_shape
from .transformer import NULL_RECORDER, Recorder, ToyGPT

__all__ = [
    "ModelConfig",
    "NULL_RECORDER",
    "Recorder",
    "ToyGPT",
    "layer_site",
    "site_names",
    "site_shape",
]
