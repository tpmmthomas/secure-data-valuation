"""
C2PI: Crypto-Clear Two-Party Neural Network Private Inference

This package implements the boundary finding algorithm from the C2PI paper
for efficient neural network split inference with privacy guarantees.
"""

from .boundary_finder import BoundaryFinder
from .config import C2PIConfig, DINAConfig, ExperimentConfig, create_default_config

__version__ = "1.0.0"
__author__ = "C2PI Implementation Team"

__all__ = [
    "BoundaryFinder",
    "C2PIConfig", 
    "DINAConfig",
    "ExperimentConfig",
    "create_default_config"
]
