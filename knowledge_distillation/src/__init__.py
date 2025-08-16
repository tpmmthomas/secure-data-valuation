"""Knowledge Distillation Framework.

A production-ready framework for neural network knowledge distillation.
"""

__version__ = "1.0.0"
__author__ = "Knowledge Distillation Team"

from .config import KnowledgeDistillationConfig, load_config, save_config
from .trainer import KnowledgeDistillationTrainer
from . import models, datasets, utils

__all__ = [
    'KnowledgeDistillationConfig',
    'load_config',
    'save_config', 
    'KnowledgeDistillationTrainer',
    'models',
    'datasets',
    'utils'
]
