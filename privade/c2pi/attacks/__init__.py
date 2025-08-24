"""
Inference Data Privacy Attacks (IDPAs) for C2PI
"""

from .dina import DINAAttack
from .utils import AttackResult, evaluate_attack_success

__all__ = [
    "DINAAttack",
    "AttackResult",
    "evaluate_attack_success"
]
