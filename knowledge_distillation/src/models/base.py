"""Base classes for models in the knowledge distillation framework."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseModel(nn.Module, ABC):
    """Base class for all models in the framework."""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self._initialize_model(**kwargs)
    
    @abstractmethod
    def _initialize_model(self, **kwargs):
        """Initialize the model architecture."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate feature maps for feature distillation."""
        return {}
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention maps for attention distillation."""
        return {}
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "num_classes": self.num_classes,
        }


class TeacherModel(BaseModel):
    """Base class for teacher models."""
    
    def __init__(self, num_classes: int, pretrained: bool = True, **kwargs):
        self.pretrained = pretrained
        super().__init__(num_classes, **kwargs)
    
    def freeze_features(self):
        """Freeze feature extraction layers."""
        pass
    
    def unfreeze_features(self):
        """Unfreeze feature extraction layers."""
        pass


class StudentModel(BaseModel):
    """Base class for student models."""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)
    
    def match_teacher_output_size(self, teacher_output_size: Tuple[int, ...]):
        """Adjust model to match teacher output dimensions if needed."""
        pass
