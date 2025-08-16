"""Teacher model implementations."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any
from .base import TeacherModel


class ResNetTeacher(TeacherModel):
    """ResNet-based teacher models."""
    
    SUPPORTED_VARIANTS = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    
    def _initialize_model(self, variant: str = 'resnet18', **kwargs):
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        
        if self.pretrained:
            try:
                # Try new torchvision API first
                if variant == 'resnet18':
                    weights = models.ResNet18_Weights.IMAGENET1K_V1
                elif variant == 'resnet34':
                    weights = models.ResNet34_Weights.IMAGENET1K_V1
                elif variant == 'resnet50':
                    weights = models.ResNet50_Weights.IMAGENET1K_V1
                elif variant == 'resnet101':
                    weights = models.ResNet101_Weights.IMAGENET1K_V1
                elif variant == 'resnet152':
                    weights = models.ResNet152_Weights.IMAGENET1K_V1
                self.model = self.SUPPORTED_VARIANTS[variant](weights=weights)
            except AttributeError:
                # Fallback to old API
                self.model = self.SUPPORTED_VARIANTS[variant](pretrained=True)
        else:
            self.model = self.SUPPORTED_VARIANTS[variant](pretrained=False)
        
        # Replace final layer for target dataset
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract feature maps from different layers."""
        features = {}
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        features['conv1'] = x
        
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        features['layer1'] = x
        
        x = self.model.layer2(x)
        features['layer2'] = x
        
        x = self.model.layer3(x)
        features['layer3'] = x
        
        x = self.model.layer4(x)
        features['layer4'] = x
        
        return features
    
    def freeze_features(self):
        """Freeze all layers except the final classifier."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        for param in self.model.fc.parameters():
            param.requires_grad = True


class VGGTeacher(TeacherModel):
    """VGG-based teacher models."""
    
    SUPPORTED_VARIANTS = {
        'vgg11': models.vgg11,
        'vgg11_bn': models.vgg11_bn,
        'vgg13': models.vgg13,
        'vgg13_bn': models.vgg13_bn,
        'vgg16': models.vgg16,
        'vgg16_bn': models.vgg16_bn,
        'vgg19': models.vgg19,
        'vgg19_bn': models.vgg19_bn,
    }
    
    def _initialize_model(self, variant: str = 'vgg16', **kwargs):
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported VGG variant: {variant}")
        
        if self.pretrained:
            try:
                # Try new torchvision API
                if variant == 'vgg16':
                    weights = models.VGG16_Weights.IMAGENET1K_V1
                    self.model = models.vgg16(weights=weights)
                else:
                    # Fallback to old API for other variants
                    self.model = self.SUPPORTED_VARIANTS[variant](pretrained=True)
            except AttributeError:
                self.model = self.SUPPORTED_VARIANTS[variant](pretrained=True)
        else:
            self.model = self.SUPPORTED_VARIANTS[variant](pretrained=False)
        
        # Replace final layer
        self.model.classifier[6] = nn.Linear(4096, self.num_classes)
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def freeze_features(self):
        """Freeze feature layers."""
        for param in self.model.features.parameters():
            param.requires_grad = False


class DenseNetTeacher(TeacherModel):
    """DenseNet-based teacher models."""
    
    SUPPORTED_VARIANTS = {
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'densenet161': models.densenet161,
    }
    
    def _initialize_model(self, variant: str = 'densenet121', **kwargs):
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported DenseNet variant: {variant}")
        
        if self.pretrained:
            try:
                if variant == 'densenet121':
                    weights = models.DenseNet121_Weights.IMAGENET1K_V1
                    self.model = models.densenet121(weights=weights)
                else:
                    self.model = self.SUPPORTED_VARIANTS[variant](pretrained=True)
            except AttributeError:
                self.model = self.SUPPORTED_VARIANTS[variant](pretrained=True)
        else:
            self.model = self.SUPPORTED_VARIANTS[variant](pretrained=False)
        
        # Replace classifier
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EfficientNetTeacher(TeacherModel):
    """EfficientNet-based teacher models."""
    
    def _initialize_model(self, variant: str = 'efficientnet_b0', **kwargs):
        try:
            # Try to use timm if available
            import timm
            self.model = timm.create_model(variant, pretrained=self.pretrained, num_classes=self.num_classes)
        except ImportError:
            # Fallback to torchvision (limited EfficientNet support)
            if self.pretrained:
                try:
                    if variant == 'efficientnet_b0':
                        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
                        self.model = models.efficientnet_b0(weights=weights)
                    else:
                        raise ValueError(f"Pretrained {variant} not available in torchvision. Install timm for more variants.")
                except AttributeError:
                    self.model = models.efficientnet_b0(pretrained=True)
            else:
                self.model = models.efficientnet_b0(pretrained=False)
            
            # Replace classifier
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Model registry for easy instantiation
TEACHER_MODELS = {
    'resnet18': lambda num_classes, **kwargs: ResNetTeacher(num_classes, variant='resnet18', **kwargs),
    'resnet34': lambda num_classes, **kwargs: ResNetTeacher(num_classes, variant='resnet34', **kwargs),
    'resnet50': lambda num_classes, **kwargs: ResNetTeacher(num_classes, variant='resnet50', **kwargs),
    'resnet101': lambda num_classes, **kwargs: ResNetTeacher(num_classes, variant='resnet101', **kwargs),
    'resnet152': lambda num_classes, **kwargs: ResNetTeacher(num_classes, variant='resnet152', **kwargs),
    
    'vgg11': lambda num_classes, **kwargs: VGGTeacher(num_classes, variant='vgg11', **kwargs),
    'vgg11_bn': lambda num_classes, **kwargs: VGGTeacher(num_classes, variant='vgg11_bn', **kwargs),
    'vgg13': lambda num_classes, **kwargs: VGGTeacher(num_classes, variant='vgg13', **kwargs),
    'vgg13_bn': lambda num_classes, **kwargs: VGGTeacher(num_classes, variant='vgg13_bn', **kwargs),
    'vgg16': lambda num_classes, **kwargs: VGGTeacher(num_classes, variant='vgg16', **kwargs),
    'vgg16_bn': lambda num_classes, **kwargs: VGGTeacher(num_classes, variant='vgg16_bn', **kwargs),
    'vgg19': lambda num_classes, **kwargs: VGGTeacher(num_classes, variant='vgg19', **kwargs),
    'vgg19_bn': lambda num_classes, **kwargs: VGGTeacher(num_classes, variant='vgg19_bn', **kwargs),
    
    'densenet121': lambda num_classes, **kwargs: DenseNetTeacher(num_classes, variant='densenet121', **kwargs),
    'densenet169': lambda num_classes, **kwargs: DenseNetTeacher(num_classes, variant='densenet169', **kwargs),
    'densenet201': lambda num_classes, **kwargs: DenseNetTeacher(num_classes, variant='densenet201', **kwargs),
    'densenet161': lambda num_classes, **kwargs: DenseNetTeacher(num_classes, variant='densenet161', **kwargs),
    
    'efficientnet_b0': lambda num_classes, **kwargs: EfficientNetTeacher(num_classes, variant='efficientnet_b0', **kwargs),
    'efficientnet_b1': lambda num_classes, **kwargs: EfficientNetTeacher(num_classes, variant='efficientnet_b1', **kwargs),
    'efficientnet_b2': lambda num_classes, **kwargs: EfficientNetTeacher(num_classes, variant='efficientnet_b2', **kwargs),
    'efficientnet_b3': lambda num_classes, **kwargs: EfficientNetTeacher(num_classes, variant='efficientnet_b3', **kwargs),
}


def create_teacher_model(model_name: str, num_classes: int, **kwargs) -> TeacherModel:
    """Factory function to create teacher models."""
    if model_name not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher model: {model_name}. Available models: {list(TEACHER_MODELS.keys())}")
    
    return TEACHER_MODELS[model_name](num_classes, **kwargs)
