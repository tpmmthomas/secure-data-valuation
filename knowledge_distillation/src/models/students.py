"""Student model implementations."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any
from .base import StudentModel


class CNN3Student(StudentModel):
    """Lightweight 3-layer CNN student model."""
    
    def _initialize_model(self, dropout: float = 0.0, **kwargs):
        self.dropout = dropout
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNN5Student(StudentModel):
    """Lightweight 5-layer CNN student model (same as original)."""
    
    def _initialize_model(self, dropout: float = 0.0, **kwargs):
        self.dropout = dropout
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4

            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract feature maps for distillation."""
        features = {}
        
        # Process in blocks to extract intermediate features
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                features[f'block_{len(features)}'] = x
        
        features['final_conv'] = x
        return features


class CNN7Student(StudentModel):
    """Medium-sized 7-layer CNN student model."""
    
    def _initialize_model(self, dropout: float = 0.0, **kwargs):
        self.dropout = dropout
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class MobileNetV2Student(StudentModel):
    """MobileNetV2-based student model."""
    
    def _initialize_model(self, width_mult: float = 1.0, **kwargs):
        try:
            # Use torchvision MobileNetV2
            self.model = models.mobilenet_v2(pretrained=False, width_mult=width_mult)
            # Replace classifier
            self.model.classifier[1] = nn.Linear(self.model.last_channel, self.num_classes)
        except TypeError:
            # Older torchvision version
            self.model = models.mobilenet_v2(pretrained=False)
            # Replace classifier
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
        
        self.width_mult = width_mult
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MobileNetV3Student(StudentModel):
    """MobileNetV3-based student model."""
    
    def _initialize_model(self, variant: str = 'small', **kwargs):
        if variant == 'small':
            self.model = models.mobilenet_v3_small(pretrained=False)
        elif variant == 'large':
            self.model = models.mobilenet_v3_large(pretrained=False)
        else:
            raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")
        
        # Replace classifier
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, self.num_classes)
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ShuffleNetV2Student(StudentModel):
    """ShuffleNetV2-based student model."""
    
    def _initialize_model(self, variant: str = 'x1_0', **kwargs):
        if variant == 'x0_5':
            self.model = models.shufflenet_v2_x0_5(pretrained=False)
        elif variant == 'x1_0':
            self.model = models.shufflenet_v2_x1_0(pretrained=False)
        elif variant == 'x1_5':
            self.model = models.shufflenet_v2_x1_5(pretrained=False)
        elif variant == 'x2_0':
            self.model = models.shufflenet_v2_x2_0(pretrained=False)
        else:
            raise ValueError(f"Unsupported ShuffleNetV2 variant: {variant}")
        
        # Replace classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SqueezeNetStudent(StudentModel):
    """SqueezeNet-based student model."""
    
    def _initialize_model(self, version: str = '1_0', **kwargs):
        if version == '1_0':
            self.model = models.squeezenet1_0(pretrained=False)
        elif version == '1_1':
            self.model = models.squeezenet1_1(pretrained=False)
        else:
            raise ValueError(f"Unsupported SqueezeNet version: {version}")
        
        # Replace classifier
        self.model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.model.num_classes = self.num_classes
        self.version = version
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Model registry for easy instantiation
STUDENT_MODELS = {
    'cnn3': CNN3Student,
    'cnn5': CNN5Student,
    'cnn7': CNN7Student,
    
    'mobilenet_v2': lambda num_classes, **kwargs: MobileNetV2Student(num_classes, **kwargs),
    'mobilenet_v2_0.5': lambda num_classes, **kwargs: MobileNetV2Student(num_classes, width_mult=0.5, **kwargs),
    'mobilenet_v2_0.75': lambda num_classes, **kwargs: MobileNetV2Student(num_classes, width_mult=0.75, **kwargs),
    
    'mobilenet_v3_small': lambda num_classes, **kwargs: MobileNetV3Student(num_classes, variant='small', **kwargs),
    'mobilenet_v3_large': lambda num_classes, **kwargs: MobileNetV3Student(num_classes, variant='large', **kwargs),
    
    'shufflenet_v2_x0.5': lambda num_classes, **kwargs: ShuffleNetV2Student(num_classes, variant='x0_5', **kwargs),
    'shufflenet_v2_x1.0': lambda num_classes, **kwargs: ShuffleNetV2Student(num_classes, variant='x1_0', **kwargs),
    'shufflenet_v2_x1.5': lambda num_classes, **kwargs: ShuffleNetV2Student(num_classes, variant='x1_5', **kwargs),
    'shufflenet_v2_x2.0': lambda num_classes, **kwargs: ShuffleNetV2Student(num_classes, variant='x2_0', **kwargs),
    
    'squeezenet1_0': lambda num_classes, **kwargs: SqueezeNetStudent(num_classes, version='1_0', **kwargs),
    'squeezenet1_1': lambda num_classes, **kwargs: SqueezeNetStudent(num_classes, version='1_1', **kwargs),
}


def create_student_model(model_name: str, num_classes: int, **kwargs) -> StudentModel:
    """Factory function to create student models."""
    if model_name not in STUDENT_MODELS:
        raise ValueError(f"Unknown student model: {model_name}. Available models: {list(STUDENT_MODELS.keys())}")
    
    return STUDENT_MODELS[model_name](num_classes, **kwargs)
