"""
Model utilities for loading and configuring models.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Load and configure model for C2PI experiments.
    
    Supported models:
    - vgg11, vgg16, vgg19: VGG architectures with ImageNet pretraining
    - resnet18, resnet50: ResNet architectures with ImageNet pretraining  
    - alexnet: AlexNet with ImageNet pretraining
    - cnn5: Simple 5-layer CNN with random initialization
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (ignored for cnn5)
        
    Returns:
        Configured PyTorch model
    """
    
    model_name = model_name.lower()
    
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        # Replace classifier for target dataset
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == "vgg11":
        model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == "cnn5":
        # 5-layer CNN with randomly initialized weights
        model = CNN5(num_classes=num_classes)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: vgg11, vgg16, vgg19, resnet18, resnet50, alexnet, cnn5")
    
    return model


class CNN5(nn.Module):
    """
    Simple 5-layer CNN model for experiments.
    Architecture: Conv -> ReLU -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> Conv -> ReLU -> MaxPool -> FC
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(CNN5, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Layer 1: Conv + ReLU
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 2: Conv + ReLU + MaxPool
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3: Conv + ReLU
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv + ReLU + MaxPool
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 5: Conv + ReLU + AdaptiveAvgPool
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def get_candidate_layers(model: nn.Module, only_conv_relu: bool = True) -> List[int]:
    """
    Get candidate layer indices for boundary search.
    
    Args:
        model: Neural network model
        only_conv_relu: If True, only consider Conv2d and ReLU layers
    
    Returns:
        List of layer indices in reverse order (tail to head)
    """
    if not hasattr(model, 'features'):
        raise NotImplementedError("Currently only supports models with .features attribute")
    
    candidates = []
    
    for i, layer in enumerate(model.features):
        if only_conv_relu:
            if isinstance(layer, (nn.Conv2d, nn.ReLU)):
                candidates.append(i)
        else:
            # Include all layers except pooling (which don't change semantics much)
            if not isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                candidates.append(i)
    
    # Return in reverse order for Algorithm 1 (scan tail to head)
    return list(reversed(candidates))


def get_model_info(model: nn.Module) -> dict:
    """Get detailed information about model architecture."""
    info = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }
    
    if hasattr(model, 'features'):
        info["num_feature_layers"] = len(model.features)
        info["feature_layer_types"] = [type(layer).__name__ for layer in model.features]
    
    if hasattr(model, 'classifier'):
        info["num_classifier_layers"] = len(model.classifier)
        info["classifier_layer_types"] = [type(layer).__name__ for layer in model.classifier]
    
    return info


def print_model_structure(model: nn.Module, max_layers: int = 20):
    """Print model structure for debugging."""
    print("Model Structure:")
    print("=" * 50)
    
    if hasattr(model, 'features'):
        print("Features:")
        for i, layer in enumerate(model.features):
            if i >= max_layers:
                print(f"  ... ({len(model.features) - max_layers} more layers)")
                break
            print(f"  {i:2d}: {layer}")
    
    if hasattr(model, 'classifier'):
        print("\nClassifier:")
        for i, layer in enumerate(model.classifier):
            print(f"  {i}: {layer}")
    
    print("=" * 50)


def freeze_layers(model: nn.Module, freeze_until: int):
    """Freeze model layers up to specified index."""
    if hasattr(model, 'features'):
        for i, layer in enumerate(model.features):
            if i <= freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False


def get_layer_by_index(model: nn.Module, layer_idx: int) -> nn.Module:
    """Get specific layer by index."""
    if hasattr(model, 'features'):
        if 0 <= layer_idx < len(model.features):
            return model.features[layer_idx]
    
    raise IndexError(f"Layer index {layer_idx} out of range")


def calculate_computational_cost(model: nn.Module, input_shape: tuple) -> dict:
    """Estimate computational cost (FLOPs) for model layers."""
    # This is a simplified estimation - for production use pytorch-OpCounter or similar
    
    def conv2d_flops(layer, input_size):
        """Estimate FLOPs for Conv2d layer."""
        kernel_flops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
        output_elements = input_size[1] * input_size[2] * layer.out_channels
        return kernel_flops * output_elements
    
    def linear_flops(layer):
        """Estimate FLOPs for Linear layer."""
        return layer.in_features * layer.out_features
    
    total_flops = 0
    layer_flops = []
    
    # Dummy forward pass to track shapes
    dummy_input = torch.randn(1, *input_shape)
    x = dummy_input
    
    if hasattr(model, 'features'):
        for i, layer in enumerate(model.features):
            input_size = x.shape[2:]  # H, W
            
            if isinstance(layer, nn.Conv2d):
                flops = conv2d_flops(layer, x.shape)
                layer_flops.append((i, type(layer).__name__, flops))
                total_flops += flops
            
            x = layer(x)
    
    return {
        "total_flops": total_flops,
        "layer_flops": layer_flops,
        "flops_per_layer": {i: flops for i, _, flops in layer_flops}
    }
