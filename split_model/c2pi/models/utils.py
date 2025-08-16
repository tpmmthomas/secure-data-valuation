"""
Model utilities for loading and configuring models.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Load and configure model for C2PI experiments."""
    
    model_name = model_name.lower()
    
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        # Replace classifier for target dataset
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


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
