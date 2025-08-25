"""
Model definitions and loading utilities for PrivaDE experiments.

Supports LeNet-5, ResNet-20, VGG-8, and ResNet-50 with dataset-specific configurations.
All models use nn.Sequential where possible for better compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import os


class LeNet5(nn.Module):
    """
    LeNet-5 architecture for MNIST and CIFAR datasets.
    Modified to handle different input sizes and channel numbers.
    Uses nn.Sequential for all layers.
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1, input_size: int = 32):
        super(LeNet5, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        
        # Calculate the size after convolutions and pooling
        # After conv1 + pool1: size -> size/2
        # After conv2 + pool2: (size/2 - 4)/2
        conv_output_size = ((input_size // 2 - 4) // 2)
        fc_input_size = 16 * conv_output_size * conv_output_size
        
        # Define all layers using nn.Sequential
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class Square(nn.Module):
    """Elementwise square activation: f(x) = x^2"""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x) if self.inplace else x * x

class LeNetXS(nn.Module):
    """
    LeNet Extra Small (LeNetXS) - A minimal CNN for MNIST.
    Architecture: 28x28 → (5x5 conv) → 24x24 → 2x2 pool → 12x12
                  → (5x5 conv) → 8x8 → 2x2 pool → 4x4 → FC(32) → 10
    Total parameters: ~4,000 (much smaller than LeNet-5)
    Uses nn.Sequential for all layers.
    """
    
    def __init__(self, num_classes: int = 10):
        super(LeNetXS, self).__init__()
        
        # Define all layers using nn.Sequential
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, 5),     # 3*(1*5*5+1) = 78 parameters
            Square(),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(3*24*24,900),
            nn.ReLU(),
            nn.Linear(900, 32),   # 96*32+32 = 3,104 parameters
            nn.ReLU(),
            nn.Linear(32, num_classes)  # 32*10+10 = 330 parameters (for 10 classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SimpleCNN(nn.Module):
    """
    Simple CNN using only nn.Sequential layers.
    Ideal for testing and compatibility with inference frameworks.
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1, input_size: int = 28):
        super(SimpleCNN, self).__init__()
        
        # Calculate dimensions through the network
        # Conv1: 28x28 -> 24x24 (5x5 conv, no padding)
        # Pool1: 24x24 -> 12x12 (2x2 pool)
        # Conv2: 12x12 -> 8x8 (5x5 conv, no padding)  
        # Pool2: 8x8 -> 4x4 (2x2 pool)
        
        self.model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 6, kernel_size=5),  # 28->24 for MNIST
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 24->12
            
            # Second convolutional block  
            nn.Conv2d(6, 16, kernel_size=5),              # 12->8
            nn.ReLU(), 
            nn.MaxPool2d(2),                              # 8->4
            
            # Flatten and classify
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 84),                    # 4x4 feature maps
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)


class BasicBlock(nn.Module):
    """Basic block for ResNet-20."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    """ResNet-20 for CIFAR datasets."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class VGG8(nn.Module):
    """VGG-8 architecture for CIFAR and ImageNet datasets."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3, input_size: int = 32):
        super(VGG8, self).__init__()
        
        self.input_size = input_size
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate classifier input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_size, input_size)
            features_output = self.features(dummy_input)
            classifier_input_size = features_output.numel()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class VGG16(nn.Module):
    """VGG-16 architecture for CIFAR and ImageNet datasets."""
    
    def __init__(self, num_classes: int = 1000, input_channels: int = 3, input_size: int = 224):
        super(VGG16, self).__init__()
        
        self.input_size = input_size
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate classifier input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_size, input_size)
            features_output = self.features(dummy_input)
            classifier_input_size = features_output.numel()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50."""
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet50(nn.Module):
    """ResNet-50 for CIFAR and ImageNet datasets."""
    
    def __init__(self, num_classes: int = 1000, input_channels: int = 3, input_size: int = 224):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        self.input_size = input_size
        
        # Adjust first layer based on input size
        if input_size <= 32:  # CIFAR datasets
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.pool_size = 4
        else:  # ImageNet
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.pool_size = 7
        
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.linear = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Add maxpool for ImageNet-sized inputs
        if self.input_size > 32:
            out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class InvertedResidual(nn.Module):
    """Inverted Residual Block for MobileNetV2."""
    
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 architecture for CIFAR and ImageNet datasets."""
    
    def __init__(self, num_classes: int = 1000, input_channels: int = 3, input_size: int = 224, width_mult: float = 1.0):
        super(MobileNetV2, self).__init__()
        
        self.input_size = input_size
        
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        
        self.features = [nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        
        # building last several layers
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# Dataset configurations
DATASET_CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 28,
        'mean': [0.1307],
        'std': [0.3081]
    },
    'cifar10': {
        'num_classes': 10,
        'input_channels': 3,
        'input_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010]
    },
    'cifar100': {
        'num_classes': 100,
        'input_channels': 3,
        'input_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    },
    'imagenet': {
        'num_classes': 1000,
        'input_channels': 3,
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


def get_model(model_name: str, dataset: str, pretrained: bool = False, 
              pretrained_path: Optional[str] = None) -> nn.Module:
    """
    Get a model configured for the specified dataset.
    
    Args:
        model_name: Name of the model ('lenet5', 'lenetxs', 'simplecnn', 'resnet20', 'vgg8', 'vgg16', 'resnet50', 'mobilenetv2')
        dataset: Target dataset ('mnist', 'cifar10', 'cifar100', 'imagenet')
        pretrained: Whether to load pretrained weights (if available)
        pretrained_path: Path to custom pretrained weights
        
    Returns:
        Configured PyTorch model
        
    Raises:
        ValueError: If model_name or dataset is not supported
        FileNotFoundError: If pretrained_path is specified but file doesn't exist
    """
    
    model_name = model_name.lower()
    dataset = dataset.lower()
    
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset}. "
                        f"Supported datasets: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset]
    
    # Create model based on name
    if model_name == 'lenet5':
        model = LeNet5(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            input_size=config['input_size']
        )
    elif model_name == 'lenetxs':
        if dataset != 'mnist':
            raise ValueError("LeNetXS is designed specifically for MNIST (28x28 input). "
                           f"For {dataset}, use LeNet5 or another model.")
        model = LeNetXS(num_classes=config['num_classes'])
    elif model_name == 'simplecnn':
        model = SimpleCNN(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            input_size=config['input_size']
        )
    elif model_name == 'resnet20':
        if dataset == 'imagenet':
            raise ValueError("ResNet-20 is not suitable for ImageNet. Use ResNet-50 instead.")
        model = ResNet20(
            num_classes=config['num_classes'],
            input_channels=config['input_channels']
        )
    elif model_name == 'vgg8':
        model = VGG8(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            input_size=config['input_size']
        )
    elif model_name == 'vgg16':
        model = VGG16(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            input_size=config['input_size']
        )
    elif model_name == 'resnet50':
        model = ResNet50(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            input_size=config['input_size']
        )
    elif model_name == 'mobilenetv2':
        model = MobileNetV2(
            num_classes=config['num_classes'],
            input_channels=config['input_channels'],
            input_size=config['input_size']
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                        f"Supported models: ['lenet5', 'lenetxs', 'resnet20', 'vgg8', 'vgg16', 'resnet50', 'mobilenetv2']")
    
    # Load pretrained weights if specified
    if pretrained_path is not None:
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found at: {pretrained_path}")
        
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    elif pretrained:
        # For now, we don't have built-in pretrained weights
        # In a real implementation, you would download from a model zoo
        print(f"Warning: Pretrained weights not available for {model_name} on {dataset}. "
              f"Using randomly initialized weights.")
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }


def print_model_summary(model_name: str, dataset: str):
    """Print a summary of the model configuration."""
    model = get_model(model_name, dataset)
    info = get_model_info(model)
    config = DATASET_CONFIGS[dataset]
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name.upper()} | Dataset: {dataset.upper()}")
    print(f"{'='*50}")
    print(f"Input shape: ({config['input_channels']}, {config['input_size']}, {config['input_size']})")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Trainable parameters: {info['trainable_parameters']:,}")
    print(f"Model size: {info['model_size_mb']:.2f} MB")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Example usage
    for model_name in ['lenet5', 'resnet20', 'vgg8', 'vgg16', 'resnet50', 'mobilenetv2']:
        for dataset in ['mnist', 'cifar10', 'cifar100', 'imagenet']:
            try:
                print_model_summary(model_name, dataset)
            except ValueError as e:
                print(f"Skipping {model_name} on {dataset}: {e}")