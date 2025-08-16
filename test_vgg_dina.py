#!/usr/bin/env python3
"""
Test script with VGG-style model to fully test DINA integration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add the split_model directory to the path
sys.path.append('/home/thomas/secure-data-valuation/split_model')

from c2pi.boundary_finder import BoundaryFinder
from c2pi.config import C2PIConfig


class SimpleVGGStyle(nn.Module):
    """Create a VGG-style model with features attribute."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 0
            nn.ReLU(inplace=True),           # 1
            nn.Conv2d(16, 16, 3, padding=1), # 2
            nn.ReLU(inplace=True),           # 3
            nn.Conv2d(16, 32, 3, padding=1), # 4
            nn.ReLU(inplace=True),           # 5
            nn.AdaptiveAvgPool2d((8, 8))     # 6
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_dummy_data(batch_size=8, img_size=32, num_samples=16):
    """Create dummy CIFAR-10 like data."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_vgg_style_dina():
    """Test DINA with VGG-style model."""
    print("Testing DINA with VGG-style model...")
    
    # Create VGG-style model
    model = SimpleVGGStyle()
    model.eval()
    
    # Create small datasets for testing
    train_loader = create_dummy_data(batch_size=4, num_samples=8)
    test_loader = create_dummy_data(batch_size=4, num_samples=8)
    
    # Create config
    config = C2PIConfig(
        privacy_threshold=0.3,
        accuracy_threshold=0.95,
        device='cpu'  # Use CPU for testing
    )
    
    # Test split at layer after first and second ReLU
    layer_candidates = [1, 3]
    
    # Create boundary finder
    boundary_finder = BoundaryFinder(
        model=model,
        layer_candidates=layer_candidates,
        config=config
    )
    
    try:
        print("Testing privacy evaluation with VGG-style model...")
        
        # Test the privacy evaluation method directly
        privacy_result = boundary_finder._evaluate_privacy_layer(
            layer_idx=1,  # After first ReLU
            train_loader=train_loader,
            test_loader=test_loader,
            input_shape=(3, 32, 32),
            attack_type='dina',
            epochs=2,  # Very short for testing
            lr=1e-3
        )
        
        print("Privacy evaluation result:")
        for key, value in privacy_result.items():
            print(f"  {key}: {value}")
        
        print("\nVGG-style DINA test completed successfully!")
        
    except Exception as e:
        print(f"Error during VGG-style testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_vgg_style_dina()
    if success:
        print("✅ VGG-style DINA test passed!")
    else:
        print("❌ VGG-style DINA test failed!")
        sys.exit(1)
