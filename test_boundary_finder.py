#!/usr/bin/env python3
"""
Test script to verify the boundary finder implementation.
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


def create_simple_model():
    """Create a simple CNN model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10)
    )


def create_dummy_data(batch_size=8, img_size=32, num_samples=16):
    """Create dummy CIFAR-10 like data."""
    images = torch.randn(num_samples, 3, img_size, img_size)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_boundary_finder():
    """Test the boundary finder implementation."""
    print("Testing Boundary Finder...")
    
    # Create simple model and data
    model = create_simple_model()
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
    
    # Only test the first few layers to keep it simple
    layer_candidates = [1, 3]  # After first ReLU and second ReLU
    
    # Create boundary finder
    boundary_finder = BoundaryFinder(
        model=model,
        layer_candidates=layer_candidates,
        config=config
    )
    
    try:
        print("Testing privacy evaluation...")
        
        # Test the privacy evaluation method directly
        privacy_result = boundary_finder._evaluate_privacy_layer(
            layer_idx=1,
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
        
        print("\nBoundary finder test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_boundary_finder()
    if success:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")
        sys.exit(1)
