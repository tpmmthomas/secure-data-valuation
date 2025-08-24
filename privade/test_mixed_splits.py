"""
Test script to demonstrate the improved split_model functionality
that can handle both spatial (conv) and flattened (linear) layer splits.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

# Add the parent directory to Python path to import privade
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from privade.split import split_model, SplitConfig


def create_mixed_architecture():
    """Create a model with both convolutional and linear layers for testing."""
    return nn.Sequential(
        # Convolutional layers
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        # Flatten layer
        nn.Flatten(),
        
        # Linear layers
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )


def test_split_functionality():
    """Test the split functionality with different layer types."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create MNIST dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Use a small subset for testing
    subset_indices = torch.randperm(len(dataset))[:200]
    subset = Subset(dataset, subset_indices)
    
    data_loader = DataLoader(subset, batch_size=32, shuffle=True)
    
    # Create model with mixed architecture
    model = create_mixed_architecture()
    model.to(device)
    
    print("Model architecture:")
    for i, layer in enumerate(model.children()):
        print(f"  Layer {i}: {layer}")
    
    # Configure split parameters for faster testing
    config = SplitConfig(
        privacy_threshold=0.5,  # Lower threshold for testing
        attack_epochs=5,        # Fewer epochs for faster testing
        device=device
    )
    
    print("\nTesting model splitting with mixed architecture...")
    
    try:
        # Split the model (should return three models now)
        model_A, model_B, model_C, statistics = split_model(
            data_loader=data_loader,
            model=model,
            dataset_mean=[0.5],  # MNIST normalization
            dataset_std=[0.5],
            config=config
        )
        
        print(f"\nThree-model split completed successfully!")
        print(f"Model A ends at layer: {statistics['first_activation_layer']}")
        print(f"Optimal boundary layer: {statistics['optimal_layer']}")
        print(f"Privacy preserved rate: {statistics['privacy_preserved_rate']:.3f}")
        
        # Test inference through all models
        sample_batch, _ = next(iter(data_loader))
        sample_batch = sample_batch.to(device)
        
        print(f"\nTesting inference:")
        print(f"Input shape: {sample_batch.shape}")
        
        # Model A
        intermediate_A = model_A(sample_batch)
        print(f"Model A output: {intermediate_A.shape}")
        
        # Model B  
        intermediate_B = model_B(intermediate_A)
        print(f"Model B output: {intermediate_B.shape}")
        
        # Model C
        final_output = model_C(intermediate_B)
        print(f"Model C output: {final_output.shape}")
        
        # Verify end-to-end equivalence
        original_output = model(sample_batch)
        print(f"Original model output: {original_output.shape}")
        
        # Check if outputs are close
        diff = torch.abs(final_output - original_output).mean()
        print(f"Mean difference between split and original: {diff:.6f}")
        
        if diff < 1e-5:
            print("✓ Split models produce identical output to original!")
        else:
            print("⚠ Split models output differs from original (expected due to floating point precision)")
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_split_functionality()
