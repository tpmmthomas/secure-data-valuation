#!/usr/bin/env python3
"""
Verification script to check model accuracy and identify potential issues.
"""

import torch
import torch.nn as nn
import sys
sys.path.append('..')
from privade.data import get_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def test_pretrained_model():
    """Test the pretrained model without any training to verify base accuracy."""
    
    print("="*60)
    print("VERIFYING PRETRAINED MODEL ACCURACY")
    print("="*60)
    
    # Load the pretrained model
    print("Loading pretrained ResNet-20...")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    
    # Load CIFAR-10 test data
    print("Loading CIFAR-10 dataset...")
    dataset = get_dataset("cifar10")
    
    # Get test data (use a subset for quick verification)
    test_data = dataset[40000:50000]  # CIFAR-10 test set is typically the last 10k samples
    print(f"Using {len(test_data)} test samples")
    
    # Create test loader
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # Evaluate the model
    model.eval()
    correct = 0
    total_samples = 0
    
    print("Evaluating pretrained model (no training)...")
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, label = data.cuda(), label.cuda()
            
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_samples += label.size(0)
            correct += (predicted == label).sum().item()
            
            if i % 20 == 0:
                batch_acc = (predicted == label).float().mean().item()
                print(f"Batch {i}: Accuracy = {batch_acc:.4f}")
    
    accuracy = correct / total_samples
    print(f"\n{'='*60}")
    print(f"PRETRAINED MODEL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Expected: ~0.70 (70%)")
    print(f"{'='*60}")
    
    return accuracy

def check_data_distribution():
    """Check if there's any issue with data distribution or labels."""
    
    print("\n" + "="*60)
    print("CHECKING DATA DISTRIBUTION")
    print("="*60)
    
    dataset = get_dataset("cifar10")
    print(f"Total dataset size: {len(dataset)}")
    
    # Check label distribution in first 1000 samples
    labels = [dataset[i][1] for i in range(min(1000, len(dataset)))]
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"Label distribution in first 1000 samples:")
    for label, count in sorted(label_counts.items()):
        print(f"  Class {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Check if labels are in expected range
    min_label, max_label = min(labels), max(labels)
    print(f"Label range: {min_label} to {max_label}")
    
    if min_label < 0 or max_label >= 10:
        print("❌ WARNING: Labels outside expected range [0,9] for CIFAR-10!")
    else:
        print("✅ Labels are in expected range [0,9]")

def check_model_architecture():
    """Check model architecture and parameters."""
    
    print("\n" + "="*60)
    print("CHECKING MODEL ARCHITECTURE")
    print("="*60)
    
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check final layer
    print("\nFinal layer information:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {module}")
            print(f"    Input features: {module.in_features}")
            print(f"    Output features: {module.out_features}")
    
    # Check if model is in training or eval mode
    print(f"\nModel training mode: {model.training}")

if __name__ == "__main__":
    try:
        # Run all verification checks
        pretrained_acc = test_pretrained_model()
        check_data_distribution() 
        check_model_architecture()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if pretrained_acc > 0.95:
            print("❌ ISSUE: Pretrained accuracy is suspiciously high!")
            print("   Possible causes:")
            print("   1. Data leakage (training on test data)")
            print("   2. Wrong dataset being used")
            print("   3. Model overfitting to evaluation set")
            print("   4. Incorrect evaluation methodology")
        elif pretrained_acc < 0.60:
            print("❌ ISSUE: Pretrained accuracy is lower than expected!")
            print("   Possible causes:")
            print("   1. Wrong model or weights loaded")
            print("   2. Data preprocessing mismatch")
            print("   3. Model not properly loaded")
        else:
            print("✅ Pretrained accuracy looks reasonable")
            
    except Exception as e:
        print(f"❌ ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
