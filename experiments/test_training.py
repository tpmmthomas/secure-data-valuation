#!/usr/bin/env python3
"""
Quick test script to evaluate current training performance and suggest improvements.
"""

import sys
sys.path.append('..')
from privade.data import get_dataset
from privade.models import get_model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data import split_dataset

# Current configuration from experiment_AL.py
BATCH_SIZE = 10
DATASET = "cifar10"  
MODEL = "resnet20"
LR = 1e-4

def test_configuration():
    """Test current configuration and suggest improvements."""
    
    print("=" * 60)
    print(f"TESTING CURRENT CONFIGURATION")
    print(f"Dataset: {DATASET}")
    print(f"Model: {MODEL}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    print("=" * 60)
    
    # Load dataset
    dataset = get_dataset(DATASET)
    pretrain_size = 1000
    train_data, _ = split_dataset(dataset, pretrain_size, 10000)
    
    print(f"Training data size: {len(train_data)}")
    print(f"Sample data shape: {train_data[0][0].shape}")
    print(f"Number of classes: {len(set([x[1] for x in train_data]))}")
    
    # Load model
    model = get_model(MODEL, DATASET)
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Quick training test
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    print("\nRunning 10 training steps...")
    for i, (data, label) in enumerate(train_loader):
        if i >= 10:  # Just test 10 batches
            break
            
        if torch.cuda.is_available():
            data, label = data.cuda(), label.cuda()
            
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if i % 5 == 0:
            print(f"  Step {i+1}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"\nAverage loss over 10 steps: {avg_loss:.4f}")
    
    return avg_loss

def suggest_improvements():
    """Suggest parameter improvements for better accuracy."""
    
    print("\n" + "=" * 60)
    print("SUGGESTED IMPROVEMENTS")
    print("=" * 60)
    
    improvements = [
        {
            "issue": "Very small batch size (10)",
            "current": "BATCH_SIZE = 10", 
            "suggested": "BATCH_SIZE = 32 or 64",
            "reason": "Small batches lead to noisy gradients and unstable training. ResNet models typically work better with larger batches."
        },
        {
            "issue": "Very low learning rate", 
            "current": "LR = 1e-4",
            "suggested": "LR = 1e-3 or 3e-4", 
            "reason": "1e-4 is too conservative for initial training. ResNet20 can handle higher learning rates."
        },
        {
            "issue": "BatchNorm set to eval mode",
            "current": "for m in model.modules(): if isinstance(m, nn.BatchNorm2d): m.eval()",
            "suggested": "Remove this line entirely",
            "reason": "BatchNorm should be in training mode during training. Setting to eval mode breaks the training process."
        },
        {
            "issue": "No learning rate scheduling",
            "current": "No scheduler in initial training",
            "suggested": "Add CosineAnnealingLR or MultiStepLR",
            "reason": "Learning rate decay helps achieve better final accuracy."
        },
        {
            "issue": "Too aggressive scheduler in train_and_evaluate",
            "current": "StepLR(optimizer, step_size=1, gamma=0.99)", 
            "suggested": "StepLR(optimizer, step_size=3, gamma=0.8) or remove",
            "reason": "Decaying LR every step is too aggressive for short training periods."
        }
    ]
    
    for i, imp in enumerate(improvements, 1):
        print(f"\n{i}. {imp['issue']}")
        print(f"   Current: {imp['current']}")
        print(f"   Suggested: {imp['suggested']}")
        print(f"   Reason: {imp['reason']}")
    
    print(f"\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 60)
    print("""
# Improved parameters
BATCH_SIZE = 32  # Better gradient estimates
LR = 3e-4        # More aggressive but stable learning rate  
num_epochs = 50  # More training epochs for initial model

# Initial training improvements:
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)  # Add weight decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Remove BatchNorm eval mode setting
# for m in model.modules():
#     if isinstance(m, nn.BatchNorm2d):
#         m.eval()  # <-- DELETE THIS LINE

# train_and_evaluate improvements:
def train_and_evaluate(model, train_data, test_data):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # Less aggressive
    num_epochs = 15  # Slightly more epochs
    """)

if __name__ == "__main__":
    test_configuration()
    suggest_improvements()
