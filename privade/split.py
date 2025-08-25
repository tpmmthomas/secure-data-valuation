"""
Model splitting functionality for PrivaDE.

This module implements the C2PI (Crypto-Clear Privacy-preserving Inference) algorithm
for finding optimal boundary layers in neural networks for split inference with privacy guarantees.
It uses the DINA (Distillation-based Inverse Network Attack) to evaluate privacy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from tqdm import tqdm
import json
import os
from dataclasses import dataclass, field
from .c2pi.boundary_finder import BoundaryFinder
from .c2pi.models.utils import get_model,  get_candidate_layers
from .c2pi.config import C2PIConfig
from .c2pi.utils.metrics import calculate_accuracy

ACTIVATION_NAMES = {
    'ReLU','LeakyReLU','ELU','SELU','GELU','Tanh','Sigmoid',
    'Softmax','LogSoftmax','ReLU6','PReLU','Square'
}

def is_activation(layer: nn.Module) -> bool:
    return layer._get_name() in ACTIVATION_NAMES

def get_first_activation_layer(model: nn.Module) -> int:
    """
    Find the index of the first activation layer in a model.
    
    Args:
        model: The neural network model
        
    Returns:
        Index of the first activation layer (ReLU, LeakyReLU, ELU, etc.)
        Returns 0 if no activation layer is found
    """
    
    # If model is Sequential, check its children directly
    if isinstance(model, nn.Sequential):
        layers = list(model.children())
        for i, layer in enumerate(layers):
            if is_activation(layer):
                print(f"Found first activation layer at index {i}: {type(layer).__name__}")
                return i
    else:
        # For non-Sequential models, we need to traverse the structure
        # This is more complex for models with residual connections, etc.
        # For now, we'll flatten as much as possible
        def get_all_modules(module):
            """Recursively get all modules in order."""
            modules = []
            for child in module.children():
                if isinstance(child, nn.Sequential):
                    # Flatten Sequential containers
                    modules.extend(get_all_modules(child))
                elif len(list(child.children())) == 0:
                    # Leaf module
                    modules.append(child)
                else:
                    # Intermediate module with children
                    modules.extend(get_all_modules(child))
            return modules
        
        all_modules = get_all_modules(model)
        for i, layer in enumerate(all_modules):
            if is_activation(layer):
                print(f"Found first activation layer at index {i}: {type(layer).__name__}")
                return i
    
    print("Warning: No activation layer found in model. Using index 0 as fallback.")
    return 0



def split_model(data_loader: DataLoader,
                model: nn.Module,
                config: Optional[C2PIConfig] = None) -> Tuple[nn.Module, nn.Module, nn.Module, Dict[str, Any]]:
    """
    Split a model into three parts using privacy-preserving boundary finding.
    
    This function implements a three-model split:
    - model_A: Fixed split at first activation layer (always the same)
    - model_B: From first activation to optimal boundary (found using C2PI)
    - model_C: From optimal boundary to end (server-side)
    
    Args:
        data_loader: DataLoader for the dataset to use for evaluation
        model: The neural network model to split
        dataset_mean: Dataset normalization mean values (auto-detected if None)
        dataset_std: Dataset normalization std values (auto-detected if None)
        config: Split configuration (uses default if None)
        
    Returns:
        Tuple of (model_A, model_B, model_C, split_statistics)
        where split_statistics contains:
        - 'first_activation_layer': Index of the first activation layer
        - 'optimal_layer': Index of the optimal boundary layer between B and C
        - 'layer_results': Dictionary of all layer evaluation results
        - 'privacy_preserved_rate': Privacy preservation rate at optimal layer
        - 'attack_success_rate': Attack success rate at optimal layer
    """
    
    #Check img size
    img_size = next(iter(data_loader))[0].shape[-1]
    
    
    # Use default config if none provided
    if config is None:
        config = C2PIConfig(
            privacy_threshold=0.6,
            ssim_threshold=0.3,
            img_size=img_size,
            batch_size=32,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Move model to device and set to eval mode
    model = model.to(config.device)
    model.eval()
    
    # Find the first activation layer for model_A split
    first_activation_layer = get_first_activation_layer(model)
    print("First activation layer: ", first_activation_layer)
    
    # Identify candidate layers for model_B/model_C boundary (starting after first activation)
    print("Identifying candidate layers for B/C boundary...")
    layer_candidates = get_candidate_layers(model, only_conv_relu=True)
    print("Candidates", layer_candidates)
    
    # Filter candidates to only include layers after the first activation
    layer_candidates = [idx for idx in layer_candidates if idx > first_activation_layer]
    
    print(f"Found {len(layer_candidates)} candidate layers: {layer_candidates}")
    
    if not layer_candidates:
        raise ValueError("No candidate layers found for boundary placement")
    
    # Initialize boundary finder
    boundary_finder = BoundaryFinder(
        model=model,
        layer_candidates=layer_candidates,
        config=config
    )
    
    print("Starting boundary analysis with DINA attack...")
    print(f"Attack epochs: {config.attack_epochs}")
    print("This may take a while...")
    
    # Run boundary finding algorithm
    results = boundary_finder.find_optimal_boundary(
        train_loader=data_loader,
        test_loader=data_loader,  # Use same loader for both train and test
        attack_type='dina',
        attack_epochs=config.attack_epochs,
        attack_lr=config.learning_rate
    )
    
    if not results:
        raise ValueError("No results obtained from boundary analysis")
    
    print("\n" + "="*60)
    print("BOUNDARY ANALYSIS RESULTS")
    print("="*60)
    
    # Print summary results
    for layer_idx in sorted(results.keys()):
        metrics = results[layer_idx]
        print(f"\nLayer {layer_idx}:")
        print(f"  Privacy Preserve Rate: {metrics['privacy_preserved_rate']:.3f}")
        print(f"  Attack Success: {metrics['attack_success_rate']:.3f}")
        print(f"  Avg SSIM: {metrics['avg_ssim']:.3f}")
    
    # Find optimal boundary
    optimal_layer = boundary_finder.select_boundary_layer(results)
    
    print(f"\n{'='*60}")
    print(f"OPTIMAL BOUNDARY LAYER: {optimal_layer}")
    print(f"{'='*60}")
    
    if optimal_layer is None:
        print("Warning: No boundary layer meets the criteria! Using best available layer.")
        # Select layer with highest privacy rate
        best_layer = max(results.keys(), 
                        key=lambda x: results[x]['privacy_preserved_rate'])
        optimal_layer = best_layer
        print(f"Using layer {optimal_layer} as fallback.")
    
    # Create three split models
    # First, flatten the model into individual layers (same logic as get_candidate_layers)
    def flatten_model(module):
        """Flatten a model into a sequential list of layers."""
        if isinstance(module, nn.Sequential):
            return list(module.children())
        else:
            # For non-Sequential models, collect all child modules
            layers = []
            def collect_layers(m):
                children = list(m.children())
                if not children:
                    # Leaf module
                    layers.append(m)
                else:
                    # Has children - recurse
                    for child in children:
                        if isinstance(child, nn.Sequential):
                            # Flatten Sequential containers
                            layers.extend(child.children())
                        else:
                            collect_layers(child)
            collect_layers(module)
            return layers
    
    layers = flatten_model(model)
    
    print(f"\nFlattened model has {len(layers)} layers:")
    for i, layer in enumerate(layers):
        print(f"  {i}: {type(layer).__name__}")
    
    # Verify that our layer indices make sense
    if first_activation_layer >= len(layers):
        raise ValueError(f"First activation layer index {first_activation_layer} exceeds model length {len(layers)}")
    if optimal_layer >= len(layers):
        raise ValueError(f"Optimal layer index {optimal_layer} exceeds model length {len(layers)}")
    
    # Model A: From start to first activation (inclusive)
    model_A_layers = layers[:first_activation_layer + 1]
    model_A = nn.Sequential(*model_A_layers)
    
    # Model B: From first activation (exclusive) to optimal boundary (inclusive)
    model_B_layers = layers[first_activation_layer + 1:optimal_layer + 1]
    model_B = nn.Sequential(*model_B_layers)
    
    # Model C: From optimal boundary (exclusive) to end
    model_C_layers = layers[optimal_layer + 1:]
    model_C = nn.Sequential(*model_C_layers)
    
    # Prepare statistics
    opt_metrics = results[optimal_layer]
    split_statistics = {
        'first_activation_layer': first_activation_layer,
        'optimal_layer': optimal_layer,
        'layer_results': results,
        'privacy_preserved_rate': opt_metrics['privacy_preserved_rate'],
        'attack_success_rate': opt_metrics['attack_success_rate'],
        'avg_ssim': opt_metrics['avg_ssim'],
        'config': {
            'privacy_threshold': config.privacy_threshold,
            'attack_epochs': config.attack_epochs,
            'total_candidates': len(layer_candidates),
            'candidate_layers': layer_candidates
        }
    }
    
    print(f"\nThree-Model Split Statistics:")
    print(f"Total layers: {len(layers)}")
    print(f"Model A ends at layer: {first_activation_layer} ({type(layers[first_activation_layer]).__name__})")
    print(f"Model B: layers {first_activation_layer + 1} to {optimal_layer} ({len(model_B_layers)} layers)")
    print(f"Model C: layers {optimal_layer + 1} to {len(layers)-1} ({len(model_C_layers)} layers)")
    print(f"Privacy Rate: {opt_metrics['privacy_preserved_rate']:.3f}")
    print(f"Attack Success: {opt_metrics['attack_success_rate']:.3f}")
    print(f"Average SSIM: {opt_metrics['avg_ssim']:.3f}")
    
    # Detailed split breakdown
    print(f"\nModel A layers ({len(model_A_layers)}):")
    for i, layer in enumerate(model_A_layers):
        print(f"  {i}: {type(layer).__name__}")
    
    print(f"\nModel B layers ({len(model_B_layers)}):")
    for i, layer in enumerate(model_B_layers):
        print(f"  {i}: {type(layer).__name__}")
    
    print(f"\nModel C layers ({len(model_C_layers)}):")
    for i, layer in enumerate(model_C_layers):
        print(f"  {i}: {type(layer).__name__}")
    
    return model_A, model_B, model_C, split_statistics
