"""
Model splitting functionality for PrivaDE.

This module implements the C2PI (Crypto-Clear Privacy-preservindef privacy_success_rate(ssim_scores: List[float], threshold: float) -> float:
    # Import the implementation from utils
    from .split_utils.metrics import privacy_success_rate as psr_impl
    return psr_impl(ssim_scores, threshold)

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
    
    # Use default config if none provided
    if config is None:
        config = C2PIConfig(
            privacy_threshold=0.6,
            ssim_threshold=0.3,
            batch_size=32,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Move model to device and set to eval mode
    model = model.to(config.device)
    model.eval()
    
    # Find the first activation layer for model_A split
    first_activation_layer = 0
    
    # Identify candidate layers for model_B/model_C boundary (starting after first activation)
    print("Identifying candidate layers for B/C boundary...")
    layer_candidates = get_candidate_layers(model, only_conv_relu=True)
    
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
    layers = list(model.children())
    
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
    print(f"Model A ends at layer: {first_activation_layer}")
    print(f"Model B starts at layer: {first_activation_layer + 1}, ends at layer: {optimal_layer}")
    print(f"Model C starts at layer: {optimal_layer + 1}")
    print(f"Privacy Rate: {opt_metrics['privacy_preserved_rate']:.3f}")
    print(f"Attack Success: {opt_metrics['attack_success_rate']:.3f}")
    print(f"Average SSIM: {opt_metrics['avg_ssim']:.3f}")
    
    return model_A, model_B, model_C, split_statistics
