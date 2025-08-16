#!/usr/bin/env python3
"""
Basic example of using C2PI to find optimal boundary layers.

This script demonstrates:
1. Loading a pre-trained model
2. Identifying candidate layers
3. Running DINA attack evaluation
4. Finding optimal boundary layer
5. Visualizing results
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2pi.boundary_finder import BoundaryFinder
from c2pi.models.utils import get_model, identify_layer_candidates
from c2pi.data.loaders import get_cifar10_loaders, get_cifar100_loaders, get_imagenet_loaders
from c2pi.attacks.dina import DINAAttacker
from c2pi.utils.visualization import plot_boundary_analysis, save_all_plots
from c2pi.config import C2PIConfig
from c2pi.utils.metrics import calculate_accuracy


def main():
    parser = argparse.ArgumentParser(description='C2PI Boundary Finding Example')
    parser.add_argument('--model', type=str, default='vgg16',
                       choices=['vgg16','vgg19','resnet50', 'alexnet'],
                       help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'imagenet'],
                       help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples for attack evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick evaluation with fewer epochs')
    parser.add_argument('--only-conv-relu', action='store_true',
                        help='Only consider conv and relu layers as candidates')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Dataset: {args.dataset}")
    only_conv_relu = args.only_conv_relu
    
    # Load configuration
    config = C2PIConfig(
        privacy_threshold=0.3,
        accuracy_threshold=0.95,
        noise_levels=[0.1, 0.05, 0.01],
        batch_size=args.batch_size,
        device=device
    )
    
    # Load dataset
    print("Loading dataset...")
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=args.batch_size,
            subset_size=args.num_samples
        )
        num_classes = 10
    elif args.dataset == 'cifar100':  # cifar100
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=args.batch_size,
            subset_size=args.num_samples
        )
        num_classes = 100
    elif args.dataset == 'imagenet':
        train_loader, test_loader = get_imagenet_loaders(
            batch_size=args.batch_size,
            subset_size=args.num_samples
        )
        num_classes = 1000
    else:
        raise NotImplementedError("Dataset not implemented")

    # Load pre-trained model
    print(f"Loading {args.model} model...")
    model = get_model(args.model, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Test model accuracy
    print("Testing baseline model accuracy...")
    baseline_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Identify candidate layers
    print("Identifying candidate layers...")
    layer_candidates = get_candidate_layers(model, only_conv_relu)
    print(f"Found {len(layer_candidates)} candidate layers: {layer_candidates}")
    
    # Initialize boundary finder
    print("Initializing boundary finder...")
    boundary_finder = BoundaryFinder(
        model=model,
        layer_candidates=layer_candidates,
        config=config
    )
    
    # Configure attack parameters
    attack_epochs = 10 if args.quick else 50
    attack_lr = 1e-3
    
    print(f"Starting boundary analysis (attack epochs: {attack_epochs})...")
    print("This may take a while...")
    
    # Run boundary finding algorithm
    try:
        results = boundary_finder.find_optimal_boundary(
            train_loader=train_loader,
            test_loader=test_loader,
            attack_type='dina',
            attack_epochs=attack_epochs,
            attack_lr=attack_lr
        )
        
        print("\n" + "="*60)
        print("BOUNDARY ANALYSIS RESULTS")
        print("="*60)
        
        # Print summary results
        for layer_idx in sorted(results.keys()):
            metrics = results[layer_idx]
            print(f"\nLayer {layer_idx}:")
            print(f"  Privacy Rate: {metrics['privacy_rate']:.3f}")
            print(f"  Attack Success: {metrics['attack_success_rate']:.3f}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Avg SSIM: {metrics['avg_ssim']:.3f}")
        
        # Find optimal boundary
        optimal_layer = boundary_finder.select_boundary_layer(results)
        
        print(f"\n{'='*60}")
        print(f"OPTIMAL BOUNDARY LAYER: {optimal_layer}")
        print(f"{'='*60}")
        
        if optimal_layer is not None:
            opt_metrics = results[optimal_layer]
            print(f"Privacy Rate: {opt_metrics['privacy_rate']:.3f}")
            print(f"Accuracy: {opt_metrics['accuracy']:.3f}")
            print(f"Attack Success: {opt_metrics['attack_success_rate']:.3f}")
        else:
            print("No boundary layer meets the criteria!")
        
        # Save results and visualizations
        print(f"\nSaving results to {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create experiment name
        experiment_name = f"{args.model}_{args.dataset}"
        
        # Save all plots
        save_all_plots(
            {'boundary_results': results},
            output_dir=args.output_dir,
            experiment_name=experiment_name
        )
        
        # Save detailed results
        import json
        results_file = os.path.join(args.output_dir, f"{experiment_name}_detailed_results.json")
        
        # Convert tensors to lists for JSON serialization
        json_results = {}
        for layer_idx, metrics in results.items():
            json_results[str(layer_idx)] = {
                k: v.tolist() if torch.is_tensor(v) else float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            }
        
        with open(results_file, 'w') as f:
            json.dump({
                'experiment_config': {
                    'model': args.model,
                    'dataset': args.dataset,
                    'num_samples': args.num_samples,
                    'batch_size': args.batch_size,
                    'attack_epochs': attack_epochs,
                    'baseline_accuracy': float(baseline_accuracy)
                },
                'optimal_boundary': int(optimal_layer) if optimal_layer is not None else None,
                'layer_results': json_results
            }, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
        print(f"Visualizations saved to: {args.output_dir}")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error during boundary analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
