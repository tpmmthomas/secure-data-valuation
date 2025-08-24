#!/usr/bin/env python3
"""
Comparative study between different attack methods (DINA vs MLA).

This script demonstrates:
1. Running both DINA and MLA attacks on the same model
2. Comparing attack effectiveness across layers
3. Analyzing trade-offs between different attack methods
4. Generating comparative visualizations
"""

import torch
import torch.nn as nn
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c2pi.boundary_finder import BoundaryFinder
from c2pi.models.utils import get_model, identify_layer_candidates
from c2pi.data.loaders import get_cifar10_loaders
from c2pi.attacks.dina import DINAAttacker
from c2pi.attacks.mla import MLAAttacker, extract_features
from c2pi.utils.visualization import plot_attack_comparison, save_all_plots
from c2pi.utils.metrics import calculate_accuracy
from c2pi.config import C2PIConfig
from torch.utils.data import DataLoader


def run_mla_evaluation(model: nn.Module,
                      layer_candidates: list,
                      train_loader: DataLoader,
                      test_loader: DataLoader,
                      device: torch.device,
                      epochs: int = 50) -> dict:
    """
    Run MLA attack evaluation across multiple layers.
    
    Args:
        model: The neural network model
        layer_candidates: List of layer indices to evaluate
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to run on
        epochs: Training epochs for MLA
    
    Returns:
        Dictionary mapping layer_idx -> MLA results
    """
    
    results = {}
    
    print("Running MLA attack evaluation...")
    
    for layer_idx in layer_candidates:
        print(f"\nEvaluating Layer {layer_idx} with MLA...")
        
        try:
            # Extract features for training MLA
            print("  Extracting training features...")
            train_features, train_targets = extract_features(
                model, train_loader, layer_idx, device
            )
            
            print("  Extracting test features...")
            test_features, test_targets = extract_features(
                model, test_loader, layer_idx, device
            )
            
            # Determine input and feature shapes
            sample_input = next(iter(test_loader))[0][:1]
            input_shape = sample_input.shape[1:]  # (C, H, W)
            feature_shape = train_features.shape[1:]
            
            print(f"  Input shape: {input_shape}, Feature shape: {feature_shape}")
            
            # Initialize MLA attacker
            mla_attacker = MLAAttacker(
                input_shape=input_shape,
                feature_shape=feature_shape,
                device=device
            )
            
            # Create data loaders for features
            from torch.utils.data import TensorDataset
            
            train_dataset = TensorDataset(train_features, train_targets)
            test_dataset = TensorDataset(test_features, test_targets)
            
            train_feat_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_feat_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            # Train MLA attack
            print(f"  Training MLA for {epochs} epochs...")
            training_history = mla_attacker.train_attack(
                feature_loader=train_feat_loader,
                target_loader=DataLoader(
                    TensorDataset(train_targets, train_targets),
                    batch_size=64, shuffle=True
                ),
                epochs=epochs,
                lr=1e-3
            )
            
            # Evaluate MLA attack
            print("  Evaluating MLA attack...")
            mla_results = mla_attacker.evaluate_attack(
                feature_loader=test_feat_loader,
                target_loader=DataLoader(
                    TensorDataset(test_targets, test_targets),
                    batch_size=64, shuffle=False
                )
            )
            
            results[layer_idx] = {
                **mla_results,
                'training_history': training_history
            }
            
            print(f"  Layer {layer_idx} MLA Results:")
            print(f"    Avg SSIM: {mla_results['avg_ssim']:.4f}")
            print(f"    Attack Success: {mla_results['attack_success']:.4f}")
            print(f"    Privacy Preserved: {mla_results['privacy_preserved']:.4f}")
            
        except Exception as e:
            print(f"  Error evaluating layer {layer_idx} with MLA: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='C2PI Attack Comparison Study')
    parser.add_argument('--model', type=str, default='vgg11',
                       choices=['vgg11', 'vgg16', 'resnet18', 'cnn5'],
                       help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=500,
                       help='Number of samples for attack evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output-dir', type=str, default='./comparison_results',
                       help='Directory to save results')
    parser.add_argument('--attack-epochs', type=int, default=30,
                       help='Training epochs for attacks')
    parser.add_argument('--layers-subset', type=int, nargs='+', default=None,
                       help='Specific layers to evaluate (default: all candidates)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Dataset: {args.dataset}")
    print(f"Running attack comparison with {args.attack_epochs} epochs")
    
    # Load configuration
    config = C2PIConfig(
        privacy_threshold=0.3,
        accuracy_threshold=0.95,
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
    else:
        from c2pi.data.loaders import get_cifar100_loaders
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=args.batch_size,
            subset_size=args.num_samples
        )
        num_classes = 100
    
    # Load model
    print(f"Loading {args.model} model...")
    model = get_model(args.model, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    model.eval()
    
    # Test baseline accuracy
    baseline_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Identify candidate layers
    layer_candidates = identify_layer_candidates(model)
    
    # Use subset if specified
    if args.layers_subset:
        layer_candidates = [l for l in layer_candidates if l in args.layers_subset]
    
    print(f"Evaluating layers: {layer_candidates}")
    
    # Run DINA evaluation
    print("\n" + "="*50)
    print("RUNNING DINA ATTACK EVALUATION")
    print("="*50)
    
    boundary_finder = BoundaryFinder(
        model=model,
        layer_candidates=layer_candidates,
        config=config
    )
    
    try:
        dina_results = boundary_finder.find_optimal_boundary(
            train_loader=train_loader,
            test_loader=test_loader,
            attack_type='dina',
            attack_epochs=args.attack_epochs,
            attack_lr=1e-3
        )
        
        print("DINA evaluation completed!")
        
    except Exception as e:
        print(f"DINA evaluation failed: {e}")
        dina_results = {}
    
    # Run MLA evaluation
    print("\n" + "="*50)
    print("RUNNING MLA ATTACK EVALUATION")
    print("="*50)
    
    try:
        mla_results = run_mla_evaluation(
            model=model,
            layer_candidates=layer_candidates,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.attack_epochs
        )
        
        print("MLA evaluation completed!")
        
    except Exception as e:
        print(f"MLA evaluation failed: {e}")
        mla_results = {}
    
    # Compare results
    print("\n" + "="*60)
    print("ATTACK COMPARISON RESULTS")
    print("="*60)
    
    common_layers = sorted(set(dina_results.keys()) & set(mla_results.keys()))
    
    if common_layers:
        print(f"\nComparing results for layers: {common_layers}")
        print("-" * 80)
        print(f"{'Layer':<8} {'DINA SSIM':<12} {'MLA SSIM':<12} {'DINA Success':<15} {'MLA Success':<15}")
        print("-" * 80)
        
        for layer in common_layers:
            dina_ssim = dina_results[layer]['avg_ssim']
            mla_ssim = mla_results[layer]['avg_ssim']
            dina_success = dina_results[layer]['attack_success_rate']
            mla_success = mla_results[layer]['attack_success']
            
            print(f"{layer:<8} {dina_ssim:<12.4f} {mla_ssim:<12.4f} "
                  f"{dina_success:<15.4f} {mla_success:<15.4f}")
    else:
        print("No common layers found between DINA and MLA results!")
    
    # Save results
    print(f"\nSaving comparison results to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    experiment_name = f"comparison_{args.model}_{args.dataset}"
    
    # Prepare combined results for visualization
    combined_results = {
        'dina_results': dina_results,
        'mla_results': mla_results
    }
    
    # Save all plots
    save_all_plots(
        combined_results,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        privacy_threshold=config.privacy_threshold,
        ssim_threshold=config.ssim_threshold
    )
    
    # Save detailed comparison
    import json
    
    # Convert results for JSON serialization
    def convert_for_json(results_dict):
        json_dict = {}
        for layer_idx, metrics in results_dict.items():
            json_dict[str(layer_idx)] = {
                k: v.tolist() if torch.is_tensor(v) else 
                   float(v) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
                if k != 'training_history'  # Skip complex nested structures
            }
        return json_dict
    
    comparison_file = os.path.join(args.output_dir, f"{experiment_name}_comparison.json")
    
    with open(comparison_file, 'w') as f:
        json.dump({
            'experiment_config': {
                'model': args.model,
                'dataset': args.dataset,
                'num_samples': args.num_samples,
                'batch_size': args.batch_size,
                'attack_epochs': args.attack_epochs,
                'baseline_accuracy': float(baseline_accuracy),
                'evaluated_layers': layer_candidates
            },
            'dina_results': convert_for_json(dina_results),
            'mla_results': convert_for_json(mla_results),
            'common_layers': common_layers
        }, f, indent=2)
    
    print(f"Detailed comparison saved to: {comparison_file}")
    print(f"Visualizations saved to: {args.output_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if common_layers:
        avg_dina_ssim = sum(dina_results[l]['avg_ssim'] for l in common_layers) / len(common_layers)
        avg_mla_ssim = sum(mla_results[l]['avg_ssim'] for l in common_layers) / len(common_layers)
        
        print(f"Average DINA SSIM: {avg_dina_ssim:.4f}")
        print(f"Average MLA SSIM: {avg_mla_ssim:.4f}")
        
        if avg_dina_ssim > avg_mla_ssim:
            print("DINA appears more effective on average")
        else:
            print("MLA appears more effective on average")
    
    print("\nComparison study completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())
