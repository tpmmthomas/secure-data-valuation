"""
Visualization utilities for C2PI boundary analysis and attack results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
from pathlib import Path


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_boundary_analysis(results: Dict[int, Dict], 
                          save_path: Optional[str] = None,
                          title: str = "Boundary Layer Analysis",
                          privacy_threshold: float = 0.7) -> plt.Figure:
    """
    Plot privacy vs accuracy trade-off for different boundary layers.
    
    Args:
        results: Dictionary mapping layer_idx -> metrics
        save_path: Path to save the plot
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    
    # Extract data for plotting
    layers = sorted(results.keys())
    privacy_rates = [results[layer]['privacy_preserved_rate'] for layer in layers]
    accuracies = [results[layer]['accuracy'] for layer in layers]
    attack_success_rates = [results[layer]['attack_success_rate'] for layer in layers]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Privacy Rate vs Layer
    ax1.plot(layers, privacy_rates, 'o-', linewidth=2, markersize=8, label='Privacy Rate')
    ax1.axhline(y=privacy_threshold, color='r', linestyle='--', alpha=0.7, label=f'Target ({privacy_threshold*100:.0f}%)')
    ax1.set_xlabel('Boundary Layer')
    ax1.set_ylabel('Privacy Rate')
    ax1.set_title('Privacy Preservation by Layer')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Plot 2: Accuracy vs Layer
    ax2.plot(layers, accuracies, 'o-', linewidth=2, markersize=8, color='green', label='Accuracy')
    ax2.axhline(y=max(accuracies) * 0.95, color='r', linestyle='--', alpha=0.7, 
                label='95% of Max')
    ax2.set_xlabel('Boundary Layer')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Accuracy by Layer')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Plot 3: Privacy vs Accuracy Trade-off
    scatter = ax3.scatter(privacy_rates, accuracies, c=layers, s=100, alpha=0.7, 
                         cmap='viridis')
    ax3.set_xlabel('Privacy Rate')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Privacy vs Accuracy Trade-off')
    ax3.grid(True, alpha=0.3)
    
    # Add colorbar for layer information
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Boundary Layer')
    
    # Add annotations for interesting points
    for i, layer in enumerate(layers):
        if i % 2 == 0:  # Annotate every other point to avoid clutter
            ax3.annotate(f'L{layer}', (privacy_rates[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_attack_comparison(dina_results: Dict, 
                          mla_results: Dict,
                          save_path: Optional[str] = None,
                          ssim_threshold: float = 0.3) -> plt.Figure:
    """
    Compare DINA and MLA attack results across layers.
    
    Args:
        dina_results: DINA attack results by layer
        mla_results: MLA attack results by layer
        save_path: Path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    
    # Find common layers
    common_layers = sorted(set(dina_results.keys()) & set(mla_results.keys()))
    
    dina_ssim = [dina_results[layer]['avg_ssim'] for layer in common_layers]
    mla_ssim = [mla_results[layer]['avg_ssim'] for layer in common_layers]
    
    dina_success = [dina_results[layer]['attack_success'] for layer in common_layers]
    mla_success = [mla_results[layer]['attack_success'] for layer in common_layers]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: SSIM Comparison
    ax1.plot(common_layers, dina_ssim, 'o-', label='DINA', linewidth=2, markersize=8)
    ax1.plot(common_layers, mla_ssim, 's-', label='MLA', linewidth=2, markersize=8)
    ax1.axhline(y=ssim_threshold, color='r', linestyle='--', alpha=0.7, label='Privacy Threshold')
    ax1.set_xlabel('Boundary Layer')
    ax1.set_ylabel('Average SSIM')
    ax1.set_title('Attack Quality: SSIM Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate Comparison
    ax2.plot(common_layers, dina_success, 'o-', label='DINA', linewidth=2, markersize=8)
    ax2.plot(common_layers, mla_success, 's-', label='MLA', linewidth=2, markersize=8)
    ax2.set_xlabel('Boundary Layer')
    ax2.set_ylabel('Attack Success Rate')
    ax2.set_title('Attack Success Rate Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.suptitle('DINA vs MLA Attack Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def plot_attack_samples(original: torch.Tensor,
                       reconstructed: torch.Tensor,
                       ssim_scores: torch.Tensor,
                       n_samples: int = 8,
                       save_path: Optional[str] = None,
                       ssim_threshold: float = 0.3) -> plt.Figure:
    """
    Plot original vs reconstructed samples from attack.
    
    Args:
        original: Original images (B, C, H, W)
        reconstructed: Reconstructed images (B, C, H, W)
        ssim_scores: SSIM scores for each pair
        n_samples: Number of samples to show
        save_path: Path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    
    # Ensure tensors are on CPU and select samples
    original = original.cpu()[:n_samples]
    reconstructed = reconstructed.cpu()[:n_samples]
    ssim_scores = ssim_scores.cpu()[:n_samples]
    
    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i in range(n_samples):
        # Original image
        img_orig = original[i].permute(1, 2, 0).numpy()
        img_orig = np.clip(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f'Original {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed image
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        img_recon = np.clip(img_recon, 0, 1)
        axes[1, i].imshow(img_recon)
        axes[1, i].set_title(f'Reconstructed\nSSIM: {ssim_scores[i]:.3f}', fontsize=10)
        axes[1, i].axis('off')
        
        # Color code based on privacy threshold
        if ssim_scores[i] >= ssim_threshold:
            # Attack succeeded (privacy compromised)
            for ax in [axes[0, i], axes[1, i]]:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
                    spine.set_visible(True)
        else:
            # Privacy preserved
            for ax in [axes[0, i], axes[1, i]]:
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)
                    spine.set_visible(True)
    
    plt.suptitle('Attack Results: Original vs Reconstructed\n' +
                 f'Red border: Privacy compromised (SSIM ≥ {ssim_threshold}), Green border: Privacy preserved',
                 fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample comparison saved to: {save_path}")
    
    return fig


def plot_training_history(history: Dict[str, List[float]],
                         save_path: Optional[str] = None,
                         title: str = "Training History") -> plt.Figure:
    """
    Plot training history (loss, SSIM, etc.).
    
    Args:
        history: Dictionary with training metrics over epochs
        save_path: Path to save the plot
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    
    n_metrics = len(history)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    epochs = range(1, len(list(history.values())[0]) + 1)
    
    for i, (metric_name, values) in enumerate(history.items()):
        axes[i].plot(epochs, values, 'o-', linewidth=2, markersize=4)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric_name.replace('_', ' ').title())
        axes[i].set_title(f'{metric_name.replace("_", " ").title()} vs Epoch')
        axes[i].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(epochs, values, 1)
        p = np.poly1d(z)
        axes[i].plot(epochs, p(epochs), "r--", alpha=0.8, linewidth=1)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    return fig


def plot_model_architecture(layer_candidates: List[int],
                           layer_names: List[str],
                           selected_boundary: int,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize model architecture with boundary layer highlighted.
    
    Args:
        layer_candidates: List of candidate layer indices
        layer_names: Names of the layers
        selected_boundary: Selected boundary layer index
        save_path: Path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create a simple architecture diagram
    n_layers = len(layer_candidates)
    positions = np.arange(n_layers)
    
    # Draw layer boxes
    colors = ['lightblue' if i < selected_boundary else 'lightcoral' 
              for i in layer_candidates]
    colors[layer_candidates.index(selected_boundary)] = 'gold'
    
    bars = ax.barh(positions, [1]*n_layers, color=colors, alpha=0.7)
    
    # Add layer names
    for i, (pos, name) in enumerate(zip(positions, layer_names)):
        ax.text(0.5, pos, name, ha='center', va='center', fontweight='bold')
    
    # Customize plot
    ax.set_yticks(positions)
    ax.set_yticklabels([f'Layer {idx}' for idx in layer_candidates])
    ax.set_xlabel('Execution Domain')
    ax.set_title('Model Architecture with Boundary Layer', fontsize=14)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', alpha=0.7, label='Crypto (MPC)'),
        Patch(facecolor='gold', alpha=0.7, label='Boundary Layer'),
        Patch(facecolor='lightcoral', alpha=0.7, label='Clear (Plaintext)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Remove x-axis ticks and spines
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to: {save_path}")
    
    return fig


def create_results_table(results: Dict[int, Dict],
                        save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a summary table of boundary analysis results.
    
    Args:
        results: Dictionary mapping layer_idx -> metrics
        save_path: Path to save CSV file
    
    Returns:
        pandas DataFrame with results
    """
    
    # Convert results to DataFrame
    data = []
    for layer_idx, metrics in results.items():
        row = {
            'Boundary_Layer': layer_idx,
            'Privacy_Rate': f"{metrics['privacy_preserved_rate']:.3f}",
            'Attack_Success_Rate': f"{metrics['attack_success_rate']:.3f}",
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Avg_SSIM': f"{metrics['avg_ssim']:.3f}",
            'Max_SSIM': f"{metrics['max_ssim']:.3f}",
            'Std_SSIM': f"{metrics['std_ssim']:.3f}"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results table saved to: {save_path}")
    
    return df


def save_all_plots(results: Dict,
                  output_dir: str,
                  experiment_name: str = "c2pi_experiment",
                  privacy_threshold: float = 0.7,
                  ssim_threshold: float = 0.3):
    """
    Save all visualization plots to a directory.
    
    Args:
        results: Complete experimental results
        output_dir: Output directory for plots
        experiment_name: Name prefix for saved files
        privacy_threshold: Privacy threshold for visualization
        ssim_threshold: SSIM threshold for visualization
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot boundary analysis
    if 'boundary_results' in results:
        plot_boundary_analysis(
            results['boundary_results'],
            save_path=output_path / f"{experiment_name}_boundary_analysis.png",
            title=f"{experiment_name.replace('_', ' ').title()} - Boundary Analysis",
            privacy_threshold=privacy_threshold
        )
    
    # Plot attack comparison if both attacks were used
    if 'dina_results' in results and 'mla_results' in results:
        plot_attack_comparison(
            results['dina_results'],
            results['mla_results'],
            save_path=output_path / f"{experiment_name}_attack_comparison.png",
            ssim_threshold=ssim_threshold
        )
    
    # Plot training histories
    if 'training_history' in results:
        plot_training_history(
            results['training_history'],
            save_path=output_path / f"{experiment_name}_training_history.png",
            title=f"{experiment_name.replace('_', ' ').title()} - Training History"
        )
    
    # Save results table
    if 'boundary_results' in results:
        create_results_table(
            results['boundary_results'],
            save_path=output_path / f"{experiment_name}_results_table.csv"
        )
    
    print(f"All plots and tables saved to: {output_path}")


def plot_performance_metrics(results: Dict[int, Dict],
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot computational performance metrics (speedup, communication reduction).
    
    Args:
        results: Dictionary mapping layer_idx -> metrics (including performance)
        save_path: Path to save the plot
    
    Returns:
        matplotlib Figure object
    """
    
    layers = sorted(results.keys())
    
    # Extract performance metrics if available
    speedups = [results[layer].get('estimated_speedup', 1.0) for layer in layers]
    comm_reductions = [results[layer].get('communication_reduction', 1.0) for layer in layers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Speedup
    ax1.plot(layers, speedups, 'o-', linewidth=2, markersize=8, color='purple')
    ax1.set_xlabel('Boundary Layer')
    ax1.set_ylabel('Estimated Speedup')
    ax1.set_title('Computational Speedup vs Boundary Layer')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Communication Reduction
    ax2.plot(layers, comm_reductions, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Boundary Layer')
    ax2.set_ylabel('Communication Reduction Factor')
    ax2.set_title('Communication Reduction vs Boundary Layer')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle('Performance Metrics Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to: {save_path}")
    
    return fig
