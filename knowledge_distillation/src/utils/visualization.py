"""Visualization utilities for knowledge distillation."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


def plot_training_curves(
    history: Dict[str, List[Dict]], 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """Plot training curves for teacher and student models."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Knowledge Distillation Training Curves', fontsize=16)
    
    # Helper function to extract metrics
    def extract_metric(data, metric_name, split='train'):
        epochs = []
        values = []
        for entry in data:
            if split in entry:
                epochs.append(entry['epoch'])
                values.append(entry[split][metric_name])
        return epochs, values
    
    # Plot teacher training curves
    if 'teacher' in history:
        teacher_data = history['teacher']
        
        # Teacher loss
        epochs, train_loss = extract_metric(teacher_data, 'loss', 'train')
        _, test_loss = extract_metric(teacher_data, 'loss', 'test')
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, test_loss, 'r-', label='Test Loss')
        axes[0, 0].set_title('Teacher Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Teacher accuracy
        _, train_acc = extract_metric(teacher_data, 'acc', 'train')
        _, test_acc = extract_metric(teacher_data, 'acc', 'test')
        axes[0, 1].plot(epochs, train_acc, 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, test_acc, 'r-', label='Test Acc')
        axes[0, 1].set_title('Teacher Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot student training curves
    if 'student_kd' in history:
        student_data = history['student_kd']
        
        # Student loss
        epochs, train_loss = extract_metric(student_data, 'loss', 'train')
        _, test_loss = extract_metric(student_data, 'loss', 'test')
        axes[1, 0].plot(epochs, train_loss, 'b-', label='Total Loss')
        
        # Also plot CE and KD components if available
        _, train_ce = extract_metric(student_data, 'ce', 'train')
        _, train_kd = extract_metric(student_data, 'kd', 'train')
        if train_ce and train_kd:
            axes[1, 0].plot(epochs, train_ce, 'g--', label='CE Loss')
            axes[1, 0].plot(epochs, train_kd, 'm--', label='KD Loss')
        
        axes[1, 0].plot(epochs, test_loss, 'r-', label='Test Loss')
        axes[1, 0].set_title('Student Loss (Knowledge Distillation)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Student accuracy
        _, train_acc = extract_metric(student_data, 'acc', 'train')
        _, test_acc = extract_metric(student_data, 'acc', 'test')
        axes[1, 1].plot(epochs, train_acc, 'b-', label='Train Acc')
        axes[1, 1].plot(epochs, test_acc, 'r-', label='Test Acc')
        axes[1, 1].set_title('Student Accuracy (Knowledge Distillation)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    teacher_acc: float,
    student_supervised_acc: float,
    student_kd_acc: float,
    save_path: Optional[str] = None
):
    """Plot accuracy comparison between models."""
    models = ['Teacher', 'Student\n(Supervised)', 'Student\n(Knowledge Distillation)']
    accuracies = [teacher_acc, student_supervised_acc, student_kd_acc]
    colors = ['blue', 'orange', 'green']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.ylim(0, max(accuracies) + 10)
    plt.grid(True, alpha=0.3)
    
    # Highlight improvement
    improvement = student_kd_acc - student_supervised_acc
    plt.text(0.5, max(accuracies) + 5, f'KD Improvement: +{improvement:.2f}%',
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def visualize_predictions(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[str] = None
):
    """Visualize predictions from teacher and student models."""
    teacher_model.eval()
    student_model.eval()
    
    # Get a batch of data
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    images, labels = images[:num_samples], labels[:num_samples]
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        teacher_outputs = teacher_model(images)
        student_outputs = student_model(images)
        
        teacher_probs = F.softmax(teacher_outputs, dim=1)
        student_probs = F.softmax(student_outputs, dim=1)
        
        teacher_preds = teacher_outputs.argmax(dim=1)
        student_preds = student_outputs.argmax(dim=1)
    
    # Create subplot grid
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
    fig.suptitle('Teacher vs Student Predictions', fontsize=16)
    
    for i in range(min(num_samples, rows * cols)):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # Convert image to displayable format
        img = images[i].cpu()
        if img.shape[0] == 1:  # Grayscale
            img = img.squeeze(0)
            ax.imshow(img, cmap='gray')
        else:  # RGB
            img = img.permute(1, 2, 0)
            # Denormalize if needed (assuming ImageNet normalization)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = torch.clamp(img, 0, 1)
            ax.imshow(img)
        
        # Add prediction information
        true_label = labels[i].item()
        teacher_pred = teacher_preds[i].item()
        student_pred = student_preds[i].item()
        
        teacher_conf = teacher_probs[i, teacher_pred].item()
        student_conf = student_probs[i, student_pred].item()
        
        title = f'True: {true_label}\n'
        title += f'Teacher: {teacher_pred} ({teacher_conf:.2f})\n'
        title += f'Student: {student_pred} ({student_conf:.2f})'
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_temperature_analysis(
    teacher_logits: torch.Tensor,
    temperatures: List[float] = [1.0, 2.0, 4.0, 8.0, 16.0],
    save_path: Optional[str] = None
):
    """Analyze the effect of temperature on softmax distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Take first sample for visualization
    logits = teacher_logits[0]
    
    for i, T in enumerate(temperatures):
        if i >= len(axes):
            break
            
        probs = F.softmax(logits / T, dim=0).cpu().numpy()
        
        ax = axes[i]
        ax.bar(range(len(probs)), probs)
        ax.set_title(f'Temperature = {T}', fontsize=14)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)
        
        # Add entropy calculation
        entropy = -(probs * np.log(probs + 1e-8)).sum()
        ax.text(0.7, 0.9, f'Entropy: {entropy:.3f}', 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Remove unused subplots
    for i in range(len(temperatures), len(axes)):
        fig.delaxes(axes[i])
    
    fig.suptitle('Effect of Temperature on Softmax Distribution', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_experiment_summary(
    config: Dict,
    results: Dict,
    save_path: str
):
    """Create a comprehensive experiment summary."""
    summary = {
        'experiment_info': {
            'name': config.get('experiment', {}).get('name', 'unknown'),
            'date': str(np.datetime64('now')),
            'config': config
        },
        'model_info': {
            'teacher': results.get('teacher_info', {}),
            'student': results.get('student_info', {}),
        },
        'performance': {
            'teacher_accuracy': results.get('final_teacher_acc', 0.0),
            'student_supervised_accuracy': results.get('final_student_supervised_acc', 0.0),
            'student_kd_accuracy': results.get('final_student_kd_acc', 0.0),
            'improvement': results.get('improvement', 0.0)
        },
        'training_details': {
            'teacher_epochs': config.get('teacher', {}).get('epochs', 0),
            'student_epochs': config.get('student', {}).get('epochs', 0),
            'distillation_alpha': config.get('distillation', {}).get('alpha', 0.0),
            'distillation_temperature': config.get('distillation', {}).get('temperature', 0.0),
            'total_training_time': results.get('total_training_time', 0.0)
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary
