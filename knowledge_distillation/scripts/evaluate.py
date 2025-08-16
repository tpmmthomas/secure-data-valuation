#!/usr/bin/env python3
"""Evaluation script for trained models."""

import argparse
import torch
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from models import create_model
from datasets import create_dataset
from utils import get_device, accuracy
import torch.nn.functional as F


def evaluate_model(model, dataloader, device):
    """Evaluate a single model."""
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
            total_loss += loss.item()
    
    accuracy = 100.0 * total_correct / total_samples
    avg_loss = total_loss / len(dataloader)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': total_correct,
        'total': total_samples
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--experiment_dir", type=str, required=True,
                       help="Path to experiment directory")
    parser.add_argument("--config", type=str,
                       help="Path to config file (if not in experiment dir)")
    parser.add_argument("--teacher_weights", type=str,
                       help="Path to teacher weights")
    parser.add_argument("--student_weights", type=str,
                       help="Path to student weights")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    exp_dir = Path(args.experiment_dir)
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        # Look for config in experiment directory
        config_files = list(exp_dir.glob("*.yaml")) + list(exp_dir.glob("*.yml"))
        if config_files:
            config_path = config_files[0]
        else:
            print("No configuration file found. Please specify --config")
            return
    
    try:
        with open(config_path, 'r') as f:
            import yaml
            config_dict = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup dataset
    dataset_config = config_dict['dataset']
    dataset_loader = create_dataset(
        dataset_config['name'],
        data_dir=dataset_config.get('data_dir', './data'),
        batch_size=args.batch_size,
        num_workers=dataset_config.get('num_workers', 4),
        download=dataset_config.get('download', True)
    )
    
    _, test_loader, _ = dataset_loader.get_dataloaders()
    num_classes = dataset_loader.num_classes
    
    print(f"Dataset: {dataset_config['name']}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    
    results = {}
    
    # Evaluate teacher
    teacher_config = config_dict['teacher']
    teacher = create_model(
        'teacher',
        teacher_config['name'],
        num_classes,
        pretrained=teacher_config.get('pretrained', False)
    ).to(device)
    
    # Load teacher weights
    teacher_weights_path = args.teacher_weights
    if not teacher_weights_path:
        # Look in experiment directory
        teacher_files = list(exp_dir.glob("*teacher*.pt"))
        if teacher_files:
            teacher_weights_path = teacher_files[0]
    
    if teacher_weights_path and Path(teacher_weights_path).exists():
        teacher.load_state_dict(torch.load(teacher_weights_path, map_location=device))
        print(f"Loaded teacher weights from: {teacher_weights_path}")
        
        teacher_results = evaluate_model(teacher, test_loader, device)
        results['teacher'] = teacher_results
        print(f"Teacher accuracy: {teacher_results['accuracy']:.2f}%")
    else:
        print("Teacher weights not found")
    
    # Evaluate student
    student_config = config_dict['student']
    student = create_model(
        'student',
        student_config['name'],
        num_classes,
        dropout=student_config.get('dropout', 0.0)
    ).to(device)
    
    # Load student weights
    student_weights_path = args.student_weights
    if not student_weights_path:
        # Look in experiment directory
        student_files = list(exp_dir.glob("*student*.pt"))
        if student_files:
            student_weights_path = student_files[0]
    
    if student_weights_path and Path(student_weights_path).exists():
        student.load_state_dict(torch.load(student_weights_path, map_location=device))
        print(f"Loaded student weights from: {student_weights_path}")
        
        student_results = evaluate_model(student, test_loader, device)
        results['student'] = student_results
        print(f"Student accuracy: {student_results['accuracy']:.2f}%")
    else:
        print("Student weights not found")
    
    # Calculate improvement
    if 'teacher' in results and 'student' in results:
        # Note: This assumes we have a supervised baseline to compare against
        # For now, we'll just show the student performance
        print(f"\n=== Model Comparison ===")
        print(f"Teacher:  {results['teacher']['accuracy']:.2f}%")
        print(f"Student:  {results['student']['accuracy']:.2f}%")
        
        # Model size comparison
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        compression_ratio = teacher_params / student_params
        
        print(f"\n=== Model Size ===")
        print(f"Teacher parameters: {teacher_params:,}")
        print(f"Student parameters: {student_params:,}")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        results['compression_ratio'] = compression_ratio
    
    # Save results
    results_path = exp_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
