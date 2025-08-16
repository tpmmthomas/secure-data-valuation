#!/usr/bin/env python3
"""Main script for knowledge distillation training."""

import argparse
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, merge_configs, KnowledgeDistillationConfig
from trainer import KnowledgeDistillationTrainer
from utils import create_experiment_summary, plot_training_curves, plot_model_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation Training Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the configuration file"
    )
    
    # Override options
    parser.add_argument("--teacher_model", type=str, help="Teacher model name")
    parser.add_argument("--student_model", type=str, help="Student model name")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--epochs_teacher", type=int, help="Number of teacher epochs")
    parser.add_argument("--epochs_student", type=int, help="Number of student epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr_teacher", type=float, help="Teacher learning rate")
    parser.add_argument("--lr_student", type=float, help="Student learning rate")
    parser.add_argument("--alpha", type=float, help="Distillation alpha parameter")
    parser.add_argument("--temperature", type=float, help="Distillation temperature")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    
    # Mode selection
    parser.add_argument("--teacher_only", action="store_true", help="Train teacher only")
    parser.add_argument("--student_only", action="store_true", help="Train student only (requires pretrained teacher)")
    parser.add_argument("--evaluate_only", action="store_true", help="Evaluate models only")
    
    return parser.parse_args()


def create_override_config(args):
    """Create configuration overrides from command line arguments."""
    overrides = {}
    
    # Model overrides
    if args.teacher_model:
        overrides['teacher'] = overrides.get('teacher', {})
        overrides['teacher']['name'] = args.teacher_model
    
    if args.student_model:
        overrides['student'] = overrides.get('student', {})
        overrides['student']['name'] = args.student_model
    
    # Dataset overrides
    if args.dataset:
        overrides['dataset'] = overrides.get('dataset', {})
        overrides['dataset']['name'] = args.dataset
    
    if args.batch_size:
        overrides['dataset'] = overrides.get('dataset', {})
        overrides['dataset']['batch_size'] = args.batch_size
    
    # Training overrides
    if args.epochs_teacher:
        overrides['teacher'] = overrides.get('teacher', {})
        overrides['teacher']['epochs'] = args.epochs_teacher
    
    if args.epochs_student:
        overrides['student'] = overrides.get('student', {})
        overrides['student']['epochs'] = args.epochs_student
    
    if args.lr_teacher:
        overrides['teacher'] = overrides.get('teacher', {})
        overrides['teacher']['lr'] = args.lr_teacher
    
    if args.lr_student:
        overrides['student'] = overrides.get('student', {})
        overrides['student']['lr'] = args.lr_student
    
    # Distillation overrides
    if args.alpha:
        overrides['distillation'] = overrides.get('distillation', {})
        overrides['distillation']['alpha'] = args.alpha
    
    if args.temperature:
        overrides['distillation'] = overrides.get('distillation', {})
        overrides['distillation']['temperature'] = args.temperature
    
    # Experiment overrides
    if args.output_dir:
        overrides['experiment'] = overrides.get('experiment', {})
        overrides['experiment']['output_dir'] = args.output_dir
    
    if args.seed:
        overrides['experiment'] = overrides.get('experiment', {})
        overrides['experiment']['seed'] = args.seed
    
    # Logging overrides
    if args.use_wandb:
        overrides['logging'] = overrides.get('logging', {})
        overrides['logging']['use_wandb'] = True
    
    if args.wandb_project:
        overrides['logging'] = overrides.get('logging', {})
        overrides['logging']['wandb_project'] = args.wandb_project
    
    return overrides

def _config_to_dict(config):
    """Convert dataclass config to JSON-serializable dictionary."""
    import dataclasses
    
    def _convert_value(value):
        if dataclasses.is_dataclass(value):
            return {field.name: _convert_value(getattr(value, field.name)) 
                    for field in dataclasses.fields(value)}
        elif isinstance(value, (list, tuple)):
            return [_convert_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: _convert_value(v) for k, v in value.items()}
        else:
            return value
    
    return _convert_value(config)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Apply command line overrides
    overrides = create_override_config(args)
    if overrides:
        config = merge_configs(config, overrides)
        
    # Safety check: ensure config is a proper dataclass instance
    if isinstance(config, dict):
        print("Warning: Config is a dictionary, converting to dataclass...")
        from omegaconf import OmegaConf
        structured_conf = OmegaConf.structured(KnowledgeDistillationConfig)
        merged_conf = OmegaConf.merge(structured_conf, OmegaConf.create(config))
        config = OmegaConf.to_object(merged_conf)
    
    print(f"Starting experiment: {config.experiment.name}")
    print(f"Configuration loaded from: {args.config}")
    
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(config)
    
    try:
        if args.teacher_only:
            # Train teacher only
            print("Training teacher model only...")
            results = trainer.train_teacher()
            
        elif args.student_only:
            # Train student only (assumes teacher is already trained)
            print("Training student model only...")
            # Load teacher weights if available
            teacher_path = Path(config.experiment.output_dir) / f"teacher_{config.teacher.name}_final.pt"
            if teacher_path.exists():
                trainer.teacher.load_state_dict(torch.load(teacher_path, map_location=trainer.device))
                print(f"Loaded teacher weights from {teacher_path}")
            else:
                print("Warning: No teacher weights found. Training from scratch.")
            
            results = trainer.train_student_kd()
            
        elif args.evaluate_only:
            # Evaluate models only
            print("Evaluating models...")
            teacher_results = trainer._evaluate(trainer.teacher, trainer.test_loader)
            student_results = trainer._evaluate(trainer.student, trainer.test_loader)
            
            print(f"Teacher accuracy: {teacher_results['acc']:.2f}%")
            print(f"Student accuracy: {student_results['acc']:.2f}%")
            
        else:
            # Full training pipeline
            print("Starting full training pipeline...")
            results = trainer.train()
            
            # Generate visualizations
            output_dir = Path(config.experiment.output_dir)
            
            # Plot training curves
            if results.get("teacher") and results.get("student_kd"):
                plot_training_curves(
                    results,
                    save_path=output_dir / "training_curves.png"
                )
                print(f"Training curves saved to {output_dir / 'training_curves.png'}")
            
            # Plot model comparison
            if results.get("final"):
                teacher_acc = results["final"]["teacher_test"]["acc"]
                student_kd_acc = results["final"]["student_kd_test"]["acc"]
                student_sup_acc = 0.0
                
                if results.get("final", {}).get("student_supervised_results"):
                    student_sup_acc = results["final"]["student_supervised_results"]["final_metrics"]["acc"]
                
                plot_model_comparison(
                    teacher_acc, student_sup_acc, student_kd_acc,
                    save_path=output_dir / "model_comparison.png"
                )
                print(f"Model comparison saved to {output_dir / 'model_comparison.png'}")
            
            # Create experiment summary
            create_experiment_summary(
                _config_to_dict(config),
                results.get("final", {}),
                output_dir / "experiment_summary.json"
            )
            print(f"Experiment summary saved to {output_dir / 'experiment_summary.json'}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
