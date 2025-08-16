"""Training logic for knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from config import KnowledgeDistillationConfig
from models import create_model
from datasets import create_dataset
from utils import (
    set_seed, accuracy, AverageMeter, ProgressMeter, 
    create_optimizer, create_scheduler, save_checkpoint,
    get_device, setup_logging
)


class KnowledgeDistillationTrainer:
    """Main trainer class for knowledge distillation."""
    
    def __init__(self, config: KnowledgeDistillationConfig):
        self.config = config
        self.device = get_device()
        
        # Set random seed
        set_seed(config.experiment.seed)
        
        # Setup output directory
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(
            self.output_dir / "logs", 
            name=config.experiment.name
        )
        
        # Setup TensorBoard
        if config.logging.use_tensorboard:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")
        else:
            self.writer = None
        
        # Initialize components
        self._setup_data()
        self._setup_models()
        self._setup_training()
        
        # Training history
        self.history = {
            "teacher": [],
            "student_supervised": [],
            "student_kd": [],
            "final": {}
        }
    
    def _setup_data(self):
        """Setup datasets and data loaders."""
        self.logger.info(f"Setting up {self.config.dataset.name} dataset...")
        
        dataset_loader = create_dataset(
            self.config.dataset.name,
            data_dir=self.config.dataset.data_dir,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory,
            download=self.config.dataset.download,
            augmentation=getattr(self.config.dataset, 'augmentation', True),
            normalize=getattr(self.config.dataset, 'normalize', True),
            validation_split=getattr(self.config.dataset, 'validation_split', 0.0)
        )
        
        loaders = dataset_loader.get_dataloaders()
        self.train_loader = loaders[0]
        self.test_loader = loaders[1]
        self.val_loader = loaders[2] if len(loaders) > 2 else None
        
        # Update config with dataset info
        self.config.dataset.num_classes = dataset_loader.num_classes
        self.config.dataset.input_shape = dataset_loader.input_shape
        
        self.logger.info(f"Dataset info: {dataset_loader.get_dataset_info()}")
    
    def _setup_models(self):
        """Setup teacher and student models."""
        self.logger.info("Setting up models...")
        
        # Teacher model
        self.teacher = create_model(
            'teacher',
            self.config.teacher.name,
            self.config.dataset.num_classes,
            pretrained=self.config.teacher.pretrained,
            **self.config.teacher.model_params
        ).to(self.device)
        
        # Student model
        self.student = create_model(
            'student',
            self.config.student.name,
            self.config.dataset.num_classes,
            dropout=getattr(self.config.student, 'dropout', 0.0),
            **self.config.student.model_params
        ).to(self.device)
        
        # Log model info
        teacher_info = self.teacher.model_info
        student_info = self.student.model_info
        
        self.logger.info(f"Teacher model: {teacher_info}")
        self.logger.info(f"Student model: {student_info}")
    
    def _setup_training(self):
        """Setup optimizers and schedulers."""
        # Teacher optimizer and scheduler
        self.teacher_optimizer = create_optimizer(
            self.teacher,
            self.config.teacher.optimizer,
            self.config.teacher.lr,
            momentum=self.config.teacher.momentum,
            weight_decay=self.config.teacher.weight_decay
        )
        
        self.teacher_scheduler = create_scheduler(
            self.teacher_optimizer,
            self.config.teacher.scheduler,
            **self.config.teacher.scheduler_params
        )
        
        # Student optimizer and scheduler
        self.student_optimizer = create_optimizer(
            self.student,
            self.config.student.optimizer,
            self.config.student.lr,
            momentum=self.config.student.momentum,
            weight_decay=self.config.student.weight_decay
        )
        
        self.student_scheduler = create_scheduler(
            self.student_optimizer,
            self.config.student.scheduler,
            **self.config.student.scheduler_params
        )
    
    def train_teacher(self) -> Dict[str, Any]:
        """Train the teacher model."""
        self.logger.info(f"Training teacher for {self.config.teacher.epochs} epochs...")
        
        best_acc = 0.0
        
        for epoch in range(1, self.config.teacher.epochs + 1):
            # Train
            train_metrics = self._train_epoch_supervised(
                self.teacher, self.teacher_optimizer, epoch, "Teacher"
            )
            
            # Evaluate
            test_metrics = self._evaluate(self.teacher, self.test_loader)
            
            # Update scheduler
            if self.teacher_scheduler:
                self.teacher_scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, test_metrics, epoch, "teacher")
            
            # Save checkpoint
            is_best = test_metrics['acc'] > best_acc
            if is_best:
                best_acc = test_metrics['acc']
            
            if epoch % self.config.logging.save_interval == 0 or is_best:
                save_checkpoint(
                    self.teacher, self.teacher_optimizer, self.teacher_scheduler,
                    epoch, best_acc, self.config.__dict__,
                    self.output_dir / f"teacher_epoch_{epoch}.pt",
                    is_best=is_best
                )
            
            # Store history
            self.history["teacher"].append({
                "epoch": epoch,
                "train": train_metrics,
                "test": test_metrics
            })
            
            self.logger.info(
                f"[Teacher E{epoch}] train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_acc={test_metrics['acc']:.4f}"
            )
        
        # Save final model
        torch.save(
            self.teacher.state_dict(),
            self.output_dir / f"teacher_{self.config.teacher.name}_final.pt"
        )
        
        return {"best_acc": best_acc, "final_metrics": test_metrics}
    
    def train_student_supervised(self) -> Dict[str, Any]:
        """Train student with supervised learning (baseline)."""
        if not self.config.student.supervised_baseline:
            return {}
        
        self.logger.info(f"Training student (supervised) for {self.config.student.supervised_epochs} epochs...")
        
        best_acc = 0.0
        
        for epoch in range(1, self.config.student.supervised_epochs + 1):
            # Train
            train_metrics = self._train_epoch_supervised(
                self.student, self.student_optimizer, epoch, "Student-Sup"
            )
            
            # Evaluate
            test_metrics = self._evaluate(self.student, self.test_loader)
            
            # Log metrics
            self._log_metrics(train_metrics, test_metrics, epoch, "student_supervised")
            
            # Track best accuracy
            if test_metrics['acc'] > best_acc:
                best_acc = test_metrics['acc']
            
            # Store history
            self.history["student_supervised"].append({
                "epoch": epoch,
                "train": train_metrics,
                "test": test_metrics
            })
            
            self.logger.info(
                f"[Student-Sup E{epoch}] train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_acc={test_metrics['acc']:.4f}"
            )
        
        return {"best_acc": best_acc, "final_metrics": test_metrics}
    
    def train_student_kd(self) -> Dict[str, Any]:
        """Train student with knowledge distillation."""
        self.logger.info(
            f"Training student with KD for {self.config.student.epochs} epochs "
            f"(alpha={self.config.distillation.alpha}, T={self.config.distillation.temperature})..."
        )
        
        best_acc = 0.0
        
        for epoch in range(1, self.config.student.epochs + 1):
            # Train
            train_metrics = self._train_epoch_kd(epoch)
            
            # Evaluate
            test_metrics = self._evaluate(self.student, self.test_loader)
            
            # Update scheduler
            if self.student_scheduler:
                self.student_scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, test_metrics, epoch, "student_kd")
            
            # Save checkpoint
            is_best = test_metrics['acc'] > best_acc
            if is_best:
                best_acc = test_metrics['acc']
            
            if epoch % self.config.logging.save_interval == 0 or is_best:
                save_checkpoint(
                    self.student, self.student_optimizer, self.student_scheduler,
                    epoch, best_acc, self.config.__dict__,
                    self.output_dir / f"student_kd_epoch_{epoch}.pt",
                    is_best=is_best
                )
            
            # Store history
            self.history["student_kd"].append({
                "epoch": epoch,
                "train": train_metrics,
                "test": test_metrics
            })
            
            self.logger.info(
                f"[Student-KD E{epoch}] train_loss={train_metrics['loss']:.4f} "
                f"(CE={train_metrics.get('ce', 0):.4f}, KD={train_metrics.get('kd', 0):.4f}) "
                f"train_acc={train_metrics['acc']:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} "
                f"test_acc={test_metrics['acc']:.4f}"
            )
        
        # Save final model
        torch.save(
            self.student.state_dict(),
            self.output_dir / f"student_{self.config.student.name}_kd_final.pt"
        )
        
        return {"best_acc": best_acc, "final_metrics": test_metrics}
    
    def _train_epoch_supervised(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        epoch: int,
        prefix: str
    ) -> Dict[str, float]:
        """Train one epoch with supervised learning."""
        model.train()
        
        losses = AverageMeter('Loss', ':.4e')
        accuracies = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, accuracies],
            prefix=f"{prefix} Epoch: [{epoch}]"
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for i, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Measure accuracy and record loss
            acc = accuracy(outputs, targets)[0]
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))
            
            # Log progress
            if i % self.config.logging.log_interval == 0:
                progress.display(i)
        
        return {"loss": losses.avg, "acc": accuracies.avg}
    
    def _train_epoch_kd(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with knowledge distillation."""
        self.student.train()
        self.teacher.eval()
        
        losses = AverageMeter('Loss', ':.4e')
        ce_losses = AverageMeter('CE Loss', ':.4e')
        kd_losses = AverageMeter('KD Loss', ':.4e')
        accuracies = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [losses, ce_losses, kd_losses, accuracies],
            prefix=f"Student-KD Epoch: [{epoch}]"
        )
        
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction="batchmean")
        
        alpha = self.config.distillation.alpha
        temperature = self.config.distillation.temperature
        
        for i, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Get teacher outputs (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher(images)
            
            # Get student outputs
            student_outputs = self.student(images)
            
            # Hard label loss (cross-entropy)
            loss_ce = criterion_ce(student_outputs, targets)
            
            # Soft label loss (knowledge distillation)
            student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
            teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
            loss_kd = criterion_kd(student_log_probs, teacher_probs) * (temperature ** 2)
            
            # Combined loss
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd
            
            # Backward pass
            self.student_optimizer.zero_grad()
            loss.backward()
            self.student_optimizer.step()
            
            # Measure accuracy and record loss
            acc = accuracy(student_outputs, targets)[0]
            losses.update(loss.item(), images.size(0))
            ce_losses.update(loss_ce.item(), images.size(0))
            kd_losses.update(loss_kd.item(), images.size(0))
            accuracies.update(acc, images.size(0))
            
            # Log progress
            if i % self.config.logging.log_interval == 0:
                progress.display(i)
        
        return {
            "loss": losses.avg,
            "ce": ce_losses.avg,
            "kd": kd_losses.avg,
            "acc": accuracies.avg
        }
    
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data loader."""
        model.eval()
        
        losses = AverageMeter('Loss', ':.4e')
        accuracies = AverageMeter('Acc@1', ':6.2f')
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                acc = accuracy(outputs, targets)[0]
                losses.update(loss.item(), images.size(0))
                accuracies.update(acc, images.size(0))
        
        return {"loss": losses.avg, "acc": accuracies.avg}
    
    def _log_metrics(
        self, 
        train_metrics: Dict[str, float], 
        test_metrics: Dict[str, float],
        epoch: int,
        phase: str
    ):
        """Log metrics to TensorBoard and other loggers."""
        if self.writer:
            # Training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"{phase}/train_{key}", value, epoch)
            
            # Test metrics
            for key, value in test_metrics.items():
                self.writer.add_scalar(f"{phase}/test_{key}", value, epoch)
            
            # Learning rate
            if phase == "teacher" and self.teacher_scheduler:
                self.writer.add_scalar(f"{phase}/lr", self.teacher_scheduler.get_last_lr()[0], epoch)
            elif phase == "student_kd" and self.student_scheduler:
                self.writer.add_scalar(f"{phase}/lr", self.student_scheduler.get_last_lr()[0], epoch)
    
    def train(self) -> Dict[str, Any]:
        """Run complete training pipeline."""
        start_time = time.time()
        
        # Train teacher
        teacher_results = self.train_teacher()
        
        # Train student (supervised baseline)
        student_sup_results = self.train_student_supervised()
        
        # Train student with knowledge distillation
        student_kd_results = self.train_student_kd()
        
        # Final evaluation
        final_teacher = self._evaluate(self.teacher, self.test_loader)
        final_student = self._evaluate(self.student, self.test_loader)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Store final results
        self.history["final"] = {
            "teacher_test": final_teacher,
            "student_kd_test": final_student,
            "teacher_results": teacher_results,
            "student_supervised_results": student_sup_results,
            "student_kd_results": student_kd_results,
            "total_training_time": total_time,
            "config": self.config.__dict__
        }
        
        # Save history
        import json
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Log final results
        self.logger.info("\n=== Final Results ===")
        self.logger.info(f"Teacher: loss={final_teacher['loss']:.4f}, acc={final_teacher['acc']:.4f}")
        self.logger.info(f"Student (KD): loss={final_student['loss']:.4f}, acc={final_student['acc']:.4f}")
        
        if student_sup_results:
            improvement = final_student['acc'] - student_sup_results['final_metrics']['acc']
            self.logger.info(f"KD Improvement: +{improvement:.2f}%")
        
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        self.logger.info(f"Artifacts saved to: {self.output_dir}")
        
        if self.writer:
            self.writer.close()
        
        return self.history
