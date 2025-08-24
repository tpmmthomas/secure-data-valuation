import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from typing import Dict, Any, Optional, Tuple


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_distilled_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    learning_rate: float = 0.01,
    alpha: float = 0.7,
    temperature: float = 4.0,
    device: Optional[torch.device] = None,
    print_freq: int = 10,
    evaluate_freq: int = 10
) -> nn.Module:
    """
    Train a student model using knowledge distillation from a teacher model.
    
    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        train_loader: Training data loader
        test_loader: Optional test data loader for evaluation
        epochs: Number of training epochs
        learning_rate: Learning rate for student optimizer
        alpha: Weight for hard targets (1-alpha for soft targets)
        temperature: Temperature for softmax in KD loss
        device: Device to train on (auto-detected if None)
        print_freq: Frequency of progress printing
        evaluate_freq: Frequency of evaluation on test set
    
    Returns:
        Trained student model
    """
    if device is None:
        device = get_device()
    
    # Move models to device
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    
    # Set teacher to eval mode (frozen)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        student_model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kd = nn.KLDivLoss(reduction="batchmean")
    
    print(f"Starting knowledge distillation training on {device}")
    print(f"Epochs: {epochs}, LR: {learning_rate}, Alpha: {alpha}, Temperature: {temperature}")
    
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        train_metrics = _train_epoch_kd(
            teacher_model=teacher_model,
            student_model=student_model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_ce=criterion_ce,
            criterion_kd=criterion_kd,
            alpha=alpha,
            temperature=temperature,
            epoch=epoch,
            device=device,
            print_freq=print_freq
        )
        
        # Update learning rate
        scheduler.step()
        
        # Evaluation phase
        if test_loader is not None and (epoch + 1) % evaluate_freq == 0:
            test_metrics = _evaluate_model(student_model, test_loader, device)
            
            # Track best accuracy
            if test_metrics['acc'] > best_acc:
                best_acc = test_metrics['acc']
            
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(CE: {train_metrics['ce_loss']:.4f}, KD: {train_metrics['kd_loss']:.4f}) "
                  f"Train Acc: {train_metrics['acc']:.2f}% | "
                  f"Test Loss: {test_metrics['loss']:.4f} "
                  f"Test Acc: {test_metrics['acc']:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"(CE: {train_metrics['ce_loss']:.4f}, KD: {train_metrics['kd_loss']:.4f}) "
                  f"Train Acc: {train_metrics['acc']:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Final evaluation
    if test_loader is not None:
        final_metrics = _evaluate_model(student_model, test_loader, device)
        print(f"Final Test Accuracy: {final_metrics['acc']:.2f}%")
    
    return student_model


def _train_epoch_kd(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_ce: nn.Module,
    criterion_kd: nn.Module,
    alpha: float,
    temperature: float,
    epoch: int,
    device: torch.device,
    print_freq: int
) -> Dict[str, float]:
    """Train one epoch with knowledge distillation"""
    student_model.train()
    teacher_model.eval()
    
    # Metrics tracking
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CE Loss', ':.4e')
    kd_losses = AverageMeter('KD Loss', ':.4e')
    accuracies = AverageMeter('Acc@1', ':6.2f')
    
    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher_model(images)
        
        # Get student outputs
        student_outputs = student_model(images)
        
        # Hard label loss (cross-entropy)
        loss_ce = criterion_ce(student_outputs, targets)
        
        # Soft label loss (knowledge distillation)
        student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
        teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
        loss_kd = criterion_kd(student_log_probs, teacher_probs) * (temperature ** 2)
        
        # Combined loss
        loss = alpha * loss_ce + (1.0 - alpha) * loss_kd
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure accuracy and record losses
        acc = accuracy(student_outputs, targets)[0]
        losses.update(loss.item(), images.size(0))
        ce_losses.update(loss_ce.item(), images.size(0))
        kd_losses.update(loss_kd.item(), images.size(0))
        accuracies.update(acc.item(), images.size(0))
        
        # Print progress
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Acc@1 {accuracies.val:.3f} ({accuracies.avg:.3f})')
    
    return {
        "loss": losses.avg,
        "ce_loss": ce_losses.avg,
        "kd_loss": kd_losses.avg,
        "acc": accuracies.avg
    }


def _evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on test data"""
    model.eval()
    
    losses = AverageMeter('Loss', ':.4e')
    accuracies = AverageMeter('Acc@1', ':6.2f')
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc = accuracy(outputs, targets)[0]
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc.item(), images.size(0))
    
    return {"loss": losses.avg, "acc": accuracies.avg}
