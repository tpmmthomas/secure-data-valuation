"""
Metrics for evaluating privacy attacks and model performance.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple

try:
    from pytorch_msssim import ssim as pytorch_ssim
    HAS_PYTORCH_MSSSIM = True
except ImportError:
    HAS_PYTORCH_MSSSIM = False


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate SSIM between two images.
    
    Args:
        img1, img2: Images as tensors (B, C, H, W) or (C, H, W)
    
    Returns:
        SSIM score (higher = more similar)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    if HAS_PYTORCH_MSSSIM:
        return pytorch_ssim(img1, img2, data_range=1.0).item()
    else:
        # Fallback to simple SSIM implementation
        return _simple_ssim(img1, img2).item()


def calculate_ssim_batch(img_batch1: torch.Tensor, img_batch2: torch.Tensor) -> torch.Tensor:
    """
    Calculate SSIM for batch of images.
    
    Args:
        img_batch1, img_batch2: Batches of images (B, C, H, W)
    
    Returns:
        Tensor of SSIM scores for each image pair
    """
    if HAS_PYTORCH_MSSSIM:
        # Calculate SSIM for each image pair in the batch
        ssim_scores = []
        for i in range(img_batch1.size(0)):
            ssim_score = pytorch_ssim(
                img_batch1[i:i+1], 
                img_batch2[i:i+1], 
                data_range=1.0
            )
            ssim_scores.append(ssim_score)
        return torch.stack(ssim_scores)
    else:
        # Fallback implementation
        return _simple_ssim_batch(img_batch1, img_batch2)


def _simple_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Simple SSIM implementation as fallback."""
    # Constants from SSIM paper
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def _simple_ssim_batch(img_batch1: torch.Tensor, img_batch2: torch.Tensor) -> torch.Tensor:
    """Simple SSIM for batch (fallback)."""
    ssim_scores = []
    for i in range(img_batch1.size(0)):
        ssim_score = _simple_ssim(img_batch1[i:i+1], img_batch2[i:i+1])
        ssim_scores.append(ssim_score)
    return torch.stack(ssim_scores)


def calculate_accuracy(model: torch.nn.Module, 
                      data_loader: DataLoader, 
                      device: torch.device) -> float:
    """Calculate model accuracy on given dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    
    return correct / total


def calculate_top_k_accuracy(model: torch.nn.Module, 
                           data_loader: DataLoader, 
                           device: torch.device,
                           k: int = 5) -> float:
    """Calculate top-k accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get top-k predictions
            _, pred = output.topk(k, dim=1, largest=True, sorted=True)
            correct += pred.eq(target.view(-1, 1).expand_as(pred)).sum().item()
            total += len(target)
    
    return correct / total


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Mean Squared Error between images."""
    return F.mse_loss(img1, img2).item()


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity).
    
    Note: This requires the lpips package to be installed.
    pip install lpips
    """
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex')  # or 'vgg', 'squeeze'
        with torch.no_grad():
            distance = loss_fn(img1, img2)
        return distance.item()
    except ImportError:
        raise ImportError("LPIPS calculation requires 'lpips' package. Install with: pip install lpips")


def privacy_success_rate(ssim_scores: torch.Tensor, threshold: float = 0.3) -> Tuple[float, float]:
    """
    Calculate privacy success rate based on SSIM threshold.
    
    Args:
        ssim_scores: Tensor of SSIM scores
        threshold: Threshold below which privacy is considered preserved
    
    Returns:
        privacy_rate: Fraction of samples with SSIM < threshold (privacy preserved)
        attack_rate: Fraction of samples with SSIM >= threshold (attack successful)
    """
    privacy_preserved = (ssim_scores < threshold).float()
    privacy_rate = privacy_preserved.mean().item()
    attack_rate = 1 - privacy_rate
    
    return privacy_rate, attack_rate


def accuracy_drop(baseline_acc: float, noisy_acc: float) -> float:
    """Calculate accuracy drop percentage."""
    return (baseline_acc - noisy_acc) / baseline_acc * 100


def speedup_estimation(boundary_layer: int, total_layers: int) -> float:
    """
    Estimate potential speedup based on boundary position.
    
    This is a simplified model. Actual speedup depends on:
    - MPC protocol overhead
    - Layer computational complexity
    - Communication costs
    """
    crypto_ratio = boundary_layer / total_layers
    # Assume MPC is 100x slower than plaintext
    mpc_overhead = 100
    
    # Speedup = Total_time_MPC / (Crypto_time_MPC + Clear_time_plain)
    total_time_mpc = total_layers * mpc_overhead
    crypto_time_mpc = boundary_layer * mpc_overhead
    clear_time_plain = (total_layers - boundary_layer) * 1
    
    speedup = total_time_mpc / (crypto_time_mpc + clear_time_plain)
    return speedup


def communication_reduction(boundary_layer: int, total_layers: int) -> float:
    """Estimate communication cost reduction."""
    # Simplified: clear layers require no communication
    crypto_ratio = boundary_layer / total_layers
    # Assume communication is proportional to number of crypto layers
    return 1 / crypto_ratio if crypto_ratio > 0 else float('inf')
