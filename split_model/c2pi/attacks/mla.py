"""
MLA (Model Layer-wise Attack) implementation for C2PI.

This implements the MLA attack mentioned in the C2PI paper as an alternative
to DINA for evaluating privacy at different boundary layers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from ..utils.metrics import calculate_ssim_batch


class MLAAttacker:
    """
    Model Layer-wise Attack for reconstructing inputs from intermediate features.
    
    This attack trains a generator to reconstruct original inputs from
    intermediate layer outputs, helping evaluate privacy leakage at different
    boundary points.
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 feature_shape: Tuple[int, ...],
                 device: torch.device = None):
        """
        Initialize MLA attacker.
        
        Args:
            input_shape: Shape of original input (C, H, W)
            feature_shape: Shape of intermediate features
            device: Device to run on
        """
        self.input_shape = input_shape
        self.feature_shape = feature_shape
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build generator network
        self.generator = self._build_generator().to(self.device)
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'loss': [],
            'ssim': []
        }
    
    def _build_generator(self) -> nn.Module:
        """Build generator network to reconstruct inputs from features."""
        
        # Calculate dimensions
        C_in, H_in, W_in = self.input_shape
        feature_dim = np.prod(self.feature_shape)
        
        # Simple fully connected generator for proof of concept
        if len(self.feature_shape) == 1:  # FC features
            generator = nn.Sequential(
                nn.Linear(self.feature_shape[0], 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, np.prod(self.input_shape)),
                nn.Sigmoid(),
                nn.Unflatten(1, self.input_shape)
            )
        else:  # Convolutional features
            generator = self._build_conv_generator()
        
        return generator
    
    def _build_conv_generator(self) -> nn.Module:
        """Build convolutional generator for spatial features."""
        
        C_in, H_in, W_in = self.input_shape
        
        if len(self.feature_shape) == 3:  # (C, H, W)
            feat_c, feat_h, feat_w = self.feature_shape
        else:  # Flattened conv features
            feat_c, feat_h, feat_w = 256, 4, 4  # Assume some spatial structure
        
        # Transpose convolution layers to upsample
        layers = []
        
        # Initial projection if needed
        if feat_h * feat_w != H_in * W_in:
            layers.extend([
                nn.ConvTranspose2d(feat_c, 256, 4, 2, 1),  # Upsample
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ])
            feat_c = 256
            feat_h *= 2
            feat_w *= 2
        
        # Continue upsampling until we reach target size
        while feat_h < H_in or feat_w < W_in:
            out_channels = max(feat_c // 2, C_in)
            layers.extend([
                nn.ConvTranspose2d(feat_c, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            feat_c = out_channels
            feat_h *= 2
            feat_w *= 2
        
        # Final layer to match input channels
        layers.extend([
            nn.Conv2d(feat_c, C_in, 3, 1, 1),
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def train_attack(self,
                    feature_loader: DataLoader,
                    target_loader: DataLoader,
                    epochs: int = 50,
                    lr: float = 1e-3) -> Dict[str, List[float]]:
        """
        Train the MLA attack.
        
        Args:
            feature_loader: DataLoader providing intermediate features
            target_loader: DataLoader providing target images
            epochs: Number of training epochs
            lr: Learning rate
        
        Returns:
            Training history dictionary
        """
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        self.generator.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_ssim = 0.0
            num_batches = 0
            
            # Use tqdm for progress bar
            pbar = tqdm(zip(feature_loader, target_loader), 
                       desc=f'Epoch {epoch+1}/{epochs}')
            
            for features, targets in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstructed = self.generator(features)
                
                # Calculate loss
                loss = self.criterion(reconstructed, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    ssim_scores = calculate_ssim_batch(reconstructed, targets)
                    avg_ssim = ssim_scores.mean().item()
                
                epoch_loss += loss.item()
                epoch_ssim += avg_ssim
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'SSIM': f'{avg_ssim:.4f}'
                })
            
            # Step scheduler
            scheduler.step()
            
            # Record epoch metrics
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_ssim = epoch_ssim / num_batches
            
            self.training_history['loss'].append(avg_epoch_loss)
            self.training_history['ssim'].append(avg_epoch_ssim)
            
            print(f'Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, SSIM={avg_epoch_ssim:.4f}')
        
        return self.training_history
    
    def evaluate_attack(self,
                       feature_loader: DataLoader,
                       target_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the trained attack.
        
        Args:
            feature_loader: DataLoader providing intermediate features
            target_loader: DataLoader providing target images
        
        Returns:
            Evaluation metrics
        """
        
        self.generator.eval()
        
        total_loss = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        all_ssim_scores = []
        
        with torch.no_grad():
            for features, targets in zip(feature_loader, target_loader):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Generate reconstructions
                reconstructed = self.generator(features)
                
                # Calculate metrics
                loss = self.criterion(reconstructed, targets)
                ssim_scores = calculate_ssim_batch(reconstructed, targets)
                
                total_loss += loss.item() * len(targets)
                total_ssim += ssim_scores.sum().item()
                num_samples += len(targets)
                
                all_ssim_scores.extend(ssim_scores.cpu().numpy().tolist())
        
        # Calculate final metrics
        avg_loss = total_loss / num_samples
        avg_ssim = total_ssim / num_samples
        
        # Privacy analysis
        ssim_array = np.array(all_ssim_scores)
        privacy_threshold = 0.3
        privacy_preserved = (ssim_array < privacy_threshold).mean()
        attack_success = (ssim_array >= privacy_threshold).mean()
        
        return {
            'mse_loss': avg_loss,
            'avg_ssim': avg_ssim,
            'max_ssim': ssim_array.max(),
            'min_ssim': ssim_array.min(),
            'std_ssim': ssim_array.std(),
            'privacy_preserved': privacy_preserved,
            'attack_success': attack_success,
            'num_samples': num_samples
        }
    
    def attack_samples(self, features: torch.Tensor) -> torch.Tensor:
        """
        Perform attack on specific feature samples.
        
        Args:
            features: Input features to attack
        
        Returns:
            Reconstructed images
        """
        self.generator.eval()
        
        with torch.no_grad():
            features = features.to(self.device)
            reconstructed = self.generator(features)
        
        return reconstructed
    
    def save_model(self, path: str):
        """Save the trained generator model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'input_shape': self.input_shape,
            'feature_shape': self.feature_shape,
            'training_history': self.training_history
        }, path)
    
    def load_model(self, path: str):
        """Load a trained generator model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.training_history = checkpoint.get('training_history', {})


def extract_features(model: nn.Module, 
                    data_loader: DataLoader,
                    layer_idx: int,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract intermediate features from a model at specified layer.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for input data
        layer_idx: Index of layer to extract features from
        device: Device to run on
    
    Returns:
        features, targets: Extracted features and corresponding targets
    """
    
    model.eval()
    
    # Hook to capture features
    features_list = []
    
    def hook_fn(module, input, output):
        features_list.append(output.detach().cpu())
    
    # Register hook at specified layer
    if hasattr(model, 'features'):
        # For models with .features (like VGG)
        hook = model.features[layer_idx].register_forward_hook(hook_fn)
    else:
        # For sequential models
        layers = list(model.children())
        hook = layers[layer_idx].register_forward_hook(hook_fn)
    
    all_targets = []
    
    try:
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(device)
                _ = model(data)  # Forward pass triggers hook
                all_targets.append(targets)
        
        # Concatenate all features and targets
        all_features = torch.cat(features_list, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
    finally:
        # Remove hook
        hook.remove()
    
    return all_features, all_targets
