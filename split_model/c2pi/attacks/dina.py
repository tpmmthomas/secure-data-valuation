"""
DINA: Distillation-based Inverse Network Attack

Enhanced IDPA with distillation points for improved inversion capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

from ..utils.metrics import calculate_ssim_batch
from .utils import AttackResult


class ResBlock(nn.Module):
    """Basic residual block for DINA inverse network."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class InverseBlock(nn.Module):
    """Basic inverse block: ResBlock + Dilated Conv + Optional Upsample."""
    
    def __init__(self, c_in: int, c_out: int, upsample_to: Optional[Tuple[int, int]] = None, dilation: int = 2):
        super().__init__()
        self.res = ResBlock(c_in)
        self.dil = nn.Conv2d(c_in, c_out, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(c_out)
        self.upsample_to = upsample_to
    
    def forward(self, x):
        x = self.res(x)
        x = F.relu(self.bn(self.dil(x)))
        if self.upsample_to is not None:
            x = F.interpolate(x, size=self.upsample_to, mode="nearest")
        return x


class DINAInverseNetwork(nn.Module):
    """
    DINA inverse network with distillation points.
    
    Architecture follows Figure 3 in the paper:
    - Chain of inverse blocks
    - Each block corresponds to a sub-block in the target model
    - Distillation points provide intermediate supervision
    """
    
    def __init__(self, 
                 channel_path: List[int], 
                 spatial_path: List[Tuple[int, int]], 
                 img_size: Tuple[int, int]):
        super().__init__()
        
        assert len(channel_path) == len(spatial_path)
        
        # Build inverse blocks
        blocks = []
        for k in range(len(channel_path) - 1):
            c_in, c_out = channel_path[k], channel_path[k + 1]
            up_to = spatial_path[k + 1] if spatial_path[k + 1] != spatial_path[k] else None
            blocks.append(InverseBlock(c_in, c_out, upsample_to=up_to, dilation=2))
        
        self.blocks = nn.ModuleList(blocks)
        
        # Final image reconstruction head
        self.img_head = nn.Sequential(
            nn.Conv2d(channel_path[-1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )
        self.img_size = img_size
    
    def forward(self, z):
        """
        Forward pass through inverse network.
        
        Returns:
            reconstructed_img: Final reconstructed image
            intermediate_features: List of intermediate features for distillation
        """
        intermediate_features = []
        
        for block in self.blocks:
            intermediate_features.append(z)
            z = block(z)
        
        # Final image reconstruction
        out = self.img_head(z)
        out = F.interpolate(out, size=self.img_size, mode="bilinear", align_corners=False)
        
        return out, intermediate_features


class DINAAttack:
    """
    DINA (Distillation-based Inverse Network Attack) implementation.
    
    Enhanced IDPA that uses distillation points for better inversion.
    """
    
    def __init__(self, target_model: nn.Module, config, split_layer: int):
        self.target_model = target_model
        self.config = config
        self.split_layer = split_layer
        self.device = torch.device(config.device)
        
        # Build inverse network
        self.inverse_net = None
        self.distillation_points = []
        self.loss_coefficients = []
        
        self._build_inverse_network()
    
    def _build_inverse_network(self):
        """Build DINA inverse network based on target model structure."""
        # Get sub-blocks and distillation points
        subblocks, taps = self._get_subblocks_and_taps()
        
        # Store distillation points (tap points between sub-blocks)
        self.distillation_points = taps
        
        if self.config.verbose:
            print(f"    Split layer: {self.split_layer}")
            print(f"    Distillation points: {self.distillation_points}")
            print(f"    Sub-blocks: {subblocks}")
        
        # Probe shapes by running example input
        channel_path, spatial_path = self._probe_network_shapes()
        
        if self.config.verbose:
            print(f"    Channel path: {channel_path}")
            print(f"    Spatial path: {spatial_path}")
        
        # Create inverse network
        self.inverse_net = DINAInverseNetwork(
            channel_path=channel_path,
            spatial_path=spatial_path,
            img_size=(self.config.img_size, self.config.img_size)
        ).to(self.device)
        
        # Set up loss coefficients: α₀=1, α₁=3, αⱼ=2×αⱼ₋₁
        self._setup_loss_coefficients(len(taps))
        
        if self.config.verbose:
            print(f"    Loss coefficients: {self.loss_coefficients}")
    
    def _get_subblocks_and_taps(self) -> Tuple[List[List[int]], List[int]]:
        """
        Partition layers before split into sub-blocks ending with ReLU.
        Returns sub-blocks and tap points for distillation.
        """
        if not hasattr(self.target_model, 'features'):
            raise NotImplementedError("DINA currently supports VGG-style models with .features")
        
        subblocks = []
        current_block = []
        taps = []
        
        for i in range(self.split_layer + 1):
            current_block.append(i)
            
            # Check if this layer is ReLU
            if isinstance(self.target_model.features[i], nn.ReLU):
                subblocks.append(current_block)
                taps.append(i)
                current_block = []
        
        # Handle remaining layers if split doesn't end on ReLU
        if current_block:
            subblocks.append(current_block)
        
        return subblocks, taps
    
    def _probe_network_shapes(self) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Probe network to get channel and spatial dimensions at each layer."""
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.config.img_size, self.config.img_size).to(self.device)
        
        # Collect shapes at distillation points and split layer
        shapes_at_points = []
        
        # Forward through features up to split layer
        x = dummy_input
        for i, layer in enumerate(self.target_model.features):
            x = layer(x)
            # Collect shapes at distillation points and split layer
            if i in self.distillation_points or i == self.split_layer:
                shapes_at_points.append((x.size(1), (x.size(2), x.size(3))))
        
        # Build paths for inverse network (from split back to image)
        # Start from split layer activation
        channel_path = [shapes_at_points[-1][0]]  # Split layer channels
        spatial_path = [shapes_at_points[-1][1]]  # Split layer spatial dims
        
        # Add distillation points in reverse order (going backwards)
        for i in range(len(shapes_at_points) - 2, -1, -1):
            channel_path.append(shapes_at_points[i][0])
            spatial_path.append(shapes_at_points[i][1])
        
        # Finally add input image dimensions (target for reconstruction)
        channel_path.append(3)
        spatial_path.append((self.config.img_size, self.config.img_size))
        
        return channel_path, spatial_path
    
    def _setup_loss_coefficients(self, num_distillation_points: int):
        """Set up loss coefficients: α₀=1, α₁=3, αⱼ=2×αⱼ₋₁ (j≥2)."""
        if self.config.alpha_base is not None:
            self.loss_coefficients = self.config.alpha_base[:num_distillation_points + 1]
        else:
            # Default from paper
            coeffs = [1.0]  # α₀ = 1
            if num_distillation_points > 0:
                coeffs.append(3.0)  # α₁ = 3
                for j in range(2, num_distillation_points + 1):
                    coeffs.append(2 * coeffs[j - 1])  # αⱼ = 2×αⱼ₋₁
            self.loss_coefficients = coeffs
    
    def train(self, train_loader: DataLoader):
        """Train DINA inverse network."""
        if self.inverse_net is None:
            raise ValueError("Inverse network not built")
        
        self.inverse_net.train()
        optimizer = torch.optim.SGD(
            self.inverse_net.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        for epoch in range(self.config.dina_epochs):
            total_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"DINA Epoch {epoch+1}/{self.config.dina_epochs}")
            
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(self.device)
                optimizer.zero_grad()
                
                # Forward through target model to get activations and distillation points
                target_activations, distillation_features = self._get_target_features(data)
                
                # Forward through inverse network
                reconstructed, inverse_features = self.inverse_net(target_activations)
                
                # Calculate DINA loss (Equation 1)
                loss = self._calculate_dina_loss(
                    data, reconstructed, 
                    distillation_features, inverse_features
                )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / num_batches
            if self.config.verbose:
                print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    def _get_target_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get activations at split layer and distillation points."""
        features = []
        
        # Forward through target model
        for i, layer in enumerate(self.target_model.features):
            x = layer(x)
            if i in self.distillation_points or i == self.split_layer:
                features.append(x.detach())
        
        # Split activation is the last one
        split_activation = features[-1]
        distillation_features = features[:-1]
        
        return split_activation, distillation_features
    
    def _calculate_dina_loss(self, 
                           original_img: torch.Tensor, 
                           reconstructed_img: torch.Tensor,
                           target_distillation: List[torch.Tensor],
                           inverse_distillation: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculate DINA loss (Equation 1 in paper):
        L_DINA = Σⱼ αⱼ||Dⱼ - Iⱼ||₂² + α₀||x - x̂||₂²
        
        where:
        - Dⱼ: feature map at distillation point j in target model
        - Iⱼ: input of basic inverse block j in DINA model  
        - αⱼ: monotonically increasing coefficients
        """
        # Pixel reconstruction loss (second term): α₀||x - x̂||₂²
        pixel_loss = F.mse_loss(reconstructed_img, original_img)
        total_loss = self.loss_coefficients[0] * pixel_loss
        
        # Distillation losses (first term): Σⱼ αⱼ||Dⱼ - Iⱼ||₂²
        # Reverse target_distillation to match inverse order
        reversed_target = list(reversed(target_distillation))
        
        # Only compute distillation losses for features with matching dimensions
        for j in range(len(inverse_distillation)):
            if j < len(reversed_target) and j + 1 < len(self.loss_coefficients):
                target_feat = reversed_target[j]
                inverse_feat = inverse_distillation[j]
                
                # Check if dimensions match for distillation loss
                if (target_feat.shape[1] == inverse_feat.shape[1] and 
                    target_feat.shape[2] == inverse_feat.shape[2] and
                    target_feat.shape[3] == inverse_feat.shape[3]):
                    # Dⱼ: feature at distillation point j (reversed target features)
                    # Iⱼ: input to basic inverse block j (inverse_distillation[j])
                    distill_loss = F.mse_loss(inverse_feat, target_feat)
                    total_loss += self.loss_coefficients[j + 1] * distill_loss
                elif self.config.verbose:
                    print(f"    Skipping distillation loss {j}: shape mismatch "
                          f"target={target_feat.shape} vs inverse={inverse_feat.shape}")
        
        return total_loss
    
    def attack(self, activations: torch.Tensor) -> torch.Tensor:
        """Run attack to reconstruct input from intermediate activations."""
        self.inverse_net.eval()
        with torch.no_grad():
            reconstructed, _ = self.inverse_net(activations)
        return reconstructed
    
    def evaluate_ssim(self, val_loader: DataLoader) -> float:
        """Evaluate attack success using SSIM metric."""
        self.inverse_net.eval()
        total_ssim = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                
                # Get split layer activations
                activations, _ = self._get_target_features(data)
                
                # Reconstruct
                reconstructed = self.attack(activations)
                
                # Calculate SSIM
                ssim_scores = calculate_ssim_batch(reconstructed, data)
                total_ssim += ssim_scores.sum().item()
                num_samples += len(data)
        
        return total_ssim / num_samples
    
    def evaluate_detailed(self, val_loader: DataLoader) -> AttackResult:
        """Detailed evaluation with multiple metrics."""
        self.inverse_net.eval()
        ssim_scores = []
        mse_scores = []
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                
                # Get split layer activations
                activations, _ = self._get_target_features(data)
                
                # Reconstruct
                reconstructed = self.attack(activations)
                
                # Calculate metrics
                ssim_batch = calculate_ssim_batch(reconstructed, data)
                mse_batch = F.mse_loss(reconstructed, data, reduction='none').mean(dim=[1,2,3])
                
                ssim_scores.extend(ssim_batch.cpu().numpy())
                mse_scores.extend(mse_batch.cpu().numpy())
        
        return AttackResult(
            ssim_scores=np.array(ssim_scores),
            mse_scores=np.array(mse_scores),
            avg_ssim=np.mean(ssim_scores),
            avg_mse=np.mean(mse_scores),
            success_rate=np.mean(np.array(ssim_scores) >= self.config.ssim_threshold)
        )
