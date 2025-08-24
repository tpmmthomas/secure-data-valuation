"""
DINA: Distillation-based Inverse Network Attack (updated)

Changes:
1) Distillation taps at pre-ReLU conv (middle points between sub-blocks).
2A) Distillation compares block *outputs* to teacher taps (1-to-1 alignment).
3) Explicit tap ordering: closest-to-split → earliest; α increases along that order.
4) SSIM is computed on de-normalized images in [0,1] via self.denorm().
5) Support for models without .features - automatically flattens any model architecture.

Model Support:
- VGG-style models with .features attribute
- ResNet, DenseNet, and other models via recursive flattening of .children()
- Custom architectures - automatically detects Conv2d and ReLU layers
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
    DINA inverse network with distillation at midpoints (pre-ReLU convs).

    We build blocks so that each block *output* aligns with one teacher tap.
    """
    def __init__(self,
                 channel_path: List[int],
                 spatial_path: List[Tuple[int, int]],
                 img_size: Tuple[int, int],
                 img_channels: int = 3):
        super().__init__()
        assert len(channel_path) == len(spatial_path)
        # IMPORTANT: channel_path/spatial_path are [split] + [tap_N, ..., tap_0]
        # -> number of blocks = len(path) - 1 == number of taps
        blocks = []
        for k in range(len(channel_path) - 1):
            c_in, c_out = channel_path[k], channel_path[k + 1]
            up_to = spatial_path[k + 1] if spatial_path[k + 1] != spatial_path[k] else None
            blocks.append(InverseBlock(c_in, c_out, upsample_to=up_to, dilation=2))
        self.blocks = nn.ModuleList(blocks)

        # Final image head maps from earliest-tap channels to correct number of image channels
        self.img_head = nn.Sequential(
            nn.Conv2d(channel_path[-1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, img_channels, 1)  # Use img_channels instead of hardcoded 3
        )
        self.img_size = img_size

    def forward(self, z):
        """
        Returns:
            reconstructed_img: Final reconstructed image
            student_block_outputs: List of block outputs for distillation (1-to-1 with taps)
        """
        student_block_outputs = []
        for block in self.blocks:
            z = block(z)
            student_block_outputs.append(z)  # <-- OUTPUTS (2A)
        out = self.img_head(z)
        out = F.interpolate(out, size=self.img_size, mode="bilinear", align_corners=False)
        return out, student_block_outputs


class DINAAttack:
    """
    DINA (Distillation-based Inverse Network Attack).
    Implements:
      (1) taps at pre-ReLU convs,
      (2A) distillation on block outputs (no skipping),
      (3) explicit ordering + α schedule,
      (4) SSIM on de-normalized images.
    """
    def __init__(self, target_model: nn.Module, config, split_layer: int):
        self.target_model = target_model
        self.config = config
        self.split_layer = split_layer
        self.device = torch.device(config.device)

        # Auto-detect image channels from config or use defaults
        img_channels = getattr(config, "img_channels", 3)
        
        # Handle different normalization for different channel counts
        if img_channels == 1:
            # MNIST-style grayscale
            default_mean = [0.5]
            default_std = [0.5]
        elif img_channels == 3:
            # ImageNet/CIFAR-style RGB
            default_mean = [0.485, 0.456, 0.406]
            default_std = [0.229, 0.224, 0.225]
        else:
            # Unknown format - use neutral values
            default_mean = [0.5] * img_channels
            default_std = [0.5] * img_channels

        # Get normalization parameters from config or use defaults
        norm_mean = getattr(config, "norm_mean", default_mean)
        norm_std = getattr(config, "norm_std", default_std)
        
        # Ensure we have the right number of channels
        if len(norm_mean) != img_channels:
            if len(norm_mean) == 1:
                norm_mean = norm_mean * img_channels
            elif len(norm_mean) == 3 and img_channels == 1:
                norm_mean = [sum(norm_mean) / 3]  # Convert RGB to grayscale
            else:
                norm_mean = default_mean
                
        if len(norm_std) != img_channels:
            if len(norm_std) == 1:
                norm_std = norm_std * img_channels
            elif len(norm_std) == 3 and img_channels == 1:
                norm_std = [sum(norm_std) / 3]  # Convert RGB to grayscale
            else:
                norm_std = default_std

        self.norm_mean = torch.tensor(
            norm_mean, dtype=torch.float32, device=self.device
        ).view(1, img_channels, 1, 1)
        self.norm_std = torch.tensor(
            norm_std, dtype=torch.float32, device=self.device
        ).view(1, img_channels, 1, 1)
        
        self.img_channels = img_channels
        self.inverse_net = None
        self.distillation_points: List[int] = []
        self.loss_coefficients: List[float] = []

        self._build_inverse_network()

    # --------- Utility: denormalize to [0,1] for SSIM (4) ----------
    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Invert dataset normalization and clamp to [0,1] for SSIM."""
        return torch.clamp(x * self.norm_std + self.norm_mean, 0.0, 1.0)

    # ---------------- Internal builders ----------------
    def _build_inverse_network(self):
        subblocks, taps = self._get_subblocks_and_taps()  # taps are pre-ReLU conv indices
        # Order taps from closest-to-split -> earliest (3)
        taps = sorted([t for t in taps if t <= self.split_layer], reverse=True)
        self.distillation_points = taps

        if self.config.verbose:
            print(f"    Split layer: {self.split_layer}")
            print(f"    Distillation taps (pre-ReLU conv), ordered near→far: {self.distillation_points}")
            print(f"    Sub-blocks: {subblocks}")

        channel_path, spatial_path = self._probe_network_shapes()

        if self.config.verbose:
            print(f"    Channel path: {channel_path}")
            print(f"    Spatial path: {spatial_path}")

        self.inverse_net = DINAInverseNetwork(
            channel_path=channel_path,
            spatial_path=spatial_path,
            img_size=(self.config.img_size, self.config.img_size),
            img_channels=self.img_channels
        ).to(self.device)

        self._setup_loss_coefficients(len(self.distillation_points))  # α schedule (3)

        if self.config.verbose:
            print(f"    Loss coefficients: {self.loss_coefficients}")

    def _flatten_model_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        Flatten a model into a sequential list of layers.
        Supports both .features attribute and general models via .children().
        """
        if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
            # VGG-style models with .features
            return list(model.features)
        else:
            # General models - flatten recursively
            layers = []
            
            def _collect_layers(module: nn.Module):
                # Get immediate children
                children = list(module.children())
                if not children:
                    # Leaf module - add it directly
                    layers.append(module)
                else:
                    # Has children - recurse
                    for child in children:
                        _collect_layers(child)
            
            _collect_layers(model)
            return layers

    def _get_subblocks_and_taps(self) -> Tuple[List[List[int]], List[int]]:
        """
        Partition layers [0..split_layer] into sub-blocks ending with ReLU.
        Distillation tap for each sub-block is set to the *pre-ReLU conv* (middle point).
        
        If no explicit ReLU layers are found in the model, assumes every Conv2d
        has an implicit ReLU activation and creates blocks accordingly.
        """
        # Get flattened layer list
        self._model_layers = self._flatten_model_layers(self.target_model)
        
        if len(self._model_layers) <= self.split_layer:
            raise ValueError(f"Split layer {self.split_layer} exceeds model depth {len(self._model_layers)}")

        # Check if we have any explicit ReLU layers in the split range
        has_explicit_relu = any(isinstance(self._model_layers[i], nn.ReLU) 
                               for i in range(self.split_layer + 1))

        subblocks: List[List[int]] = []
        current_block: List[int] = []
        taps: List[int] = []

        def find_prev_conv(idx: int) -> Optional[int]:
            j = idx - 1
            while j >= 0:
                if isinstance(self._model_layers[j], nn.Conv2d):
                    return j
                j -= 1
            return None

        if has_explicit_relu:
            # Original logic: use explicit ReLU layers to define blocks
            for i in range(self.split_layer + 1):
                current_block.append(i)
                if isinstance(self._model_layers[i], nn.ReLU):
                    subblocks.append(current_block)
                    # tap = conv immediately before this ReLU (middle point)
                    t = find_prev_conv(i)
                    if t is not None:
                        taps.append(t)
                    current_block = []

            # If split doesn't end on ReLU, close the last fragment (no additional tap)
            if current_block:
                subblocks.append(current_block)
        else:
            # Fallback: assume every Conv2d has an implicit ReLU after it
            conv_indices = [i for i in range(self.split_layer + 1) 
                           if isinstance(self._model_layers[i], nn.Conv2d)]
            
            if not conv_indices:
                # No Conv2d layers found - create one block with all layers
                subblocks.append(list(range(self.split_layer + 1)))
            else:
                # Create blocks based on Conv2d positions
                start_idx = 0
                for conv_idx in conv_indices:
                    # Block goes from start_idx to conv_idx (inclusive)
                    block = list(range(start_idx, conv_idx + 1))
                    subblocks.append(block)
                    taps.append(conv_idx)  # The Conv2d itself is the tap point
                    start_idx = conv_idx + 1
                
                # Add remaining layers after the last Conv2d (if any)
                if start_idx <= self.split_layer:
                    remaining_block = list(range(start_idx, self.split_layer + 1))
                    subblocks.append(remaining_block)

        return subblocks, taps

    def _probe_network_shapes(self) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Build channel/spatial paths so that:
          path = [split] + [tap_near, tap_next, ..., tap_far]
        This yields #blocks == #taps and each block output matches one teacher tap.
        """
        # Use the flattened layers (should be already set by _get_subblocks_and_taps)
        if not hasattr(self, '_model_layers'):
            self._model_layers = self._flatten_model_layers(self.target_model)
            
        # Auto-detect input channels from model or use config
        try:
            # Try to get input channels from first layer
            first_layer = self._model_layers[0]
            if isinstance(first_layer, nn.Conv2d):
                input_channels = first_layer.in_channels
            else:
                input_channels = self.img_channels
        except:
            input_channels = self.img_channels
            
        dummy = torch.randn(1, input_channels, self.config.img_size, self.config.img_size, device=self.device)

        # Collect split activation + taps (already ordered near→far)
        shapes: Dict[str, Tuple[int, Tuple[int,int]]] = {}

        x = dummy
        for i, layer in enumerate(self._model_layers):
            if i > self.split_layer:
                break
            x = layer(x)
            if i == self.split_layer:
                shapes["split"] = (x.size(1), (x.size(2), x.size(3)))
            if i in self.distillation_points:
                shapes[f"tap_{i}"] = (x.size(1), (x.size(2), x.size(3)))

        # Build paths
        ch = [shapes["split"][0]]
        sp = [shapes["split"][1]]
        for i in self.distillation_points:  # already near→far
            ci, si = shapes[f"tap_{i}"]
            ch.append(ci)
            sp.append(si)

        # DO NOT append (3, img) here; img_head handles RGB mapping.
        return ch, sp

    def _setup_loss_coefficients(self, num_distillation_points: int):
        """α₀=1 (image), α₁=3, αⱼ=2×αⱼ₋₁, j≥2 — with length = 1 + #taps."""
        if getattr(self.config, "alpha_base", None) is not None:
            coeffs = self.config.alpha_base[:num_distillation_points + 1]
        else:
            coeffs = [1.0]  # α0 for image MSE
            if num_distillation_points > 0:
                coeffs.append(3.0)  # α1
                for j in range(2, num_distillation_points + 1):
                    coeffs.append(2.0 * coeffs[j - 1])
        assert len(coeffs) == num_distillation_points + 1, "α length must be 1 + #taps"
        self.loss_coefficients = coeffs

    # ---------------- Training / Features ----------------
    def train(self, train_loader: DataLoader):
        """Train DINA inverse network."""
        if self.inverse_net is None:
            raise ValueError("Inverse network not built")

        self.inverse_net.train()
        optimizer = torch.optim.SGD(
            self.inverse_net.parameters(),
            lr=self.config.learning_rate,
        )

        for epoch in range(self.config.dina_epochs):
            total_loss = 0.0
            num_batches = 0
            pbar = tqdm(train_loader, desc=f"DINA Epoch {epoch+1}/{self.config.dina_epochs}")

            for data, _ in pbar:
                data = data.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                # Teacher features: split activation + ordered taps (near→far)
                split_act, teacher_taps = self._get_target_features(data)

                # Student: inversion from split_act → block outputs (aligned 1:1 with taps)
                reconstructed, student_block_outputs = self.inverse_net(split_act)

                # Loss
                loss = self._calculate_dina_loss(
                    data, reconstructed,
                    teacher_taps,  # list length = #taps
                    student_block_outputs  # list length = #taps
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if self.config.verbose:
                print(f"Epoch {epoch+1}: Average Loss = {total_loss / max(1, num_batches):.4f}")

    def _get_target_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
          split_activation
          teacher_taps: list of taps ordered near→far relative to split
        """
        # Use the flattened layers (should be already set by _get_subblocks_and_taps)
        if not hasattr(self, '_model_layers'):
            self._model_layers = self._flatten_model_layers(self.target_model)
            
        taps_set = set(self.distillation_points)
        collected: Dict[int, torch.Tensor] = {}
        split_act = None

        for i, layer in enumerate(self._model_layers):
            x = layer(x)
            if i in taps_set:
                collected[i] = x.detach()
            if i == self.split_layer:
                split_act = x.detach()
                # we can early break after split (no need to run further)
                break

        assert split_act is not None, "Failed to capture split activation"
        # Order taps exactly as self.distillation_points (near→far)
        teacher_taps = [collected[i] for i in self.distillation_points if i in collected]
        return split_act, teacher_taps

    def _calculate_dina_loss(self,
                             original_img: torch.Tensor,
                             reconstructed_img: torch.Tensor,
                             teacher_taps: List[torch.Tensor],
                             student_block_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        L_DINA = Σ_j α_j || D_j - S_j ||_2^2 + α_0 || x - x_hat ||_2^2
          where j iterates taps from near→far,
          D_j = teacher tap feature at j,
          S_j = output of inverse block j.
        """
        # Image term
        pixel_loss = F.mse_loss(reconstructed_img, original_img)
        total = self.loss_coefficients[0] * pixel_loss

        # Distillation terms (1-to-1, no skipping)
        assert len(teacher_taps) == len(student_block_outputs), \
            f"#teacher taps ({len(teacher_taps)}) != #student outputs ({len(student_block_outputs)})"
        for j, (stud, teach) in enumerate(zip(student_block_outputs, teacher_taps), start=1):
            if self.config.assert_shapes:
                assert stud.shape[1:] == teach.shape[1:], \
                    f"Shape mismatch at tap {j}: student {stud.shape}, teacher {teach.shape}"
            total = total + self.loss_coefficients[j] * F.mse_loss(stud, teach)

        return total

    # ---------------- Inference & Evaluation ----------------
    def attack(self, activations: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from intermediate activations."""
        self.inverse_net.eval()
        with torch.no_grad():
            reconstructed, _ = self.inverse_net(activations)
        return reconstructed

    def evaluate_ssim(self, val_loader: DataLoader) -> float:
        """Evaluate average SSIM (on de-normalized images)."""
        self.inverse_net.eval()
        total_ssim = 0.0
        num_samples = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device, non_blocking=True)
                activations, _ = self._get_target_features(data)
                reconstructed = self.attack(activations)
                # (4) SSIM on [0,1]
                rec_dn = self.denorm(reconstructed)
                data_dn = self.denorm(data)
                
                # Handle channel mismatch for SSIM calculation
                if rec_dn.shape[1] != data_dn.shape[1]:
                    if data_dn.shape[1] == 1 and rec_dn.shape[1] == 3:
                        # Convert RGB reconstruction to grayscale for MNIST
                        rec_dn = 0.299 * rec_dn[:, 0:1] + 0.587 * rec_dn[:, 1:2] + 0.114 * rec_dn[:, 2:3]
                    elif data_dn.shape[1] == 3 and rec_dn.shape[1] == 1:
                        # Replicate grayscale to RGB
                        rec_dn = rec_dn.repeat(1, 3, 1, 1)
                
                ssim_scores = calculate_ssim_batch(rec_dn, data_dn)
                total_ssim += ssim_scores.sum().item()
                num_samples += data.size(0)
        return total_ssim / max(1, num_samples)

    def evaluate_detailed(self, val_loader: DataLoader) -> AttackResult:
        """Detailed evaluation with SSIM (on [0,1]) and MSE."""
        self.inverse_net.eval()
        ssim_scores = []
        mse_scores = []
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device, non_blocking=True)
                activations, _ = self._get_target_features(data)
                reconstructed = self.attack(activations)
                # (4) SSIM on [0,1]
                rec_dn = self.denorm(reconstructed)
                data_dn = self.denorm(data)
                
                # Handle channel mismatch for SSIM calculation
                if rec_dn.shape[1] != data_dn.shape[1]:
                    if data_dn.shape[1] == 1 and rec_dn.shape[1] == 3:
                        # Convert RGB reconstruction to grayscale for MNIST
                        rec_dn = 0.299 * rec_dn[:, 0:1] + 0.587 * rec_dn[:, 1:2] + 0.114 * rec_dn[:, 2:3]
                    elif data_dn.shape[1] == 3 and rec_dn.shape[1] == 1:
                        # Replicate grayscale to RGB
                        rec_dn = rec_dn.repeat(1, 3, 1, 1)
                
                ssim_batch = calculate_ssim_batch(rec_dn, data_dn)
                mse_batch = F.mse_loss(reconstructed, data, reduction='none').mean(dim=[1, 2, 3])
                ssim_scores.extend(ssim_batch.detach().cpu().numpy())
                mse_scores.extend(mse_batch.detach().cpu().numpy())

        return AttackResult(
            ssim_scores=np.array(ssim_scores),
            mse_scores=np.array(mse_scores),
            avg_ssim=float(np.mean(ssim_scores)) if len(ssim_scores) else 0.0,
            avg_mse=float(np.mean(mse_scores)) if len(mse_scores) else 0.0,
            success_rate=float(np.mean(np.array(ssim_scores) >= self.config.ssim_threshold)) if len(ssim_scores) else 0.0
        )
