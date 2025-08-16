"""
DINA: Distillation-based Inverse Network Attack (updated)

Changes:
1) Distillation taps at pre-ReLU conv (middle points between sub-blocks).
2A) Distillation compares block *outputs* to teacher taps (1-to-1 alignment).
3) Explicit tap ordering: closest-to-split → earliest; α increases along that order.
4) SSIM is computed on de-normalized images in [0,1] via self.denorm().
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
                 img_size: Tuple[int, int]):
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

        # Final image head maps from earliest-tap channels to RGB
        self.img_head = nn.Sequential(
            nn.Conv2d(channel_path[-1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
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

        # Defaults for de-normalization (ImageNet-style) if not provided
        self.norm_mean = torch.tensor(
            getattr(config, "norm_mean", [0.485, 0.456, 0.406]),
            dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.norm_std = torch.tensor(
            getattr(config, "norm_std", [0.229, 0.224, 0.225]),
            dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

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
            img_size=(self.config.img_size, self.config.img_size)
        ).to(self.device)

        self._setup_loss_coefficients(len(self.distillation_points))  # α schedule (3)

        if self.config.verbose:
            print(f"    Loss coefficients: {self.loss_coefficients}")

    def _get_subblocks_and_taps(self) -> Tuple[List[List[int]], List[int]]:
        """
        Partition layers [0..split_layer] into sub-blocks ending with ReLU.
        Distillation tap for each sub-block is set to the *pre-ReLU conv* (middle point).
        """
        if not hasattr(self.target_model, 'features'):
            raise NotImplementedError("DINA currently supports VGG-style models with .features")

        feats = self.target_model.features
        subblocks: List[List[int]] = []
        current_block: List[int] = []
        taps: List[int] = []

        def find_prev_conv(idx: int) -> Optional[int]:
            j = idx - 1
            while j >= 0:
                if isinstance(feats[j], nn.Conv2d):
                    return j
                j -= 1
            return None

        for i in range(self.split_layer + 1):
            current_block.append(i)
            if isinstance(feats[i], nn.ReLU):
                subblocks.append(current_block)
                # tap = conv immediately before this ReLU (middle point)
                t = find_prev_conv(i)
                if t is not None:
                    taps.append(t)
                current_block = []

        # If split doesn't end on ReLU, close the last fragment (no additional tap)
        if current_block:
            subblocks.append(current_block)

        return subblocks, taps

    def _probe_network_shapes(self) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Build channel/spatial paths so that:
          path = [split] + [tap_near, tap_next, ..., tap_far]
        This yields #blocks == #taps and each block output matches one teacher tap.
        """
        feats = self.target_model.features
        dummy = torch.randn(1, 3, self.config.img_size, self.config.img_size, device=self.device)

        # Collect split activation + taps (already ordered near→far)
        shapes: Dict[str, Tuple[int, Tuple[int,int]]] = {}

        x = dummy
        for i, layer in enumerate(feats):
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
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
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
        feats = self.target_model.features
        taps_set = set(self.distillation_points)
        collected: Dict[int, torch.Tensor] = {}
        split_act = None

        for i, layer in enumerate(feats):
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
