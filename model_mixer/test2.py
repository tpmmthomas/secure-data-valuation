import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.manual_seed(0)

# -------- A simple 5-layer CNN (no skips, no BN) --------
class CNN5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        # Layer 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # Layer 5
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Head
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # --- prefix up to conv2 (our cut will be here) ---
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)            # /2
        x = F.relu(self.conv2(x))         # SHAPE: [N, 64, H/2, W/2]

        # --- suffix conv3..conv5 + head ---
        x = F.relu(self.conv3(x))         # /2
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x, 1)   # [N, 256, 1, 1]
        x = x.view(x.size(0), -1)         # [N, 256]
        return self.fc(x)

    # convenience: output after conv2 (the cut)
    def forward_to_conv2(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        return x

    # convenience: run from conv3 onward
    def forward_from_conv3(self, x_after_conv2):
        x = F.relu(self.conv3(x_after_conv2))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------- helper: fold a right-side matrix into conv's in-channel dim --------
def fold_right_into_in_channels(conv: nn.Conv2d, R: torch.Tensor):
    """
    conv.weight: [out_c, in_c, kh, kw]
    We want W' @ (R @ x) == W @ x  ->  W' = W @ R^{-1}
    Here we pass Rinv directly, so do: W' = W @ Rinv
    This function multiplies on the 'in' axis (right-multiply).
    """
    with torch.no_grad():
        W = conv.weight.data
        out_c, in_c, kh, kw = W.shape
        assert R.shape == (in_c, in_c), "R must match in_channels"
        W_flat = W.permute(0, 2, 3, 1).reshape(-1, in_c)  # [(out*kh*kw), in]
        W_new_flat = W_flat @ R                           # right-multiply
        W_new = W_new_flat.reshape(out_c, kh, kw, in_c).permute(0, 3, 1, 2).contiguous()
        conv.weight.copy_(W_new)
        # bias unchanged


# ======================= Demo =======================
device = "cpu"
model = CNN5().to(device).eval()
x = torch.randn(2, 3, 224, 224, device=device)

# Ground-truth output from the original model
with torch.no_grad():
    y_ref = model(x)

# We'll cut after conv2. Let z be features after conv2 (original basis).
with torch.no_grad():
    z = model.forward_to_conv2(x)   # [N, 64, H/2, W/2]
C = z.shape[1]                      # 64 channels

# ----- Secret mixer M (orthogonal), and its inverse -----
with torch.no_grad():
    Q, _ = torch.linalg.qr(torch.randn(C, C, device=device))
M = Q                  # orthogonal mixer (secret)
Minv = M.T             # inverse

# ----- Build "A": mixer only (1x1 conv with weight = M) -----
mixer = nn.Conv2d(C, C, kernel_size=1, bias=False).to(device).eval()
with torch.no_grad():
    mixer.weight.copy_(M.view(C, C, 1, 1))

# Compute mixed features u = M z
with torch.no_grad():
    u = mixer(z)

# ----- Build "B_mod": copy of the suffix with weight surgery on conv3 -----
B_mod = copy.deepcopy(model).eval()

# Fold Minv into conv3's in-channels (the first conv that consumes features after conv2)
fold_right_into_in_channels(B_mod.conv3, Minv)

# Now, running B_mod.from_conv3(u) should match model.from_conv3(z)
with torch.no_grad():
    y_fold = B_mod.forward_from_conv3(u)
    y_true_tail = model.forward_from_conv3(z)

# Compare just the tail outputs (should match closely)
print("Tail max abs diff:", (y_true_tail - y_fold).abs().max().item())

# For completeness, run full end-to-end two ways:
# 1) Original model
# 2) "Split": run prefix to conv2, apply mixer, then run B_mod tail
with torch.no_grad():
    y_split = B_mod.forward_from_conv3(mixer(model.forward_to_conv2(x)))

print("End-to-end max abs diff:", (y_ref - y_split).abs().max().item())
