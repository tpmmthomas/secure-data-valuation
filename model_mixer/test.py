import copy
import torch, torch.nn as nn
from torchvision import models

# torch.manual_seed(0)

# Build a standard ResNet18
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.eval()

# A = stem
stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

# Keep ORIGINAL B for ground truth, and a MODIFIED copy for surgery
B_orig = copy.deepcopy(resnet.layer1)
B_mod  = copy.deepcopy(resnet.layer1)

# Secret orthogonal mixer M (C x C), inverse is M.T
C = resnet.conv1.out_channels  # 64
with torch.no_grad():
    Q, _ = torch.linalg.qr(torch.randn(C, C))
M = Q
Minv = M.T

# ---- Fold M^{-1} into the FIRST block of B_mod ----
blk = B_mod[0]
assert isinstance(blk.conv1, nn.Conv2d) and blk.conv1.in_channels == C

with torch.no_grad():
    # Main-branch first conv: W' = W @ M^{-1} (right-multiply on in-channels)
    W = blk.conv1.weight.data            # [out, in=C, k, k]
    out_c, in_c, kh, kw = W.shape
    W_flat = W.permute(0,2,3,1).reshape(-1, in_c)   # [(out*kh*kw), in]
    W_flat = W_flat @ Minv                           # <-- key: @ Minv
    W_new  = W_flat.reshape(out_c, kh, kw, in_c).permute(0,3,1,2).contiguous()
    blk.conv1.weight.copy_(W_new)

    # Skip path: insert a 1x1 conv with weight = M^{-1}
    skip = nn.Conv2d(C, C, kernel_size=1, bias=False)
    skip.weight.copy_(Minv.view(C, C, 1, 1))
    blk.downsample = skip  # identity becomes M^{-1}

B_mod.eval()

# Define A = stem + mixer(M)
mixer = nn.Conv2d(C, C, kernel_size=1, bias=False)
with torch.no_grad():
    mixer.weight.copy_(M.view(C, C, 1, 1))
A = nn.Sequential(stem, mixer)
A.eval()

# ----- Sanity check -----
x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    z = stem(x)                 # original basis
    y_true = B_orig(z.clone())  # ORIGINAL B on z
    u = A(x)                    # mixed basis u = M z
    y_fold = B_mod(u)           # MODIFIED B on u

print("max abs diff:", (y_true - y_fold).abs().max().item())  # ~1e-6 to 1e-5
