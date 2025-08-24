# ====== put your LogME implementation above this line ======
# (the LogME class you pasted in your message)
from logme import LogME
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# ------------------ config ------------------
SEED = 42
BATCH_SIZE = 256
NUM_WORKERS = 4
RESIZE = 224                    # match ImageNet pretraining
N_SUPPORT_PER_CLASS = 100       # 100 x 10 = 1000 labeled train samples
K_KNN = 5                       # k in kNN
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# ------------------ reproducibility ------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed()

# ------------------ data ------------------
# ImageNet normalization for pretrained ResNet
imnet_mean = [0.485, 0.456, 0.406]
imnet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(RESIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imnet_mean, std=imnet_std),
])

train_set = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
test_set  = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

# Stratified pick of N_SUPPORT_PER_CLASS from training set
def stratified_indices(dataset, per_class=100):
    cls_to_idxs = {c: [] for c in range(10)}
    for idx, (_, y) in enumerate(dataset):
        cls_to_idxs[y].append(idx)
    for c in cls_to_idxs:
        rng = np.random.default_rng(SEED + c)
        cls_to_idxs[c] = rng.choice(cls_to_idxs[c], size=per_class, replace=False).tolist()
    support_idxs = sum((cls_to_idxs[c] for c in range(10)), [])
    return support_idxs

support_indices = stratified_indices(train_set, N_SUPPORT_PER_CLASS)
support_sampler = torch.utils.data.SubsetRandomSampler(support_indices)
support_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, sampler=support_sampler,
                                             num_workers=NUM_WORKERS, pin_memory=USE_GPU)
test_loader    = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                             num_workers=NUM_WORKERS, pin_memory=USE_GPU)

# ------------------ model & block splitter ------------------
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()

# We’ll grab activations after:
#  - stem relu (conv1 -> bn1 -> relu)
#  - layer1 (end of stage)
#  - layer2
#  - layer3
#  - layer4
BLOCK_NAMES = ["stem_relu", "layer1", "layer2", "layer3", "layer4"]

@torch.no_grad()
def forward_blocks_and_collect(x, model):
    feats = {}
    o = model.conv1(x)
    o = model.bn1(o)
    o = model.relu(o)
    feats["stem_relu"] = o.clone()

    o = model.maxpool(o)
    o = model.layer1(o)    # BasicBlocks end with ReLU
    feats["layer1"] = o.clone()

    o = model.layer2(o)
    feats["layer2"] = o.clone()

    o = model.layer3(o)
    feats["layer3"] = o.clone()

    o = model.layer4(o)
    feats["layer4"] = o.clone()
    return feats

@torch.no_grad()
def extract_features(loader, model):
    """
    Returns:
      feats_dict: {block_name: np.array [N, D]}
      labels:     np.array [N]
    Features are globally averaged to vectors and L2-normalized (good for kNN).
    """
    all_feats = {bn: [] for bn in BLOCK_NAMES}
    all_labels = []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        feats = forward_blocks_and_collect(x, model)
        # global average pool -> [N, C]
        for bn in BLOCK_NAMES:
            f = feats[bn]
            f = F.adaptive_avg_pool2d(f, 1).squeeze(-1).squeeze(-1)  # [N, C]
            # L2 normalize for cosine distance
            f = F.normalize(f, p=2, dim=1)
            all_feats[bn].append(f.cpu().numpy())
        all_labels.append(y.numpy())
    feats_np = {bn: np.concatenate(all_feats[bn], axis=0) for bn in BLOCK_NAMES}
    labels_np = np.concatenate(all_labels, axis=0)
    return feats_np, labels_np

print("Extracting features (support set)...")
support_feats, support_labels = extract_features(support_loader, resnet)
print("Extracting features (test set)...")
test_feats, test_labels = extract_features(test_loader, resnet)

# ------------------ kNN and LogME per block ------------------
def knn_accuracy(block, Z_support, y_support, Z_query, y_query, k=5, metric='cosine'):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
    knn.fit(Z_support, y_support)
    return float(knn.score(Z_query, y_query))  # fraction correct

def logme_score(block, Z, y):
    # LogME expects numpy arrays; returns average per-sample evidence across classes
    logme = LogME(regression=False)
    return float(logme.fit(Z.astype(np.float64), y.astype(np.int64)))

results = []
for bn in BLOCK_NAMES:
    print(f"Scoring block: {bn}")
    Zs = support_feats[bn]
    ys = support_labels
    Zq = test_feats[bn]
    yq = test_labels

    beta_knn = knn_accuracy(bn, Zs, ys, Zq, yq, k=K_KNN, metric='cosine')
    beta_logme = logme_score(bn, Zs, ys)  # per-sample evidence (already /N in your impl)

    results.append((bn, beta_knn, beta_logme))
    print(f"  kNN@{K_KNN}: {beta_knn*100:.2f}% | LogME/N: {beta_logme:.4f}")

# ------------------ plot ------------------
blocks = [r[0] for r in results]
knn_vals = [r[1]*100.0 for r in results]      # %
logme_vals = [r[2] for r in results]          # per-sample evidence (can be negative)

fig, ax1 = plt.subplots(figsize=(8,4.8))
ax1.plot(blocks, knn_vals, marker='o', linewidth=2, label=f'kNN@{K_KNN} Accuracy (%)')
ax1.set_ylabel('kNN Accuracy (%)')
ax1.set_ylim(0, 100)
ax1.grid(True, axis='y', alpha=0.3)

# Secondary y-axis for LogME
ax2 = ax1.twinx()
ax2.plot(blocks, logme_vals, marker='s', linestyle='--', linewidth=2, color='tab:orange', label='LogME / sample')
ax2.set_ylabel('LogME (per sample)')
# optional: auto-scale around observed values
ymin = min(logme_vals) - 0.05*max(1e-6, abs(min(logme_vals)))
ymax = max(logme_vals) + 0.05*max(1e-6, abs(max(logme_vals)))
if ymin == ymax:
    ymin, ymax = ymin - 1.0, ymax + 1.0
ax2.set_ylim(ymin, ymax)

# Build a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

ax1.set_title('Transferability across ResNet-18 blocks on CIFAR-10')
plt.tight_layout()
plt.savefig("transferability_plot.png")
