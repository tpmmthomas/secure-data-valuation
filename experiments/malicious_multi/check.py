import os
import importlib
import torch
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import torch.nn.functional as F

image_path = "data/selected_images.pth"
labels_path = "data/selected_labels.pth"
DIM: int = 50
K: int = 20
alpha1: float = 0.3
alpha2: float = 0.3
alpha3: float = 0.4
random_state: int = 42
model_name = "SVM"
# 1) Load data
selected_images = torch.load(image_path).numpy()     # shape (N, …)
selected_labels = torch.load(labels_path).numpy()   # shape (N,)

# 2) Load model
model_module = importlib.import_module("model")
ModelClass   = getattr(model_module, model_name)
model        = ModelClass()
model.load_state_dict(torch.load("data/model.pth"))
model.eval()


N = selected_images.shape[0]

# 3) Dimension reduction
flat = selected_images.reshape(N, -1)
proj = SparseRandomProjection(n_components=DIM, random_state=random_state)
red  = proj.fit_transform(flat)        # (N, DIM)
red  = MinMaxScaler((0,1)).fit_transform(red)

# 4) Clustering + representative‐set
kmeans = KMeans(n_clusters=K, random_state=random_state).fit(red)
labels_k = kmeans.labels_             # (N,)
centers  = kmeans.cluster_centers_    # (K, DIM)
reps     = []
for c in range(K):
    idxs = np.where(labels_k == c)[0]
    dists = np.linalg.norm(red[idxs, :] - centers[c], axis=1)
    reps.append(idxs[np.argmin(dists)])
reps = np.array(reps, dtype=int)      # shape (K,)

# 5) Inference on raw images
#    assume model accepts input shape == selected_images[reps].shape
with torch.no_grad():
    inp = torch.tensor(selected_images[reps])
    logits = model(inp)
    probs  = F.softmax(logits, dim=1).numpy()   # (K, cls)
    
#Flatten probs to one dimensional array
probs_flat = probs.flatten()  # (K * cls,)
# Read the file and parse the space-separated values
file_path = os.path.join('../../MP-SPDZ/Player-Data', 'Input-P1-0')
with open(file_path, 'r') as f:
    file_contents = f.read().strip()
file_values = np.array([float(val) for val in file_contents.split()])

# Verify lengths are the same
if file_values.shape[0] != probs_flat.shape[0]:
    raise ValueError(f"Length mismatch: file has {file_values.shape[0]} values, but probs_flat has {probs_flat.shape[0]}.")
else:
    print(f"Length match: file has {file_values.shape[0]} values, probs_flat has {probs_flat.shape[0]} values.")
# Check if absolute difference at any index exceeds 0.1% (i.e., 0.001)
tolerance = 0.005
differences = np.abs(file_values - probs_flat)
if np.any(differences > tolerance):
    raise ValueError("Absolute difference between file and probs_flat value exceeds 0.1% for at least one index.")

print("OK!")
