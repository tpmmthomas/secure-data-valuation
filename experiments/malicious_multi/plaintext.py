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

def pipeline_total_score(
    model_name: str
):
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
        

    # 6a) Diversity score on reduced features of reps
    red_reps = red[reps]                       # (K, DIM)
    stds     = red_reps.std(axis=0, ddof=0)    # (DIM,)
    diversity_score = stds.mean()
    

    # 6b) Uncertainty = mean entropy (base-2)
    #     H(p) = – ∑ p * log2(p)  (treat 0·log2(0)=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(probs > 0,
                        probs * np.log(probs),
                        0.0)
    entropies = -term.sum(axis=1)             # (K,)
    uncertainty_score = entropies.mean()

    # 6c) Loss = mean cross‐entropy loss in bits: –log2(p_true)
    true_probs = probs[np.arange(K), selected_labels[reps]]
    with np.errstate(divide='ignore'):
        losses = -np.log(true_probs)
    loss_score = losses.mean()

    # Total
    total_score = alpha1*diversity_score \
                + alpha2*uncertainty_score \
                + alpha3*loss_score
                

    return total_score


if __name__ == "__main__":
    x = pipeline_total_score("SVM")
    print(x)