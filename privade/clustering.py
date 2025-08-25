#First, we run the Kmeans clustering algorithm locally on Bob's device
from sklearn.cluster import KMeans
import numpy as np

def kmeans_clustering(reduced_images, K):
    # Reshape the images to be a 2D array (each image is flattened)
    flattened_images = reduced_images.reshape(reduced_images.shape[0], -1)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=K, random_state=0).fit(flattened_images)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    representative_set = []
    for i in range(K):
        # Find indices of points assigned to the i-th cluster
        candidate_indices = np.where(cluster_labels == i)[0]
        # Compute the Euclidean distances of these points to the cluster center
        distances = np.linalg.norm(flattened_images[candidate_indices] - cluster_centers[i], axis=1)
        # Select the point with the smallest distance
        representative_set.append(candidate_indices[np.argmin(distances)])
    representative_set = np.array(representative_set)
    return representative_set