import torch
from torch import nn
from valuation_template import SingleDataValuation, MultiDataValuation
from MonteCarloShapley import MonteCarloShapley
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

##############################################
# SINGLE-DATA VALUATIONS (unchanged)
##############################################

class SingleLossValuation(SingleDataValuation):
    def __init__(self, model, data_point, label, loss_fn):
        """
        model: a scikit-learn SVM model (with probability=True)
        data_point: a torch tensor representing a single data example
        label: a torch tensor representing the label
        loss_fn: a PyTorch loss function
        """
        super().__init__(model, data_point, label)
        self.loss_fn = loss_fn
        self.value = None

    def data_value(self):
        # Convert data_point to NumPy (with batch dimension) and get probabilities.
        data_point_np = self.data_point.unsqueeze(0).cpu().numpy()
        output_np = self.model.predict_proba(data_point_np)
        output = torch.tensor(output_np, dtype=torch.float64).cuda()
        loss = self.loss_fn(output, self.label.unsqueeze(0))
        self.value = loss.item()
        return self.value


class SingleShapleyValuation(SingleDataValuation):
    def __init__(self, model, data_point, label, trainer_data, testset,
                 datasize, learning_rate, epochs, device, batch_size):
        super().__init__(model, data_point, label)
        self.trainer_data = trainer_data
        self.value = None
        # Insert the data point as the first element of the trainer data.
        trainer_data.insert(0, (data_point, label))
        self.trainer_data = trainer_data
        self.MC = MonteCarloShapley(
            model, self.trainer_data, testset,
            L=1, beta=1, c=1, a=0.05, b=0.05, sup=5,
            num_classes=10, datasize=datasize,
            learning_rate=learning_rate, epochs=epochs,
            device=device, batch_size=batch_size
        )

    def data_value(self):
        shapleyValues = self.MC.run([0])
        return shapleyValues[0]


class SingleRandomValuation(SingleDataValuation):
    def __init__(self, model, data_point, label):
        super().__init__(model, data_point, label)
        self.value = None

    def data_value(self):
        self.value = torch.rand(1).item()
        return self.value


class SingleEntropyValuation(SingleDataValuation):
    def __init__(self, model, data_point, label):
        super().__init__(model, data_point, label)
        self.value = None

    def data_value(self):
        # Get predicted probabilities and compute entropy.
        data_point_np = self.data_point.unsqueeze(0).cpu().numpy()
        output_np = self.model.predict_proba(data_point_np)
        output = torch.tensor(output_np, dtype=torch.float64).cuda()
        entropy = -torch.sum(output * torch.log(output + 1e-9), dim=1)
        self.value = entropy.item()
        return self.value


##############################################
# MULTI-DATA VALUATIONS WITHOUT DIMENSION REDUCTION
##############################################

class MultiKMeansValuation(MultiDataValuation):
    def __init__(self, model, data_points, labels, trainer_data, loss_fn,
                 cluster_size, a1, a2, a3):
        """
        Here we assume that both data_points and trainer_data are already in an
        appropriate (e.g., flattened) format.
        """
        super().__init__(model, data_points, labels, trainer_data)
        self.loss_fn = loss_fn
        self.value = None
        self.K = cluster_size
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        assert a1 + a2 + a3 == 1, "a1 + a2 + a3 must equal 1"
        self.summarize_trained_data()

    def summarize_trained_data(self):
        # Use the trainer data directly.
        if torch.is_tensor(self.trainer_data):
            trainer_np = self.trainer_data.cpu().numpy().reshape(self.trainer_data.shape[0], -1)
        else:
            trainer_np = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(trainer_np)
        cluster_centers = kmeans.cluster_centers_
        self.trained_clusters = torch.tensor(cluster_centers, dtype=torch.float64).cuda()
        self.avg_l2norm = torch.linalg.norm(self.trained_clusters, dim=1).mean().item()

    def select_data(self):
        self.selected_idx = []
        if torch.is_tensor(self.data_points):
            data_np = self.data_points.cpu().numpy().reshape(self.data_points.shape[0], -1)
        else:
            data_np = self.data_points.reshape(self.data_points.shape[0], -1)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(data_np)
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        for cluster in range(self.K):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_data = data_np[cluster_indices]
            distances = np.linalg.norm(cluster_data - cluster_centers[cluster], axis=1)
            sorted_indices = np.argsort(distances)
            self.selected_idx.append(cluster_indices[sorted_indices[0]])
        assert len(self.selected_idx) == self.K, "Number of selected indices must equal K"
        return True

    def data_value(self):
        self.select_data()
        # Get the chosen points.
        if torch.is_tensor(self.data_points):
            chosen_points = self.data_points[self.selected_idx]
        else:
            chosen_points = self.data_points[self.selected_idx]
        if torch.is_tensor(chosen_points):
            chosen_points_np = chosen_points.cpu().numpy()
        else:
            chosen_points_np = chosen_points
        output_np = self.model.predict_proba(chosen_points_np)
        outputs = torch.tensor(output_np, dtype=torch.float64).cuda()
        # print(outputs, self.labels[self.selected_idx])
        loss = self.loss_fn(outputs, self.labels[self.selected_idx])
        loss_score = loss.item()
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()
        # Compute diversity score relative to the trainer clusters.
        if torch.is_tensor(self.data_points):
            data_np = self.data_points.cpu().numpy().reshape(self.data_points.shape[0], -1)
        else:
            data_np = self.data_points.reshape(self.data_points.shape[0], -1)
        chosen_data = data_np[self.selected_idx]
        diversity_score = 0
        for point in chosen_data:
            point_tensor = torch.tensor(point, dtype=torch.float64).cuda()
            min_distance = torch.min(torch.linalg.norm(point_tensor - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
        diversity_score /= len(self.selected_idx)
        diversity_score /= self.avg_l2norm
        total_score = self.a1 * loss_score + self.a2 * uncertainty_score + self.a3 * diversity_score
        self.value = total_score
        return self.value


class MultiUncKMeansValuation(MultiDataValuation):
    def __init__(self, model, data_points, labels, trainer_data, loss_fn,
                 cluster_size, a1, a2, a3):
        super().__init__(model, data_points, labels, trainer_data)
        self.loss_fn = loss_fn
        self.value = None
        self.K = cluster_size
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        assert a1 + a2 + a3 == 1, "a1 + a2 + a3 must equal 1"
        self.summarize_trained_data()

    def summarize_trained_data(self):
        if torch.is_tensor(self.trainer_data):
            trainer_np = self.trainer_data.cpu().numpy().reshape(self.trainer_data.shape[0], -1)
        else:
            trainer_np = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(trainer_np)
        cluster_centers = kmeans.cluster_centers_
        self.trained_clusters = torch.tensor(cluster_centers, dtype=torch.float64).cuda()
        self.avg_l2norm = torch.linalg.norm(self.trained_clusters, dim=1).mean().item()

    def select_data(self):
        self.selected_idx = []
        if torch.is_tensor(self.data_points):
            data_np = self.data_points.cpu().numpy().reshape(self.data_points.shape[0], -1)
        else:
            data_np = self.data_points.reshape(self.data_points.shape[0], -1)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(data_np)
        cluster_labels = kmeans.labels_
        for cluster in range(self.K):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_data = data_np[cluster_indices]
            output_np = self.model.predict_proba(cluster_data)
            outputs = torch.tensor(output_np, dtype=torch.float64).cuda()
            unc_scores = (-torch.sum(outputs * torch.log(outputs + 1e-9), dim=1)).cpu().numpy()
            sorted_indices = np.argsort(unc_scores)
            self.selected_idx.append(cluster_indices[sorted_indices[0]])
        assert len(self.selected_idx) == self.K, "Number of selected indices must equal K"
        return True

    def data_value(self):
        self.select_data()
        if torch.is_tensor(self.data_points):
            chosen_points = self.data_points[self.selected_idx]
        else:
            chosen_points = self.data_points[self.selected_idx]
        if torch.is_tensor(chosen_points):
            chosen_points_np = chosen_points.cpu().numpy()
        else:
            chosen_points_np = chosen_points
        output_np = self.model.predict_proba(chosen_points_np)
        outputs = torch.tensor(output_np, dtype=torch.float64).cuda()
        loss = self.loss_fn(outputs, self.labels[self.selected_idx])
        loss_score = loss.item()
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()
        if torch.is_tensor(self.data_points):
            data_np = self.data_points.cpu().numpy().reshape(self.data_points.shape[0], -1)
        else:
            data_np = self.data_points.reshape(self.data_points.shape[0], -1)
        chosen_data = data_np[self.selected_idx]
        diversity_score = 0
        for point in chosen_data:
            point_tensor = torch.tensor(point, dtype=torch.float64).cuda()
            min_distance = torch.min(torch.linalg.norm(point_tensor - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
        diversity_score /= len(self.selected_idx)
        diversity_score /= self.avg_l2norm
        total_score = self.a1 * loss_score + self.a2 * uncertainty_score + self.a3 * diversity_score
        self.value = total_score
        return self.value


class MultiSubModValuation(MultiDataValuation):
    def __init__(self, model, data_points, labels, trainer_data, loss_fn,
                 cluster_size, a1, a2, a3):
        super().__init__(model, data_points, labels, trainer_data)
        self.loss_fn = loss_fn
        self.value = None
        self.K = cluster_size
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        assert a1 + a2 + a3 == 1, "a1 + a2 + a3 must equal 1"
        self.summarize_trained_data()

    def summarize_trained_data(self):
        if torch.is_tensor(self.trainer_data):
            trainer_np = self.trainer_data.cpu().numpy().reshape(self.trainer_data.shape[0], -1)
        else:
            trainer_np = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(trainer_np)
        cluster_centers = kmeans.cluster_centers_
        self.trained_clusters = torch.tensor(cluster_centers, dtype=torch.float64).cuda()
        self.avg_l2norm = torch.linalg.norm(self.trained_clusters, dim=1).mean().item()

    def select_data(self):
        if torch.is_tensor(self.data_points):
            data_np = self.data_points.cpu().numpy().reshape(self.data_points.shape[0], -1)
        else:
            data_np = self.data_points.reshape(self.data_points.shape[0], -1)
        # Combine trainer and candidate data features.
        trainer_np = self.trainer_data
        if torch.is_tensor(trainer_np):
            trainer_np = trainer_np.cpu().numpy().reshape(trainer_np.shape[0], -1)
        else:
            trainer_np = trainer_np.reshape(trainer_np.shape[0], -1)
        all_features = np.vstack([trainer_np, data_np])
        numA = trainer_np.shape[0]
        sim_matrix = cosine_similarity(all_features)
        max_similarities = np.max(sim_matrix[:, :numA], axis=1)
        self.selected_idx = []
        for _ in range(self.K):
            gains = []
            for i in range(numA, numA + data_np.shape[0]):
                new_max_sim = np.maximum(max_similarities, sim_matrix[:, i])
                marginal_gain = np.sum(new_max_sim) - np.sum(max_similarities)
                gains.append((marginal_gain, i))
            best_point = max(gains, key=lambda x: x[0])[1]
            self.selected_idx.append(best_point - numA)
            max_similarities = np.maximum(max_similarities, sim_matrix[:, best_point])
        assert len(self.selected_idx) == self.K, "Number of selected indices must equal K"
        return True

    def data_value(self):
        self.select_data()
        if torch.is_tensor(self.data_points):
            chosen_points = self.data_points[self.selected_idx]
        else:
            chosen_points = self.data_points[self.selected_idx]
        if torch.is_tensor(chosen_points):
            chosen_points_np = chosen_points.cpu().numpy()
        else:
            chosen_points_np = chosen_points
        output_np = self.model.predict_proba(chosen_points_np)
        outputs = torch.tensor(output_np, dtype=torch.float64).cuda()
        loss = self.loss_fn(outputs, self.labels[self.selected_idx])
        loss_score = loss.item()
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()
        if torch.is_tensor(self.data_points):
            data_np = self.data_points.cpu().numpy().reshape(self.data_points.shape[0], -1)
        else:
            data_np = self.data_points.reshape(self.data_points.shape[0], -1)
        chosen_data = data_np[self.selected_idx]
        diversity_score = 0
        for point in chosen_data:
            point_tensor = torch.tensor(point, dtype=torch.float64).cuda()
            min_distance = torch.min(torch.linalg.norm(point_tensor - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
        diversity_score /= len(self.selected_idx)
        diversity_score /= self.avg_l2norm
        total_score = self.a1 * loss_score + self.a2 * uncertainty_score + self.a3 * diversity_score
        self.value = total_score
        return self.value


class MultiRandomValuation(MultiDataValuation):
    def __init__(self, model, data_points, labels, trainer_data):
        super().__init__(model, data_points, labels, trainer_data)
        self.value = None

    def select_data(self):
        return True

    def data_value(self):
        self.value = torch.rand(1).item()
        return self.value


class MultiEntropyValuation(MultiDataValuation):
    def __init__(self, model, data_points, labels, trainer_data, num_clusters):
        super().__init__(model, data_points, labels, trainer_data)
        self.K = num_clusters
        self.value = None

    def select_data(self):
        self.selected_idx = []
        if torch.is_tensor(self.data_points):
            data_np = self.data_points.cpu().numpy().reshape(self.data_points.shape[0], -1)
        else:
            data_np = self.data_points.reshape(self.data_points.shape[0], -1)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(data_np)
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        for cluster in range(self.K):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_data = data_np[cluster_indices]
            distances = np.linalg.norm(cluster_data - cluster_centers[cluster], axis=1)
            sorted_indices = np.argsort(distances)
            self.selected_idx.append(cluster_indices[sorted_indices[0]])
        assert len(self.selected_idx) == self.K, "Number of selected indices must equal K"
        return True  

    def data_value(self):
        self.select_data()
        if torch.is_tensor(self.data_points):
            chosen_points = self.data_points[self.selected_idx]
        else:
            chosen_points = self.data_points[self.selected_idx]
        if torch.is_tensor(chosen_points):
            chosen_points_np = chosen_points.cpu().numpy()
        else:
            chosen_points_np = chosen_points
        output_np = self.model.predict_proba(chosen_points_np)
        output = torch.tensor(output_np, dtype=torch.float64).cuda()
        entropy = (-torch.sum(output * torch.log(output + 1e-9), dim=1)).mean()
        self.value = entropy.item()
        return self.value


class MultiCoreSetValuation(MultiDataValuation):
    def __init__(self, model, data_points, labels, trainer_data, num_clusters):
        super().__init__(model, data_points, labels, trainer_data)
        self.value = None
        self.min_distances = None
        self.already_selected = []
        self.n_obs = self.data_points.shape[0]
        self.N = num_clusters
        # Flatten data and combine trainer and candidate data.
        if torch.is_tensor(self.data_points):
            data_np = self.data_points.cpu().numpy().reshape(self.data_points.shape[0], -1)
        else:
            data_np = self.data_points.reshape(self.data_points.shape[0], -1)
        if torch.is_tensor(self.trainer_data):
            trainer_np = self.trainer_data.cpu().numpy().reshape(self.trainer_data.shape[0], -1)
        else:
            trainer_np = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        self.all_features = np.vstack([trainer_np, data_np])
        self.already_selected = [i for i in range(trainer_np.shape[0])]

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if cluster_centers:
            x = self.all_features[cluster_centers]
            dist = pairwise_distances(self.all_features, x, metric='euclidean')
            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def data_value(self):
        self.update_distances(self.already_selected, only_new=False, reset_dist=False)
        new_batch = []
        for _ in range(self.N):
            if self.already_selected is None:
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            self.update_distances([ind], only_new=True, reset_dist=False)
            self.already_selected.append(ind)
            new_batch.append(ind)
        return max(self.min_distances)[0]
