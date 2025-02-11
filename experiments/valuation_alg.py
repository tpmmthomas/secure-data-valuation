import torch
from torch import nn
from valuation_template import SingleDataValuation, MultiDataValuation
from sklearn.cluster import KMeans
import numpy as np

class SingleLossValuation(SingleDataValuation):
    def __init__(self, model: nn.Module, data_point, label, loss_fn):
        super().__init__(model, data_point, label)
        self.loss_fn = loss_fn
        self.value = None

    def data_value(self):
        self.model.eval()
        with torch.no_grad():
            data_point_batch = self.data_point.unsqueeze(0)  # Add batch dimension
            output = self.model(data_point_batch)
            loss = self.loss_fn(output, self.label.unsqueeze(0))
        self.value = loss.item()
        return loss.item()


class MultiSubModValuation(MultiDataValuation):
    def __init__(self, model: nn.Module, data_points, labels,trainer_data, loss_fn):
        super().__init__(model, data_points, labels, trainer_data)
        self.loss_fn = loss_fn
        self.value = None

    def select_data(self):
        #Subnodular Optimization
        #TODO generated by copilot, may need fixing
        selected_indices = []
        covered_points = set(self.trainer_data.tolist())
        alpha = 1.0  # Define the radius of the ball
        K = 10  # Number of points to select

        for _ in range(K):
            best_point = None
            best_coverage = 0

            for i, point in enumerate(self.data_points):
                if i in selected_indices:
                    continue

            new_coverage = 0
            for trainer_point in self.trainer_data:
                if torch.dist(point, trainer_point) <= alpha:
                    new_coverage += 1

            if new_coverage > best_coverage:
                best_coverage = new_coverage
                best_point = i

            if best_point is not None:
                selected_indices.append(best_point)
            covered_points.update(self.data_points[best_point].tolist())

        self.value = selected_indices

    def data_value(self):
        #TODO written by copilot, fix manually later
        total_score = 0
        #Uncertainty score, Diversity score and Loss score
        # Compute loss score
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data_points)
            loss = self.loss_fn(outputs, self.labels)
            loss_score = loss.item()

        # Compute uncertainty score
        probabilities = torch.softmax(outputs, dim=1)
        uncertainty_score = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1).mean().item()

        # Compute diversity score
        diversity_score = 0
        alpha = 1.0  # Define the radius of the ball
        for point in self.data_points:
            for trainer_point in self.trainer_data:
                if torch.dist(point, trainer_point) <= alpha:
                    diversity_score += 1
        diversity_score /= len(self.data_points)

        total_score = loss_score + uncertainty_score + diversity_score
        self.value = total_score
        return self.value

class MultiKMeansValuation(MultiDataValuation):
    def __init__(self, model: nn.Module, data_points, labels, trainer_data, loss_fn, cluster_size):
        super().__init__(model, data_points, labels, trainer_data)
        self.loss_fn = loss_fn
        self.value = None
        self.K = cluster_size

    def _uncertainty(self, input):
        pass

    def select_data(self):
        # K-means clustering
        self.selected_idx = []
        flattened_images = self.data_points.reshape(self.data_points.shape[0], -1)
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(flattened_images)
        cluster_labels = kmeans.labels_
        # cluster_centers = kmeans.cluster_centers_
        #Loop through each individual cluster
        for cluster in range(self.K):
            #Retrieve indices and images of the cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_images = self.data_points[cluster_indices]
            #Version without constraints -- check uncertinaty of every point
            uncertianty_scores = []
            for image in cluster_images:
                uncertianty_scores.append(self._uncertainty(image))
            most_uncertain_idx = cluster_indices[np.argmax(uncertianty_scores)]
            self.selected_idx.append(most_uncertain_idx)
            

    def data_value(self):
        #TODO written by copilot, fix manually later
        total_score = 0
        #Uncertainty score, Diversity score and Loss score
        # Compute loss score
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.data_points)
            loss = self.loss_fn(outputs, self.labels)
            loss_score = loss.item()

        # Compute uncertainty score
        probabilities = torch.softmax(outputs, dim=1)
        uncertainty_score = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1).mean().item()

        # Compute diversity score
        diversity_score = 0
        alpha = 1.0  # Define the radius of the ball
        for point in self.data_points:
            for trainer_point in self.trainer_data:
                if torch.dist(point, trainer_point) <= alpha:
                    diversity_score += 1
        diversity_score /= len(self.data_points)

        total_score = loss_score + uncertainty_score + diversity_score
        self.value = total_score
        return self.value