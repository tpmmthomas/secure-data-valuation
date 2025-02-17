import torch
from torch import nn
from valuation_template import SingleDataValuation, MultiDataValuation
from MonteCarloShapley import MonteCarloShapley
from sklearn.cluster import KMeans
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics.pairwise import cosine_similarity

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
    
class SingleShapleyValuation(SingleDataValuation):
    def __init__(self, model: nn.Module, data_point, label, trainer_data, testset,datasize,learning_rate,epochs,device,batch_size):
        super().__init__(model, data_point, label)
        self.trainer_data = trainer_data
        self.value = None
        #Insert the data point as the first element of the trainer data
        self.trainer_data = torch.cat([self.data_point.unsqueeze(0), self.trainer_data], dim=0)
        self.MC = MonteCarloShapley(self.trainer_data, testset, L = 1,  beta = 1, c = 1, a = 0.05, b = 0.05, sup = 5, num_classes = 10, datasize = datasize, learning_rate = learning_rate, epochs = epochs, device = device, batch_size = batch_size)

    def data_value(self):
        shapleyValues = self.MC.run([0])
        return shapleyValues[0]


class MultiKMeansValuation(MultiDataValuation):
    def __init__(self, model: nn.Module, data_points, labels, trainer_data, loss_fn, cluster_size, a1,a2,a3):
        super().__init__(model, data_points, labels, trainer_data)
        self.loss_fn = loss_fn
        self.value = None
        self.K = cluster_size
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        assert a1+a2+a3 == 1, "a1 + a2 + a3 must equal 1"
        self.dim_reduction()
        self.summarize_trained_data()

    def dim_reduction(self):
        #Reduce dimension by random projection
        flattened_images = self.data_points.reshape(self.data_points.shape[0], -1)
        flattened_trainerData = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        transformer = SparseRandomProjection(n_components='auto', eps=0.5, random_state=0)
        self.reduced_images = transformer.fit_transform(flattened_images)
        self.reduced_trainerData = transformer.fit_transform(flattened_trainerData)
        return True

    def summarize_trained_data(self):
        #Compute Kmeans of Alice data
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.reduced_trainerData)
        #Compute average of L2 norm of cluster centers
        cluster_centers = kmeans.cluster_centers_
        self.trained_clusters = torch.tensor(cluster_centers)
        self.avg_l2norm = torch.linalg.norm(torch.tensor(cluster_centers), dim=1).mean().item()
        return True

    def select_data(self):
        # K-means clustering
        self.selected_idx = []
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.reduced_images)
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        #Loop through each individual cluster
        for cluster in range(self.K):
            #Retrieve indices and images of the cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_images = self.reduced_images[cluster_indices]

            distances = np.linalg.norm(cluster_images - cluster_centers[cluster], axis=1)
            sorted_indices = np.argsort(distances)
            self.selected_idx.append(cluster_indices[sorted_indices[0]])

        assert len(self.selected_idx) == self.K, "Number of selected indices must equal K"
        return True

    def data_value(self):
        #TODO written by copilot, fix manually later
        total_score = 0
        chosen_points = self.data_points[self.selected_idx]
        chosen_reduced_points = self.reduced_images[self.selected_idx]
        chosen_labels = self.labels[self.selected_idx]
        #Uncertainty score, Diversity score and Loss score
        # Compute loss score
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(chosen_points)
            loss = self.loss_fn(outputs, chosen_labels)
            loss_score = loss.item()

        # Compute uncertainty score
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()

        # Compute diversity score
        diversity_score = 0
        for point in chosen_reduced_points:
            #Compute the min. distance of point to any of self.trained_clusters
            min_distance = torch.min(torch.linalg.norm(point - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
        diversity_score /= len(chosen_reduced_points)
        diversity_score /= self.avg_l2norm #Normalization

        total_score = self.a1 * loss_score + self.a2 * uncertainty_score + self.a3 *  diversity_score
        self.value = total_score
        return self.value
    
    
class MultiUncKMeansValuation(MultiDataValuation):
    def __init__(self, model: nn.Module, data_points, labels, trainer_data, loss_fn, cluster_size, a1,a2,a3):
        super().__init__(model, data_points, labels, trainer_data)
        self.loss_fn = loss_fn
        self.value = None
        self.K = cluster_size
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        assert a1+a2+a3 == 1, "a1 + a2 + a3 must equal 1"
        self.dim_reduction()
        self.summarize_trained_data()

    def dim_reduction(self):
        #Reduce dimension by random projection
        flattened_images = self.data_points.reshape(self.data_points.shape[0], -1)
        flattened_trainerData = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        transformer = SparseRandomProjection(n_components='auto', eps=0.5, random_state=0)
        self.reduced_images = transformer.fit_transform(flattened_images)
        self.reduced_trainerData = transformer.fit_transform(flattened_trainerData)
        return True

    def summarize_trained_data(self):
        #Compute Kmeans of Alice data
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.reduced_trainerData)
        #Compute average of L2 norm of cluster centers
        cluster_centers = kmeans.cluster_centers_
        self.trained_clusters = torch.tensor(cluster_centers)
        self.avg_l2norm = torch.linalg.norm(torch.tensor(cluster_centers), dim=1).mean().item()
        return True

    def select_data(self):
        # K-means clustering
        self.selected_idx = []
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.reduced_images)
        cluster_labels = kmeans.labels_
        #Loop through each individual cluster
        for cluster in range(self.K):
            #Retrieve indices and images of the cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            
            #Compute an uncertainty score
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.data_points[cluster_indices])
                unc_scores = (-torch.sum(outputs * torch.log(outputs + 1e-9), dim=1)).numpy()

            sorted_indices = np.argsort(unc_scores)
            self.selected_idx.append(cluster_indices[sorted_indices[0]])

        assert len(self.selected_idx) == self.K, "Number of selected indices must equal K"
        return True

    def data_value(self):
        #TODO written by copilot, fix manually later
        total_score = 0
        chosen_points = self.data_points[self.selected_idx]
        chosen_reduced_points = self.reduced_images[self.selected_idx]
        chosen_labels = self.labels[self.selected_idx]
        #Uncertainty score, Diversity score and Loss score
        # Compute loss score
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(chosen_points)
            loss = self.loss_fn(outputs, chosen_labels)
            loss_score = loss.item()

        # Compute uncertainty score
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()

        # Compute diversity score
        diversity_score = 0
        for point in chosen_reduced_points:
            #Compute the min. distance of point to any of self.trained_clusters
            min_distance = torch.min(torch.linalg.norm(point - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
        diversity_score /= len(chosen_reduced_points)
        diversity_score /= self.avg_l2norm #Normalization

        total_score = self.a1 * loss_score + self.a2 * uncertainty_score + self.a3 *  diversity_score
        self.value = total_score
        return self.value
    
    
class MultiSubModValuation(MultiDataValuation):
    def __init__(self, model: nn.Module, data_points, labels, trainer_data, loss_fn, cluster_size, a1,a2,a3):
        super().__init__(model, data_points, labels, trainer_data)
        self.loss_fn = loss_fn
        self.value = None
        self.K = cluster_size
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        assert a1+a2+a3 == 1, "a1 + a2 + a3 must equal 1"
        self.dim_reduction()
        self.summarize_trained_data()

    def dim_reduction(self):
        #Reduce dimension by random projection
        flattened_images = self.data_points.reshape(self.data_points.shape[0], -1)
        flattened_trainerData = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        transformer = SparseRandomProjection(n_components='auto', eps=0.5, random_state=0)
        self.reduced_images = transformer.fit_transform(flattened_images)
        self.reduced_trainerData = transformer.fit_transform(flattened_trainerData)
        return True

    def summarize_trained_data(self):
        #Compute Kmeans of Alice data
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.reduced_trainerData)
        #Compute average of L2 norm of cluster centers
        cluster_centers = kmeans.cluster_centers_
        self.trained_clusters = torch.tensor(cluster_centers)
        self.avg_l2norm = torch.linalg.norm(torch.tensor(cluster_centers), dim=1).mean().item()
        return True

    def select_data(self):
        # K-means clustering
        all_features = np.vstack([self.reduced_trainerData, self.reduced_images])
        numA = self.reduced_trainerData.shape[0]
        
        sim_matrix = cosine_similarity(all_features)
        max_similarities = np.max(sim_matrix[:, :numA], axis=1)  # Best similarity to A so far
        self.selected_idx = []
        
        for _ in range(self.K):
            gains = []
            for i in range(numA, numA+len(self.reduced_images)):
                new_max_sim = np.maximum(max_similarities, sim_matrix[:, i])
                marginal_gain = np.sum(new_max_sim) - np.sum(max_similarities)
                gains.append((marginal_gain, i))
                
            best_point = max(gains, key=lambda x: x[0])[1]
            self.selected_idx.append(best_point - numA)
            
            max_similarities = np.maximum(max_similarities, sim_matrix[:, best_point])

        assert len(self.selected_idx) == self.K, "Number of selected indices must equal K"
        return True

    def data_value(self):
        #TODO written by copilot, fix manually later
        total_score = 0
        chosen_points = self.data_points[self.selected_idx]
        chosen_reduced_points = self.reduced_images[self.selected_idx]
        chosen_labels = self.labels[self.selected_idx]
        #Uncertainty score, Diversity score and Loss score
        # Compute loss score
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(chosen_points)
            loss = self.loss_fn(outputs, chosen_labels)
            loss_score = loss.item()

        # Compute uncertainty score
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()

        # Compute diversity score
        diversity_score = 0
        for point in chosen_reduced_points:
            #Compute the min. distance of point to any of self.trained_clusters
            min_distance = torch.min(torch.linalg.norm(point - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
        diversity_score /= len(chosen_reduced_points)
        diversity_score /= self.avg_l2norm #Normalization

        total_score = self.a1 * loss_score + self.a2 * uncertainty_score + self.a3 *  diversity_score
        self.value = total_score
        return self.value