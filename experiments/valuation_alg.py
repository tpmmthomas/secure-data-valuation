import torch
from torch import nn
from valuation_template import SingleDataValuation, MultiDataValuation
from MonteCarloShapley import MonteCarloShapley
from sklearn.cluster import KMeans
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

class SingleLossValuation(SingleDataValuation):
    def __init__(self, model: nn.Module, data_point, label, loss_fn):
        super().__init__(model, data_point, label)
        self.loss_fn = loss_fn
        self.value = None

    def data_value(self):
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                data_point_batch = self.data_point.unsqueeze(0)  # Add batch dimension
                output = self.model(data_point_batch)
                loss = self.loss_fn(output, self.label.unsqueeze(0))
            self.value = loss.item()
            return loss.item()
        # else:
        #     output = torch.tensor(self.model.predict([self.data_point]))
        #     loss = self.loss_fn(output, self.label.unsqueeze(0))
        #     self.value = loss.item()
        #     return loss.item()
    
class SingleShapleyValuation(SingleDataValuation):
    def __init__(self, model: nn.Module, data_point, label, trainer_data, testset,datasize,learning_rate,epochs,device,batch_size):
        super().__init__(model, data_point, label)
        self.trainer_data = trainer_data
        self.value = None
        #Insert the data point as the first element of the trainer data
        trainer_data.insert(0, (data_point, label))
        self.trainer_data = trainer_data
        self.MC = MonteCarloShapley(model,self.trainer_data, testset, L = 1,  beta = 1, c = 1, a = 0.05, b = 0.05, sup = 5, num_classes = 10, datasize = datasize, learning_rate = learning_rate, epochs = epochs, device = device, batch_size = batch_size)

    def data_value(self):
        shapleyValues = self.MC.run([0])
        return shapleyValues[0]
    
class SingleRandomValuation(SingleDataValuation):
    def __init__(self, model: nn.Module, data_point, label):
        super().__init__(model, data_point, label)
        self.value = None

    def data_value(self):
        self.value = torch.rand(1).item()
        return self.value
    
class SingleEntropyValuation(SingleDataValuation):
    def __init__(self, model: nn.Module, data_point, label):
        super().__init__(model, data_point, label)
        self.value = None

    def data_value(self):
        self.model.eval()
        with torch.no_grad():
            data_point_batch = self.data_point.unsqueeze(0)  # Add batch dimension
            output = self.model(data_point_batch)
            #Softmax
            output = nn.Softmax(dim=1)(output)
            entropy = -torch.sum(output * torch.log(output + 1e-9), dim=1)
        self.value = entropy.item()
        return self.value
    

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
        self.reduced_images = transformer.fit_transform(flattened_images.cpu())
        self.reduced_trainerData = transformer.transform(flattened_trainerData.cpu())
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
        if len(cluster_centers) != self.K:
            #Make a random selection
            self.selected_idx = np.random.choice(len(self.data_points), self.K, replace=False)
            return True
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
        self.select_data()
        total_score = 0
        chosen_points = self.data_points[self.selected_idx]
        chosen_reduced_points = torch.tensor(np.array(self.reduced_images[self.selected_idx]))
        chosen_labels = self.labels[self.selected_idx]
        #Uncertainty score, Diversity score and Loss score
        # Compute loss score
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(chosen_points)
            loss = self.loss_fn(outputs, chosen_labels)
            loss_score = loss.item()

        # Compute uncertainty score
        outputs = nn.Softmax(dim=1)(outputs)
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()

        # Compute diversity score
        diversity_score = 0
        for i in range(len(chosen_reduced_points)):
            point = chosen_reduced_points[i]
            if not isinstance(point, torch.Tensor):
                point = torch.tensor(point)
            #Compute the min. distance of point to any of self.trained_clusters
            min_distance = torch.min(torch.linalg.norm(point - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
            dist_self = torch.linalg.norm(point - chosen_reduced_points, dim=1)
            dist_self[i] = float('inf')
            min_dist_self = torch.min(dist_self).item()
            diversity_score += min_dist_self
        diversity_score /= len(chosen_reduced_points)
        diversity_score /= self.avg_l2norm #Normalization
        diversity_score /= 2
        
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
        self.reduced_images = transformer.fit_transform(flattened_images.cpu())
        self.reduced_trainerData = transformer.transform(flattened_trainerData.cpu())
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
        cluster_centers = kmeans.cluster_centers_
        if len(cluster_centers) != self.K:
            #Make a random selection
            self.selected_idx = np.random.choice(len(self.data_points), self.K, replace=False)
            return True
        for cluster in range(self.K):
            #Retrieve indices and images of the cluster
            cluster_indices = np.where(cluster_labels == cluster)[0]
            
            #Compute an uncertainty score
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(self.data_points[cluster_indices])
                outputs = nn.Softmax(dim=1)(outputs).cpu()
                unc_scores = (-torch.sum(outputs * torch.log(outputs + 1e-9), dim=1)).numpy()

            sorted_indices = np.argsort(unc_scores)
            self.selected_idx.append(cluster_indices[sorted_indices[0]])

        assert len(self.selected_idx) == self.K, "Number of selected indices must equal K"
        return True

    def data_value(self):
        self.select_data()
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
        outputs = nn.Softmax(dim=1)(outputs)
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()

        # Compute diversity score
        diversity_score = 0
        for i in range(len(chosen_reduced_points)):
            point = chosen_reduced_points[i]
            if not isinstance(point, torch.Tensor):
                point = torch.tensor(point)
            #Compute the min. distance of point to any of self.trained_clusters
            min_distance = torch.min(torch.linalg.norm(point - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
            dist_self = torch.linalg.norm(point - chosen_reduced_points, dim=1)
            dist_self[i] = float('inf')
            min_dist_self = torch.min(dist_self).item()
            diversity_score += min_dist_self
        diversity_score /= len(chosen_reduced_points)
        diversity_score /= self.avg_l2norm #Normalization
        diversity_score /= 2

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
        self.reduced_images = transformer.fit_transform(flattened_images.cpu())
        self.reduced_trainerData = transformer.transform(flattened_trainerData.cpu())
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
        self.select_data()
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
        outputs = nn.Softmax(dim=1)(outputs)
        uncertainty_score = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1).mean().item()

        # Compute diversity score
        diversity_score = 0
        for i in range(len(chosen_reduced_points)):
            point = chosen_reduced_points[i]
            if not isinstance(point, torch.Tensor):
                point = torch.tensor(point)
            #Compute the min. distance of point to any of self.trained_clusters
            min_distance = torch.min(torch.linalg.norm(point - self.trained_clusters, dim=1)).item()
            diversity_score += min_distance
            dist_self = torch.linalg.norm(point - chosen_reduced_points, dim=1)
            dist_self[i] = float('inf')
            min_dist_self = torch.min(dist_self).item()
            diversity_score += min_dist_self
        diversity_score /= len(chosen_reduced_points)
        diversity_score /= self.avg_l2norm #Normalization
        diversity_score /= 2

        total_score = self.a1 * loss_score + self.a2 * uncertainty_score + self.a3 *  diversity_score
        self.value = total_score
        return self.value
    
class MultiRandomValuation(MultiDataValuation):
    def __init__(self, model: nn.Module, data_points, labels, trainer_data):
        super().__init__(model, data_points, labels, trainer_data)
        self.value = None
        
    def dim_reduction(self):
        return True
    
    def select_data(self):
        return True

    def data_value(self):
        self.value = torch.rand(1).item()
        return self.value
    
class MultiEntropyValuation(MultiDataValuation):
    def __init__(self, model: nn.Module, data_points, labels, trainer_data, num_clusters):
        super().__init__(model, data_points, labels, trainer_data)
        self.K = num_clusters
        self.value = None
        self.dim_reduction()
    
    def dim_reduction(self):
        #Reduce dimension by random projection
        flattened_images = self.data_points.reshape(self.data_points.shape[0], -1)
        flattened_trainerData = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        transformer = SparseRandomProjection(n_components='auto', eps=0.5, random_state=0)
        self.reduced_images = transformer.fit_transform(flattened_images.cpu())
        self.reduced_trainerData = transformer.transform(flattened_trainerData.cpu())
        return True
    
    def select_data(self):
        # K-means clustering
        self.selected_idx = []
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.reduced_images)
        cluster_labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        #Loop through each individual cluster
        cluster_centers = kmeans.cluster_centers_
        if len(cluster_centers) != self.K:
            #Make a random selection
            self.selected_idx = np.random.choice(len(self.data_points), self.K, replace=False)
            return True
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
        self.select_data()
        chosen_points = self.data_points[self.selected_idx]
        self.model.eval()
        with torch.no_grad():
            output = self.model(chosen_points)
            output = nn.Softmax(dim=1)(output)
            entropy = (-torch.sum(output * torch.log(output + 1e-9), dim=1)).mean()
        self.value = entropy.item()
        return self.value
    
    
class MultiCoreSetValuation(MultiDataValuation):
    def __init__(self, model: nn.Module, data_points, labels, trainer_data, num_clusters):
        super().__init__(model, data_points, labels, trainer_data)
        self.value = None
        self.min_distances = None
        self.already_selected = []
        self.n_obs = self.data_points.shape[0]
        self.N = num_clusters
        self.dim_reduction() 
        
    def dim_reduction(self):
        #Reduce dimension by random projection
        flattened_images = self.data_points.reshape(self.data_points.shape[0], -1)
        flattened_trainerData = self.trainer_data.reshape(self.trainer_data.shape[0], -1)
        transformer = SparseRandomProjection(n_components='auto', eps=0.5, random_state=0)
        self.reduced_images = transformer.fit_transform(flattened_images.cpu())
        self.reduced_trainerData = transformer.transform(flattened_trainerData.cpu())
        self.all_features = np.vstack([self.reduced_trainerData, self.reduced_images])
        self.already_selected = [i for i in range(self.reduced_trainerData.shape[0])]
        return True

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.

        Args:
        cluster_centers: indices of cluster centers
        only_new: only calculate distance for newly selected points and update
            min_distances.
        rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                                if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.all_features[cluster_centers]
            dist = pairwise_distances(self.all_features, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def data_value(self):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
        model: model with scikit-like API with decision_function implemented
        already_selected: index of datapoints already selected
        N: batch size

        Returns:
        indices of points selected to minimize distance to cluster centers
        """

        self.update_distances(self.already_selected, only_new=False, reset_dist=False)
        new_batch = []

        for _ in range(self.N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
        # New examples should not be in already selected since those points
        # should have min_distance of zero to a cluster center.
            self.update_distances([ind], only_new=True, reset_dist=False)
            self.already_selected.append(ind)
            new_batch.append(ind)
        return max(self.min_distances)[0]