import torch.nn as nn
import torch

class SingleDataValuation:
    def __init__(self, model: nn.Module, data_point, label):
        self.model = model
        self.data_point = data_point
        self.label = label
        self.value = None

    def data_value(self):
        raise NotImplementedError("Subclasses must implement this method")

class MultiDataValuation:
    def __init__(self, model: nn.Module, data_points, labels):
        self.model = model
        self.data_points = data_points
        self.labels = labels
        self.selected_indices = None
        self.value = None

    def select_data(self):
        raise NotImplementedError("Subclasses must implement this method")

    def data_value(self):
        raise NotImplementedError("Subclasses must implement this method")