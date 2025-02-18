import torch.nn as nn
import torch

class SingleDataValuation:
    def __init__(self, model: nn.Module, data_point, label):
        self.model = model
        if isinstance(data_point, torch.Tensor):
            self.data_point = data_point
        else:
            self.data_point = torch.tensor(data_point)
        if isinstance(label, torch.Tensor):
            self.label = label
        else:
            self.label = torch.tensor(label)
        self.value = None
        if torch.cuda.is_available():
            self.data_point = self.data_point.cuda()
            self.label = self.label.cuda()

    def data_value(self):
        raise NotImplementedError("Subclasses must implement this method")

class MultiDataValuation:
    def __init__(self, model: nn.Module, data_points, labels, trainer_data):
        self.model = model
        if isinstance(data_points, torch.Tensor):
            self.data_points = data_points
        else:
            self.data_points = torch.tensor(data_points)
        if isinstance(labels, torch.Tensor):
            self.labels = labels
        else:
            self.labels = torch.tensor(labels)
        if isinstance(trainer_data, torch.Tensor):
            self.trainer_data = trainer_data
        else:
            self.trainer_data = torch.tensor(trainer_data)
        if torch.cuda.is_available():
            self.data_points = self.data_points.cuda()
            self.labels = self.labels.cuda()
            self.trainer_data = self.trainer_data.cuda()
        self.selected_indices = None
        self.value = None
        self.is_run = False
        

    def dim_reduction(self):
        raise NotImplementedError("Subclasses must implement this method")

    def select_data(self):
        raise NotImplementedError("Subclasses must implement this method")

    def data_value(self):
        raise NotImplementedError("Subclasses must implement this method")
