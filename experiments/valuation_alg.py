import torch
from torch import nn
from valuation_template import SingleDataValuation


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


