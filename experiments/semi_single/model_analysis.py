import torch.nn as nn
import importlib

def print_total_parameters(name,model: nn.Module):
    total_parameters = sum(param.numel() for param in model.parameters())
    print(f"{name}: Total number of parameters: {total_parameters}")

models = ['SVM', 'LeNet', 'AlexNet', 'ResNet18']


for model_name in models:
    model_module = importlib.import_module("model")
    ModelClass = getattr(model_module, model_name)
    model = ModelClass()
    print_total_parameters(model_name,model)
