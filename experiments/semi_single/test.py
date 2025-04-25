import time
import os
import importlib
import torch
import subprocess
import re
import torch.nn as nn

model_name = "TwoLayerCNN"
model_module = importlib.import_module("model")
ModelClass = getattr(model_module, model_name)
model = ModelClass()
# torch.save(model.state_dict(), "data/model.pth")

model.load_state_dict(torch.load('./data/model.pth'))
data = torch.load("data/data.pth")
label = torch.load("data/lbl.pth")
loss = nn.CrossEntropyLoss()

output = model(data)
_, label = label.max(dim=0)
label = label.unsqueeze(0)
print(label.shape)
print(output,label)
loss_value = loss(output, label).item()
print(loss_value)
