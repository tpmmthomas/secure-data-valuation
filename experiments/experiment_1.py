from valuation_alg import * 
from data import get_dataset, split_dataset
from models import get_model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm

torch.manual_seed(0)

DATASET = "cifar10"
MODEL = "resnet18"
LR = 1e-4

dataset = get_dataset(DATASET)
train_data, test_data = split_dataset(dataset, 100,100)
model = get_model(MODEL).cuda()

#Trian the model with train_data

optimizer = optim.SGD(model.parameters(), lr=LR)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for data, label in train_data:
        data = data.unsqueeze(0)
        label = [label]
        data, label = data.cuda(), torch.tensor(label).cuda()
        label_one_hot = F.one_hot(label, num_classes=10).float()
        # print(label_one_hot)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label_one_hot)
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")



# Experiment 1: Compare our rankings with data shapley 
data_to_be_valued = test_data[:50]
test_for_shapley = test_data[50:]
idx, lossvalue, shapvalue, randvalue, entrvalue = [], [], [], [], []
for i, (data, label) in tqdm.tqdm(enumerate(data_to_be_valued)):
    label = np.eye(10)[label]
    loss_val = SingleLossValuation(model, data, label, torch.nn.CrossEntropyLoss())
    # Remove current data point from data_to_be_valued as new variavble
    data_to_be_valued_excluding_i = data_to_be_valued[:i] + data_to_be_valued[i+1:]
    shap_val = SingleShapleyValuation(model, data, label,data_to_be_valued_excluding_i, test_for_shapley, 10, LR, 10, 'cuda', 10)
    rand_val = SingleRandomValuation(model, data, label)
    entr_val = SingleEntropyValuation(model, data, label)
    idx.append(i)
    lossvalue.append(loss_val.data_value())
    shapvalue.append(shap_val.data_value())
    randvalue.append(rand_val.data_value())
    entrvalue.append(entr_val.data_value())
    
df = pd.DataFrame({"Index": idx, "Loss": lossvalue, "Shapley": shapvalue, "Random": randvalue, "Entropy": entrvalue})
df.to_csv(f"results/experiment_1_{DATASET}_{MODEL}.csv", index=False)