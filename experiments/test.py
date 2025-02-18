from valuation_alg import * 
from data import get_dataset, split_dataset
from models import get_model
import torch
import numpy as np

torch.manual_seed(0)

dataset = get_dataset("mnist")
train_data, test_data = split_dataset(dataset, 100,100)
model = get_model("cnn")

data, label = train_data[0]
#Turn label to one-hot encoding
label = np.eye(10)[label]


#Loss function
loss = torch.nn.CrossEntropyLoss()

loss_val = SingleLossValuation(model, data, label, loss)
print(loss_val.data_value())

loss_val = SingleRandomValuation(model, data, label)
print(loss_val.data_value())

loss_val = SingleEntropyValuation(model, data, label)
print(loss_val.data_value())

data_bob = [np.array(x[0]) for x in train_data]
label_bob = [np.eye(10)[x[1]] for x in train_data]
data_alice = [np.array(x[0]) for x in test_data]
loss_val = MultiKMeansValuation(model, data_bob, label_bob, data_alice, loss, 10,0.3,0.3,0.4)
print(loss_val.data_value())

loss_val = MultiUncKMeansValuation(model, data_bob, label_bob, data_alice,loss, 10,0.3,0.3,0.4)
print(loss_val.data_value())

loss_val = MultiSubModValuation(model, data_bob, label_bob, data_alice,loss, 10,0.3,0.3,0.4)
print(loss_val.data_value())

loss_val = MultiRandomValuation(model, data_bob, label_bob, data_alice)
print(loss_val.data_value())

