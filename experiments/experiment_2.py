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
import copy
import os

torch.manual_seed(0)


DATASET = "cifar10"
MODEL = "resnet18"
LR = 1e-4

def fprint(msg):
    print(msg)
    with open(f"results/exp2_{DATASET}_{MODEL}.txt", "a") as f:
        f.write(msg + "\n")

#Create the file and clear it 
with open(f"results/exp2_{DATASET}_{MODEL}.txt", "w") as f:
    pass


dataset = get_dataset(DATASET)
train_data, remain_data = split_dataset(dataset, 100,600)
model = get_model(MODEL).cuda()

#Assume we have 10 batches of data
batches = []
total = 600
for _ in range(10):
    batch, remain_data = split_dataset(remain_data, 50, total-50)
    batches.append(batch)
    total -= 50
assert len(remain_data) == 100
test_data = remain_data

#Trian the model with train_data

optimizer = optim.SGD(model.parameters(), lr=LR)
num_epochs = 10

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
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, label in test_data:
        data = data.unsqueeze(0).cuda()
        label = torch.tensor([label]).cuda()
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy_init = correct / total
    


#We gradually feed these 10 batches to training to see which is best.
current_model_dict = copy.deepcopy(model.state_dict())

def train_and_evaluate(model, train_data, test_data):
    optimizer = optim.SGD(model.parameters(), lr=LR)
    num_epochs = 10

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
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label_one_hot)
            loss.backward()
            optimizer.step()
            
    #Evaluate the accuracy of the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_data:
            data = data.unsqueeze(0).cuda()
            label = torch.tensor([label]).cuda()
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

model.load_state_dict(current_model_dict)
selected_batch_sequence = []
acc_sequence = [accuracy_init]
loss = nn.CrossEntropyLoss()
data_alice = [np.array(x[0]) for x in train_data]
while len(selected_batch_sequence) < len(batches):
    best_batch = None
    best_score = None
    for i in range(len(batches)):
        if i in selected_batch_sequence:
            continue
        data_batch = [np.array(x[0]) for x in batches[i]]
        label_batch = [np.eye(10)[x[1]] for x in batches[i]]
        val = MultiKMeansValuation(model, data_batch, label_batch, data_alice, loss, 10, 0.3, 0.3, 0.4)
        dv = val.data_value()
        if best_score is None or dv > best_score:
            best_score = dv
            best_batch = i
    selected_batch_sequence.append(best_batch)
    acc = train_and_evaluate(model, batches[best_batch], test_data)
    data_alice += [np.array(x[0]) for x in batches[best_batch]]
    acc_sequence.append(acc)

fprint("KMeans Selected batch sequence:" + str(selected_batch_sequence))
fprint("KMeans Accuracy sequence:" + str(acc_sequence))

model.load_state_dict(current_model_dict)
selected_batch_sequence = []
acc_sequence = [accuracy_init]
loss = nn.CrossEntropyLoss()
data_alice = [np.array(x[0]) for x in train_data]
while len(selected_batch_sequence) < len(batches):
    best_batch = None
    best_score = None
    for i in range(len(batches)):
        if i in selected_batch_sequence:
            continue
        data_batch = [np.array(x[0]) for x in batches[i]]
        label_batch = [np.eye(10)[x[1]] for x in batches[i]]
        val = MultiUncKMeansValuation(model, data_batch, label_batch, data_alice, loss, 10, 0.3, 0.3, 0.4)
        dv = val.data_value()
        if best_score is None or dv > best_score:
            best_score = dv
            best_batch = i
    selected_batch_sequence.append(best_batch)
    acc = train_and_evaluate(model, batches[best_batch], test_data)
    data_alice += [np.array(x[0]) for x in batches[best_batch]]
    acc_sequence.append(acc)

fprint("KMeans+Unc Selected batch sequence:" + str(selected_batch_sequence))
fprint("KMeans+Unc Accuracy sequence:" + str(acc_sequence))
        
        
model.load_state_dict(current_model_dict)
selected_batch_sequence = []
acc_sequence = [accuracy_init]
loss = nn.CrossEntropyLoss()
data_alice = [np.array(x[0]) for x in train_data]
while len(selected_batch_sequence) < len(batches):
    best_batch = None
    best_score = None
    for i in range(len(batches)):
        if i in selected_batch_sequence:
            continue
        data_batch = [np.array(x[0]) for x in batches[i]]
        label_batch = [np.eye(10)[x[1]] for x in batches[i]]
        val = MultiSubModValuation(model, data_batch, label_batch, data_alice, loss, 10, 0.3, 0.3, 0.4)
        dv = val.data_value()
        if best_score is None or dv > best_score:
            best_score = dv
            best_batch = i
    selected_batch_sequence.append(best_batch)
    acc = train_and_evaluate(model, batches[best_batch], test_data)
    data_alice += [np.array(x[0]) for x in batches[best_batch]]
    acc_sequence.append(acc)

fprint("SubMod Selected batch sequence:" + str(selected_batch_sequence))
fprint("SubMod Accuracy sequence:" + str(acc_sequence))

model.load_state_dict(current_model_dict)
selected_batch_sequence = []
acc_sequence = [accuracy_init]
loss = nn.CrossEntropyLoss()
data_alice = [np.array(x[0]) for x in train_data]
while len(selected_batch_sequence) < len(batches):
    best_batch = None
    best_score = None
    for i in range(len(batches)):
        if i in selected_batch_sequence:
            continue
        data_batch = [np.array(x[0]) for x in batches[i]]
        label_batch = [np.eye(10)[x[1]] for x in batches[i]]
        val = MultiRandomValuation(model, data_batch, label_batch, data_alice)
        dv = val.data_value()
        if best_score is None or dv > best_score:
            best_score = dv
            best_batch = i
    selected_batch_sequence.append(best_batch)
    acc = train_and_evaluate(model, batches[best_batch], test_data)
    data_alice += [np.array(x[0]) for x in batches[best_batch]]
    acc_sequence.append(acc)

fprint("Random Selected batch sequence:" + str(selected_batch_sequence))
fprint("Random Accuracy sequence:" + str(acc_sequence))


model.load_state_dict(current_model_dict)
selected_batch_sequence = []
acc_sequence = [accuracy_init]
loss = nn.CrossEntropyLoss()
data_alice = [np.array(x[0]) for x in train_data]
while len(selected_batch_sequence) < len(batches):
    best_batch = None
    best_score = None
    for i in range(len(batches)):
        if i in selected_batch_sequence:
            continue
        data_batch = [np.array(x[0]) for x in batches[i]]
        label_batch = [np.eye(10)[x[1]] for x in batches[i]]
        val = MultiEntropyValuation(model, data_batch, label_batch, data_alice,10)
        dv = val.data_value()
        if best_score is None or dv > best_score:
            best_score = dv
            best_batch = i
    selected_batch_sequence.append(best_batch)
    acc = train_and_evaluate(model, batches[best_batch], test_data)
    data_alice += [np.array(x[0]) for x in batches[best_batch]]
    acc_sequence.append(acc)

fprint("Entropy Selected batch sequence:" + str(selected_batch_sequence))
fprint("Entropy Accuracy sequence:" + str(acc_sequence))


model.load_state_dict(current_model_dict)
selected_batch_sequence = []
acc_sequence = [accuracy_init]
loss = nn.CrossEntropyLoss()
data_alice = [np.array(x[0]) for x in train_data]
while len(selected_batch_sequence) < len(batches):
    best_batch = None
    best_score = None
    for i in range(len(batches)):
        if i in selected_batch_sequence:
            continue
        data_batch = [np.array(x[0]) for x in batches[i]]
        label_batch = [np.eye(10)[x[1]] for x in batches[i]]
        val = MultiCoreSetValuation(model, data_batch, label_batch, data_alice,10)
        dv = val.data_value()
        if best_score is None or dv > best_score:
            best_score = dv
            best_batch = i
    selected_batch_sequence.append(best_batch)
    acc = train_and_evaluate(model, batches[best_batch], test_data)
    data_alice += [np.array(x[0]) for x in batches[best_batch]]
    acc_sequence.append(acc)

fprint("CoreSet Selected batch sequence:" + str(selected_batch_sequence))
fprint("CoreSet Accuracy sequence:" + str(acc_sequence))