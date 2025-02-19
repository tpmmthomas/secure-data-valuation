from valuation_alg_convex import * 
from sklearn.metrics import accuracy_score, classification_report
from data import get_dataset, split_dataset, add_noise
from models import get_convex_model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import tqdm
import copy
import random
import os

for seed in range(1,6):
    
    DATASET = "adult"
    MODEL = "logistic"

    def fprint(msg):
        print(msg)
        with open(f"results/exp2_{DATASET}_{MODEL}_{seed}.txt", "a") as f:
            f.write(msg + "\n")

    #Create the file and clear it 
    with open(f"results/exp2_{DATASET}_{MODEL}_{seed}.txt", "w") as f:
        pass

    dataset = get_dataset(DATASET)
    train_data, remain_data = split_dataset(dataset, 1000,11000)
    model = get_convex_model(MODEL,seed)

    #Assume we have 10 batches of data
    batches = []
    total = 11000
    for _ in range(10):
        batch, remain_data = split_dataset(remain_data, 1000, total-1000)
        # var = random.random() * 0.3 + 0.01
        # batch = add_noise(batch,var)
        batches.append(batch)
        total -= 1000
    assert len(remain_data) == 1000
    test_data = remain_data

    #Trian the model with train_data
    # train_data = add_noise(train_data, 0.5)
    x_train = np.array([x[0] for x in train_data])
    y_train = np.array([x[1] for x in train_data])
    x_test = np.array([x[0] for x in test_data])
    y_test = np.array([x[1] for x in test_data])
    model.fit(x_train,y_train)
    y_pred_log = model.predict(x_test)
    accuracy_init = accuracy_score(y_test, y_pred_log)
        


    #We gradually feed these 10 batches to training to see which is best.

    def train_and_evaluate(model, train_data, test_data):
        x_train = np.array([x[0] for x in train_data])
        y_train = np.array([x[1] for x in train_data])
        x_test = np.array([x[0] for x in test_data])
        y_test = np.array([x[1] for x in test_data])
        model.fit(x_train,y_train)
        y_pred_log = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred_log)
        return accuracy

    selected_batch_sequence = []
    acc_sequence = [accuracy_init]
    loss = nn.BCELoss()
    data_alice = [np.array(x[0]) for x in train_data]
    while len(selected_batch_sequence) < len(batches):
        best_batch = None
        best_score = None
        for i in range(len(batches)):
            if i in selected_batch_sequence:
                continue
            data_batch = [np.array(x[0]) for x in batches[i]]
            label_batch = [np.eye(2)[x[1]] for x in batches[i]]
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

    selected_batch_sequence = []
    acc_sequence = [accuracy_init]
    loss = nn.BCELoss()
    data_alice = [np.array(x[0]) for x in train_data]
    while len(selected_batch_sequence) < len(batches):
        best_batch = None
        best_score = None
        for i in range(len(batches)):
            if i in selected_batch_sequence:
                continue
            data_batch = [np.array(x[0]) for x in batches[i]]
            label_batch = [np.eye(2)[x[1]] for x in batches[i]]
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
            
            
    
    selected_batch_sequence = []
    acc_sequence = [accuracy_init]
    loss = nn.BCELoss()
    data_alice = [np.array(x[0]) for x in train_data]
    while len(selected_batch_sequence) < len(batches):
        best_batch = None
        best_score = None
        for i in range(len(batches)):
            if i in selected_batch_sequence:
                continue
            data_batch = [np.array(x[0]) for x in batches[i]]
            label_batch = [np.eye(2)[x[1]] for x in batches[i]]
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

    
    selected_batch_sequence = []
    acc_sequence = [accuracy_init]
    loss = nn.BCELoss()
    data_alice = [np.array(x[0]) for x in train_data]
    while len(selected_batch_sequence) < len(batches):
        best_batch = None
        best_score = None
        for i in range(len(batches)):
            if i in selected_batch_sequence:
                continue
            data_batch = [np.array(x[0]) for x in batches[i]]
            label_batch = [np.eye(2)[x[1]] for x in batches[i]]
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


    
    selected_batch_sequence = []
    acc_sequence = [accuracy_init]
    loss = nn.BCELoss()
    data_alice = [np.array(x[0]) for x in train_data]
    while len(selected_batch_sequence) < len(batches):
        best_batch = None
        best_score = None
        for i in range(len(batches)):
            if i in selected_batch_sequence:
                continue
            data_batch = [np.array(x[0]) for x in batches[i]]
            label_batch = [np.eye(2)[x[1]] for x in batches[i]]
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


    
    selected_batch_sequence = []
    acc_sequence = [accuracy_init]
    loss = nn.BCELoss()
    data_alice = [np.array(x[0]) for x in train_data]
    while len(selected_batch_sequence) < len(batches):
        best_batch = None
        best_score = None
        for i in range(len(batches)):
            if i in selected_batch_sequence:
                continue
            data_batch = [np.array(x[0]) for x in batches[i]]
            label_batch = [np.eye(2)[x[1]] for x in batches[i]]
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