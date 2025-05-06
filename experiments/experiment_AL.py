from valuation_alg import * 
from data import get_dataset, split_dataset, add_noise, create_challenging_batches_with_skew
from models import get_model
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
from torch.utils.data import DataLoader

# Set desired batch size for training and evaluation.
BATCH_SIZE = 10
DATASET = "cifar10"
MODEL = "resnet18"
NAME = "formal1"

for seed in range(21,26):

    random.seed(seed)
    torch.manual_seed(seed)
    
    LR = 1e-5

    def fprint(msg):
        print(msg)
        with open(f"results/exp2_{DATASET}_{MODEL}_{NAME}_{seed}.txt", "a") as f:
            f.write(msg + "\n")

    # Create/clear the results file.
    with open(f"results/exp2_{DATASET}_{MODEL}_{NAME}_{seed}.txt", "w") as f:
        pass

    # Load dataset and split.
    dataset = get_dataset(DATASET)
    pretrain_size = 100
    num_batch = 15
    per_batch = 300
    pool_size =  num_batch * per_batch
    test_size = 3000
    train_data, remain_data = split_dataset(dataset, pretrain_size,  pool_size + test_size + 10000)
    test_data, remain_data = split_dataset(remain_data, test_size, pool_size + 10000)
    model = get_model(MODEL).cuda()

    random.shuffle(remain_data)
    batches, _ = create_challenging_batches_with_skew(
        dataset=remain_data,
        num_batch=num_batch,
        per_batch=per_batch,
        num_classes=10,   # CIFAR-10 OR MNIST
        degrade_prob=0.6
    )    
    assert len(batches) == num_batch
    assert len(test_data) == test_size

    # # Create 10 batches from the remaining data.
    # batches = []
    # total = pool_size + test_size
    # for _ in range(num_batch):
    #     batch, remain_data = split_dataset(remain_data, per_batch, total - per_batch)
    #     var = random.random() * 0.1
    #     # batch = add_noise(batch, var)
    #     batches.append(batch)
    #     total -= per_batch
    # assert len(remain_data) == test_size
    # test_data = remain_data

    # Train the model on the initial train_data (with noise) using a DataLoader.
    # train_data = add_noise(train_data, 0.5)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        # Optionally set BatchNorm layers to eval mode.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        epoch_loss = 0.0
        for data, label in train_loader:
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        avg_loss = epoch_loss / len(train_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluate on test set using a DataLoader.
    model.eval()
    correct = 0
    total_samples = 0
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            _, predicted = torch.max(output, 1)
            total_samples += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy_init = correct / total_samples

    # Save the current model state for resetting between valuation methods.
    current_model_dict = copy.deepcopy(model.state_dict())
    
    # The train_and_evaluate function using DataLoader.
    def train_and_evaluate(model, train_data, test_data):
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        num_epochs = 10
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(num_epochs):
            model.train()
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            for data, label in train_loader:
                data, label = data.cuda(), label.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, label)
                loss.backward()
                optimizer.step()
                scheduler.step()
        model.eval()
        correct = 0
        total_samples = 0
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.cuda(), label.cuda()
                output = model(data)
                _, predicted = torch.max(output, 1)
                total_samples += label.size(0)
                correct += (predicted == label).sum().item()
        accuracy = correct / total_samples
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    # Define a helper function to run the valuation selection loop.
    def run_valuation(method_name, ValuationClass, extra_args_fn):
        # Reset the model state.
        model.load_state_dict(current_model_dict)
        selected_batch_sequence = []
        scores = []
        acc_sequence = [accuracy_init]
        # Define loss function locally.
        loss_fn = nn.CrossEntropyLoss()
        data_alice = [np.array(x[0]) for x in train_data]
        # Start with the initial training data.
        all_train_data_method = train_data.copy()
        # Determine extra arguments (if any) based on the method.
        extra_args = extra_args_fn(loss_fn)
        while len(selected_batch_sequence) < len(batches):
            best_batch = None
            best_score = None
            for i in range(len(batches)):
                if i in selected_batch_sequence:
                    continue
                data_batch = [np.array(x[0]) for x in batches[i]]
                label_batch = [np.eye(10)[x[1]] for x in batches[i]]
                val = ValuationClass(model, data_batch, label_batch, data_alice, *extra_args)
                dv = val.data_value()
                if best_score is None or dv > best_score:
                    best_score = dv
                    best_batch = i
            selected_batch_sequence.append(best_batch)
            scores.append(best_score)
            # Extend training data with the selected batch.
            all_train_data_method = all_train_data_method + batches[best_batch]
            # Evaluate the model on the test set.
            acc = train_and_evaluate(model, all_train_data_method, test_data)
            # Update data_alice with the newly added batch.
            data_alice += [np.array(x[0]) for x in batches[best_batch]]
            acc_sequence.append(acc)
        fprint(f"{method_name} Selected batch sequence:" + str(selected_batch_sequence))
        fprint(f"{method_name} Accuracy sequence:" + str(acc_sequence))
        # fprint(f"{method_name} Scores:" + str(scores))

    

    # Dictionary mapping method names to a tuple of (valuation class, lambda that returns extra args).
    # For methods that require a loss function, the lambda uses the local loss_fn.
    valuation_methods = {
        "Random":      (MultiRandomValuation,   lambda loss_fn: ()),
        "Entropy":     (MultiEntropyValuation,  lambda loss_fn: (10,)),
        "CoreSet":     (MultiCoreSetValuation,  lambda loss_fn: (10,)),
        "Ours":       (MultiMMSSValuation,     lambda loss_fn: (loss_fn,10, 0.2,0.1,0.7)),
    }

    # Loop over each valuation method.
    for method_name, (ValuationClass, extra_args_fn) in valuation_methods.items():
        run_valuation(method_name, ValuationClass, extra_args_fn)
