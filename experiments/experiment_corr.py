from valuation_alg import * 
from data import split_dataset, add_noise, create_challenging_batches_with_skew
import sys
sys.path.append('..')  # Add privade directory to path
from privade.data import get_dataset
from privade.models import get_model
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
DATASET = "cifar100"
TEACHERMODEL = "vgg8"
STUDENTMODEL = "cifarcnn5"
NAME = "formal1"
NUM_classes = 100

for seed in range(21,26):

    random.seed(seed)
    torch.manual_seed(seed)
    
    LR = 1e-5

    def fprint(msg):
        print(msg)
        with open(f"results/exp_corr_{DATASET}_{TEACHERMODEL}_{STUDENTMODEL}_{NAME}_{seed}.txt", "a") as f:
            f.write(msg + "\n")

    # Create/clear the results file.
    with open(f"results/exp_corr_{DATASET}_{TEACHERMODEL}_{STUDENTMODEL}_{NAME}_{seed}.txt", "w") as f:
        pass

    # Load dataset and split.
    dataset = get_dataset(DATASET)
    pretrain_size = 1500
    num_batch = 15
    per_batch = 300
    pool_size =  num_batch * per_batch
    test_size = 1500
    train_data, remain_data = split_dataset(dataset, pretrain_size,  pool_size + test_size + 10000)
    test_data, remain_data = split_dataset(remain_data, test_size, pool_size + 10000)
    teacher_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True).cuda()
    student_model = get_model(STUDENTMODEL,DATASET).cuda()

    random.shuffle(remain_data)
    batches, _ = create_challenging_batches_with_skew(
        dataset=remain_data,
        num_batch=num_batch,
        per_batch=per_batch,
        num_classes=10,   # CIFAR-10 OR MNIST
        degrade_prob=0.7
    )    
    assert len(batches) == num_batch
    assert len(test_data) == test_size

    # Train the model on the initial train_data (with noise) using a DataLoader.
    # train_data = add_noise(train_data, 0.5)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(teacher_model.parameters(), lr=LR)
    num_epochs = 50

    for epoch in range(num_epochs):
        teacher_model.train()
        # Optionally set BatchNorm layers to eval mode.
        for m in teacher_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        epoch_loss = 0.0
        for data, label in train_loader:
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            output = teacher_model(data)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        avg_loss = epoch_loss / len(train_data)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Evaluate on test set using a DataLoader.
    teacher_model.eval()
    correct = 0
    total_samples = 0
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()
            output = teacher_model(data)
            _, predicted = torch.max(output, 1)
            total_samples += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy_init = correct / total_samples
    
    from privade.distillation import train_distilled_model

    trained_student_model = train_distilled_model(teacher_model, student_model, train_loader, test_loader, epochs=50)

    data_alice = [np.array(x[0]) for x in train_data]
    loss_fn = nn.CrossEntropyLoss()
    scores_teacher = []
    scores_student = []
    for i in range(len(batches)):
        data_batch = [np.array(x[0]) for x in batches[i]]
        label_batch = [np.eye(NUM_classes)[x[1]] for x in batches[i]]
        teacherval = MultiMMSSValuation(teacher_model, data_batch, label_batch, data_alice, loss_fn,10, 0.2,0.1,0.7)
        scores_teacher.append(float(teacherval.data_value()))
        studentval = MultiMMSSValuation(student_model, data_batch, label_batch, data_alice, loss_fn,10, 0.2,0.1,0.7)
        scores_student.append(float(studentval.data_value() ))
    fprint(f"Teacher Score: {scores_teacher}")
    fprint(f"Student Score:  {scores_student}")
    
    student_rankings = sorted(range(len(batches)), key=lambda k: scores_student[k], reverse=True)
    teacher_rankings = sorted(range(len(batches)), key=lambda k: scores_teacher[k], reverse=True)
    
    #Calcule correlation
    from scipy.stats import spearmanr, kendalltau, weightedtau

    rho = spearmanr(scores_teacher, scores_student).correlation
    fprint(f"Correlation: {rho}")