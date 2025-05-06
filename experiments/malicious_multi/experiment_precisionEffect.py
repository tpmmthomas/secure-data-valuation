import time
import os
import importlib
import torch
import subprocess
import re
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import ezkl
import os
import torchvision
import json
import random 
import numpy as np
import torchvision.transforms as transforms
from launcher import run_experiment
import asyncio
from plaintext import pipeline_total_score

#Specifying some path parameters
model_path = os.path.join('data','network.onnx')
compiled_model_path = os.path.join('data','network.compiled')
pk_path = os.path.join('data','test.pk')
vk_path = os.path.join('data','test.vk')
settings_path = os.path.join('data','settings.json')
cal_path = os.path.join('data',"calibration.json")
witness_path = os.path.join('data','witness.json')
data_path = os.path.join('data','input.json')
output_path = os.path.join('data','output.json')
label_path = os.path.join('data','label.json')
proof_path = os.path.join('data','test.pf')


# with open("results/exp5.txt", "w") as f:
#         pass
    
def fprint(msg,fn="results/exp4.txt"):
    print(msg)
    with open(fn, "a") as f:
        f.write(msg + "\n")
        
# Set torch seeds
torch.manual_seed(42)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

N = 1000 #Bob's dataset size
DIM = 50 # Dimension of the reduced dataset
K = 20 # Representative set size
M = 20 # Number of CP checks by Alice

transform = transforms.Compose(
    [transforms.ToTensor()])

#Only need to run once, comment out if not needed
# os.system("circom cp_overall2.circom --r1cs --wasm --sym")
# os.system("snarkjs powersoftau new bn128 18 data/pot18_0000.ptau -v")
# os.system('echo "random_string" | snarkjs powersoftau contribute data/pot18_0000.ptau data/pot18_0001.ptau --name="First contribution" -v')
# os.system("snarkjs powersoftau prepare phase2 data/pot18_0001.ptau data/pot18_final.ptau -v")
# os.system("snarkjs groth16 setup cp_overall2.r1cs data/pot18_final.ptau data/cp_0000.zkey")
# os.system('echo "random" | snarkjs zkey contribute data/cp_0000.zkey data/cp_0001.zkey --name="1st Contributor Name" -v')
# os.system("snarkjs zkey export verificationkey data/cp_0001.zkey data/verification_key.json")

model_name="LeNet"
model_module = importlib.import_module("model")
ModelClass = getattr(model_module, model_name)
model = ModelClass()


async def main(selected_images, selected_labels):
    N = 1000 #Bob's dataset size
    DIM = 50 # Dimension of the reduced dataset
    K = 20 # Representative set size
    M = 20 # Number of CP checks by Alice

    # Save images and labels separately
    torch.save(torch.tensor(selected_images), 'data/selected_images.pth')
    torch.save(torch.tensor(selected_labels), 'data/selected_labels.pth')
    print("Start exp")
    # Build the command. The '/usr/bin/time -v' prefix will output detailed resource usage.
    command = f"/usr/bin/time -v python launcher_5.py --model_name {model_name} --N {N} --DIM {DIM} --K {K} --M {M}"
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        universal_newlines=True
    )
    correct_score = pipeline_total_score(model_name)
    secure_score = None
    for line in result.stdout.splitlines():
        if "Final Valuation:" in line:
            m = re.search(r"Final Valuation:\s*([\d\.]+)", line)
            if m:
                secure_score = float(m.group(1))
    return correct_score, secure_score


 # Create the model and save it in the data directory.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# Create 10 non-overlapping datasets of size N
all_indices = list(range(len(trainset)))
random.shuffle(all_indices)
selected_images_list = []
selected_labels_list = []
for i in range(10):
    subset_indices = all_indices[i * N:(i + 1) * N]
    images = np.array([trainset[idx][0].numpy() for idx in subset_indices])
    labels = np.array([trainset[idx][1] for idx in subset_indices])
    selected_images_list.append(images)
    selected_labels_list.append(labels)
    
#Score each of the 10 datasets
secure_scores = []
correct_scores = []
for i in range(10):
    selected_images = selected_images_list[i]
    selected_labels = selected_labels_list[i]
    correct_score, secure_score = asyncio.run(main(selected_images, selected_labels))
    secure_scores.append(secure_score)
    correct_scores.append(correct_score)

# Create an index array for secure_scores sorted from highest to lowest
secure_sort_indices = sorted(range(len(secure_scores)), key=lambda i: secure_scores[i], reverse=True)

# Create an index array for correct_scores sorted from highest to lowest
correct_sort_indices = sorted(range(len(correct_scores)), key=lambda i: correct_scores[i], reverse=True)

# Optionally, print the sorted indices and their corresponding scores
print("Secure Score Indices (highest to lowest):", secure_sort_indices)
print("Secure Scores (sorted):", [secure_scores[i] for i in secure_sort_indices])
print("Correct Score Indices (highest to lowest):", correct_sort_indices)
print("Correct Scores (sorted):", [correct_scores[i] for i in correct_sort_indices])

if secure_sort_indices == correct_sort_indices:
    print("Both sorted indices arrays are equal.")
else:
    print("The sorted indices arrays differ.")