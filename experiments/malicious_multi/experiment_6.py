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
# trainset = torchvision.datasets.CIFAR10(root='./data', train=False,transform=transform,download=True)
# # Randomly select 1000 images as Bob's dataset
# indices = random.sample(range(len(trainset)), N)
# selected_images = np.array([trainset[i][0].numpy() for i in indices])
# selected_labels = np.array([trainset[i][1]  for i in indices])

# # Save images and labels separately
# torch.save(torch.tensor(selected_images), 'data/selected_images.pth')
# torch.save(torch.tensor(selected_labels), 'data/selected_labels.pth')
# torch.save(model.state_dict(), "data/model.pth")
# #Setup the model
# data = torch.load("data/selected_images.pth")
# torch.onnx.export(model,               # model being run
#                 data,                   # model input (or a tuple for multiple inputs)
#                 model_path,            # where to save the model (can be a file or file-like object)
#                 export_params=True,        # store the trained parameter weights inside the model file
#                 opset_version=10,          # the ONNX version to export the model to
#                 do_constant_folding=True,  # whether to execute constant folding for optimization
#                 input_names = ['input'],   # the model's input names
#                 output_names = ['output'], # the model's output names
#                 dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                             'output' : {0 : 'batch_size'}})

# py_run_args = ezkl.PyRunArgs()
# py_run_args.input_visibility = "public" #Bob can see this
# py_run_args.output_visibility = "hashed/public" #This hash is given to Bob
# py_run_args.param_visibility = "private"
# print("Generating settings")
# res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
# assert res
# # await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
# print("Compiling circuit")
# res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
# assert res
# # print("Generating srs")
# # res = ezkl.get_srs(settings_path)
# print("Setup here")
# res = ezkl.setup(
#     compiled_model_path,
#     vk_path,
#     pk_path,
#     )

async def main(selected_images, selected_labels):
    N = 1000 #Bob's dataset size
    DIM = 50 # Dimension of the reduced dataset
    K = 20 # Representative set size
    M = 20 # Number of CP checks by Alice

    # Save images and labels separately
    torch.save(torch.tensor(selected_images), 'data/selected_images.pth')
    torch.save(torch.tensor(selected_labels), 'data/selected_labels.pth')
    # Set up challenge protocol for the desired dimension
    # with open('cp_overall2.circom', 'r') as f:
    #     lines = f.readlines()
    # for i, line in enumerate(lines):
    #     if "component main" in line and "DistanceProof" in line:
    #         lines[i] = f"component main {{public [points,d, commitX, commitY]}} = DistanceProof({K}, {DIM}, 32);\n"
    #         break
    # else:
    #     # If no matching line is found, append the desired line.
    #     lines.append(f"component main {{public [points,d, commitX, commitY]}} = DistanceProof({K}, {DIM}, 32);\n")
    # with open('cp_overall2.circom', 'w') as f:
    #     f.writelines(lines)
    # code = os.system("circom cp_overall2.circom --r1cs --wasm --sym")
    # assert code == 0, "Error in circom command"
    # os.system("snarkjs groth16 setup cp_overall2.r1cs data/pot18_final.ptau data/cp_0000.zkey")
    # os.system('echo "random" | snarkjs zkey contribute data/cp_0000.zkey data/cp_0001.zkey --name="1st Contributor Name" -v')
    # os.system("snarkjs zkey export verificationkey data/cp_0001.zkey data/verification_key.json")
    # #Set up the multi_point)val protocol
    # # Update multi_point_val.mpc with the new values for data points and features
    # mp_spdz_file = os.path.join("..", "..", "MP-SPDZ", "Programs", "Source", "multi_point_val.mpc")
    # with open(mp_spdz_file, "r") as f:
    #     lines = f.readlines()
    # for i, line in enumerate(lines):
    #     if "n" in line and "Number of data points" in line:
    #         lines[i] = f"n  = {K} # Number of data points\n"
    #     if "m" in line and "Number of features per data point" in line:
    #         lines[i] = f"m = {DIM} # Number of features per data point.\n"
    # with open(mp_spdz_file, "w") as f:
    #     f.writelines(lines)
    # #Compile it
    # os.system("cd ../../MP-SPDZ/ && ./compile.py multi_point_val -R 64")
    #Repeat experiment 5 times
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