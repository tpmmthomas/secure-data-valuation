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
trainset = torchvision.datasets.CIFAR10(root='./data', train=False,transform=transform,download=True)
# Randomly select 1000 images as Bob's dataset
indices = random.sample(range(len(trainset)), N)
selected_images = np.array([trainset[i][0].numpy() for i in indices])
selected_labels = np.array([trainset[i][1]  for i in indices])

# Save images and labels separately
torch.save(torch.tensor(selected_images), 'data/selected_images.pth')
torch.save(torch.tensor(selected_labels), 'data/selected_labels.pth')
torch.save(model.state_dict(), "data/model.pth")
#Setup the model
data = torch.load("data/selected_images.pth")
torch.onnx.export(model,               # model being run
                data,                   # model input (or a tuple for multiple inputs)
                model_path,            # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=10,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['input'],   # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                            'output' : {0 : 'batch_size'}})

py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "public" #Bob can see this
py_run_args.output_visibility = "hashed/public" #This hash is given to Bob
py_run_args.param_visibility = "private"
print("Generating settings")
res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
assert res
# await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
print("Compiling circuit")
res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
assert res
# print("Generating srs")
# res = ezkl.get_srs(settings_path)
print("Setup here")
res = ezkl.setup(
    compiled_model_path,
    vk_path,
    pk_path,
    )

async def main(N,DIM,K,M):
    # Create the model and save it in the data directory.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=False,transform=transform,download=True)
    # Randomly select 1000 images as Bob's dataset
    indices = random.sample(range(len(trainset)), N)
    selected_images = np.array([trainset[i][0].numpy() for i in indices])
    selected_labels = np.array([trainset[i][1]  for i in indices])

    # Save images and labels separately
    torch.save(torch.tensor(selected_images), 'data/selected_images.pth')
    torch.save(torch.tensor(selected_labels), 'data/selected_labels.pth')
    # Set up challenge protocol for the desired dimension
    with open('cp_overall2.circom', 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "component main" in line and "DistanceProof" in line:
            lines[i] = f"component main {{public [points,d, commitX, commitY]}} = DistanceProof({K}, {DIM}, 32);\n"
            break
    else:
        # If no matching line is found, append the desired line.
        lines.append(f"component main {{public [points,d, commitX, commitY]}} = DistanceProof({K}, {DIM}, 32);\n")
    with open('cp_overall2.circom', 'w') as f:
        f.writelines(lines)
    code = os.system("circom cp_overall2.circom --r1cs --wasm --sym")
    assert code == 0, "Error in circom command"
    os.system("snarkjs groth16 setup cp_overall2.r1cs data/pot18_final.ptau data/cp_0000.zkey")
    os.system('echo "random" | snarkjs zkey contribute data/cp_0000.zkey data/cp_0001.zkey --name="1st Contributor Name" -v')
    os.system("snarkjs zkey export verificationkey data/cp_0001.zkey data/verification_key.json")
    #Set up the multi_point)val protocol
    # Update multi_point_val.mpc with the new values for data points and features
    mp_spdz_file = os.path.join("..", "..", "MP-SPDZ", "Programs", "Source", "multi_point_val.mpc")
    with open(mp_spdz_file, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "n" in line and "Number of data points" in line:
            lines[i] = f"n  = {K} # Number of data points\n"
        if "m" in line and "Number of features per data point" in line:
            lines[i] = f"m = {DIM} # Number of features per data point.\n"
    with open(mp_spdz_file, "w") as f:
        f.writelines(lines)
    #Compile it
    os.system("cd ../../MP-SPDZ/ && ./compile.py multi_point_val -R 64")
    #Repeat experiment 5 times
    print("Start exp")
    # Build the command. The '/usr/bin/time -v' prefix will output detailed resource usage.
    command = f"/usr/bin/time -v python launcher_5.py --model_name {model_name} --N {N} --DIM {DIM} --K {K} --M {M}"
    start = time.time()
    result = subprocess.run(
        command,
        shell=True,
        universal_newlines=True
    )
    elapsed = time.time() - start
    return elapsed

# N_tests = [500,1000,5000,10000]
# time_N = []
# for N in N_tests:
#     elapsed = asyncio.run(main(N,50,20,20))
#     time_N.append(elapsed)
#     fprint(f"Time for N={N}: {elapsed}", "results/exp5.txt")
    
DIM_tests = [20,50,100,500]
time_DIM = []
for DIM in DIM_tests:
    elapsed = asyncio.run(main(1000,DIM,20,20))
    time_DIM.append(elapsed)
    fprint(f"Time for DIM={DIM}: {elapsed}", "results/exp5.txt")
    
K_tests = [10,20,50,100]
time_K = []
for K in K_tests:
    elapsed = asyncio.run(main(1000,50,K,20))
    time_K.append(elapsed)
    fprint(f"Time for K={K}: {elapsed}", "results/exp5.txt")
    
M_tests = [10,20,50,100]
time_M = []
for M in M_tests:
    elapsed = asyncio.run(main(1000,50,20,M))
    time_M.append(elapsed)
    fprint(f"Time for M={M}: {elapsed}", "results/exp5.txt")
