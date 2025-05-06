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

# List of models to benchmark.
models = ['SVM', 'LeNet','MobileNetV2'] # 'AlexNet',SVM', 'LeNet','

with open("results/exp4.txt", "w") as f:
        pass
    
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

trainset = torchvision.datasets.CIFAR10(root='./data', train=False,transform=transform,download=True)
# Randomly select 1000 images as Bob's dataset
indices = random.sample(range(len(trainset)), N)
selected_images = np.array([trainset[i][0].numpy() for i in indices])
selected_labels = np.array([trainset[i][1]  for i in indices])

# Save images and labels separately
torch.save(torch.tensor(selected_images), 'data/selected_images.pth')
torch.save(torch.tensor(selected_labels), 'data/selected_labels.pth')

#Only need to run once, comment out if not needed
# os.system("circom cp_overall.circom --r1cs --wasm --sym")
# os.system("snarkjs powersoftau new bn128 18 data/pot18_0000.ptau -v")
# os.system('echo "random_string" | snarkjs powersoftau contribute data/pot18_0000.ptau data/pot18_0001.ptau --name="First contribution" -v')
# os.system("snarkjs powersoftau prepare phase2 data/pot18_0001.ptau data/pot18_final.ptau -v")
# os.system("snarkjs groth16 setup cp_overall.r1cs data/pot18_final.ptau data/cp_0000.zkey")
# os.system('echo "random" | snarkjs zkey contribute data/cp_0000.zkey data/cp_0001.zkey --name="1st Contributor Name" -v')
# os.system("snarkjs zkey export verificationkey data/cp_0001.zkey data/verification_key.json")

async def main():
    for model_name in models:
        total_times = []
        total_memory = []
        total_precision = []
        total_comm = []
        total_cluster = []
        total_cp_off = []
        total_cp_on = []
        total_zkp_off = []
        total_zkp_on = []
        total_mpc = []
        # Create the model and save it in the data directory.
        model_module = importlib.import_module("model")
        ModelClass = getattr(model_module, model_name)
        model = ModelClass()
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
        print("Generating srs")
        res = await ezkl.get_srs(settings_path)
        print("Setup here")
        res = ezkl.setup(
            compiled_model_path,
            vk_path,
            pk_path,
        )
        #Repeat experiment 5 times
        print("Start exp")
        for i in range(5):
            print("Running exp ", i)
            # Build the command. The '/usr/bin/time -v' prefix will output detailed resource usage.
            command = f"/usr/bin/time -v python launcher.py --model_name {model_name}"
            start = time.time()
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            elapsed = time.time() - start
            total_times.append(elapsed)
            correct_score = pipeline_total_score(model_name)
            secure_score = None
            # Parse result.stdout for timing information
            data_sent = 0
            for line in result.stdout.splitlines():
                if "Clustering time:" in line:
                    m = re.search(r"Clustering time:\s*([\d\.]+)", line)
                    if m:
                        total_cluster.append(float(m.group(1)))
                elif "Challenge Protocol online time:" in line:
                    m = re.search(r"Challenge Protocol online time:\s*([\d\.]+)", line)
                    if m:
                        total_cp_on.append(float(m.group(1)))
                elif "Challenge Protocol offline time:" in line:
                    m = re.search(r"Challenge Protocol offline time:\s*([\d\.]+)", line)
                    if m:
                        total_cp_off.append(float(m.group(1)))
                elif "ZKP for model inferences offline time:" in line:
                    m = re.search(r"ZKP for model inferences offline time:\s*([\d\.]+)", line)
                    if m:
                        total_zkp_off.append(float(m.group(1)))
                elif "ZKP for model inferences online time:" in line:
                    m = re.search(r"ZKP for model inferences online time:\s*([\d\.]+)", line)
                    if m:
                        total_zkp_on.append(float(m.group(1)))
                elif "MPC time:" in line:
                    m = re.search(r"MPC time:\s*([\d\.]+)", line)
                    if m:
                        total_mpc.append(float(m.group(1)))
                elif "Final Valuation:" in line:
                    m = re.search(r"Final Valuation:\s*([\d\.]+)", line)
                    if m:
                        secure_score = float(m.group(1))
                elif "Global data sent" in line:
                    m = re.search(r"Global data sent\s*=\s*([\d\.]+)", line)
                    if m:
                        data_sent += float(m.group(1))
            mem_usage = None
            for line in result.stderr.splitlines():
                print(line)
                if "Maximum resident set size" in line:
                    try:
                        mem_usage = int(line.split(":")[-1].strip())
                    except ValueError:
                        mem_usage = None
                    break
            if mem_usage is not None:
                total_memory.append(mem_usage)
            else:
                print("Warning: Memory usage not found in output.")
            
            # Calculate precision
            if secure_score is not None:
                print("Score", secure_score, correct_score)
                diff = abs(secure_score - correct_score)
                percentage_diff = (diff / correct_score) * 100
                total_precision.append(percentage_diff)
            else:
                print("Warning: Secure score not found in output.")
                
            # Calculate communication overhead
            def get_file_size(file_path):
                if os.path.exists(file_path):
                    return os.path.getsize(file_path) / (1024 * 1024)
                else:
                    assert False, f"File {file_path} does not exist."
            
            assert data_sent > 0, "Data sent should be greater than 0"
            data_sent += get_file_size('data/rep_points_to_submit.pth')
            # Estimate that each integer takes 4 bytes (32-bit int) and convert to MB.
            data_sent += (M * 4) / (1024 * 1024)
            data_sent += get_file_size("data/verification_key.json")
            data_sent += get_file_size("data/public.json") * M
            data_sent += get_file_size("data/proof.json") * M
            data_sent += get_file_size(proof_path)
            data_sent += get_file_size(vk_path)
            total_comm.append(data_sent)
        results_file = f"results/exp4_{model_name}.txt"
        fprint(f"times: {total_times}", results_file)
        fprint(f"memory: {total_memory}", results_file)
        fprint(f"precision: {total_precision}", results_file)
        fprint(f"comm: {total_comm}", results_file)
        fprint(f"cluster: {total_cluster}", results_file)
        fprint(f"cp_off: {total_cp_off}", results_file)
        fprint(f"cp_on: {total_cp_on}", results_file)
        fprint(f"zkp_off: {total_zkp_off}", results_file)
        fprint(f"zkp_on: {total_zkp_on}", results_file)
        fprint(f"mpc: {total_mpc}", results_file)
        fprint(f"correct score: {correct_score}", results_file)
        fprint(f"secure score: {secure_score}", results_file)
        print("Finish all!")


asyncio.run(main())